# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent Source dependency lifecycle for native Memory and Topic Memory."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENTS_TABLE,
    SOURCE_LIFECYCLE_NODES_TABLE,
)
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """Reject an invalid lifecycle binding, event, or resolution."""


class SourceLifecycleEvent(BaseModel):
    """One strictly validated, monotonically ordered Source state event."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or blank")  # noqa: TRY003
        return value


_NodeKind: TypeAlias = Literal["source", "memory", "topic"]
_Policy: TypeAlias = Literal["all", "any"]
_Reference: TypeAlias = SourceRef | MemoryCitation | ArtifactRef


@dataclass(frozen=True, slots=True)
class _Node:
    node_id: str
    kind: _NodeKind
    reference: _Reference
    parents: tuple[str, ...]
    policy: _Policy | None
    revision: int | None
    available: bool


@dataclass(frozen=True, slots=True)
class _EventRecord:
    event: SourceLifecycleEvent
    completed: bool


class SourceLifecycleApplication:
    """Select the persistent lifecycle service for one Scope."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, self._database, scope_id)

    async def unavailable_topic_refs(self, scope_id: str, /) -> frozenset[tuple[str, str, int]]:
        if self._database is None:
            return frozenset()
        return await self.for_scope(scope_id)._unavailable_topic_refs()


class ScopedSourceLifecycle:
    """Own dependency registration and retryable event propagation for one Scope."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self.scope_id = validate_scope_id(scope_id)

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        node_id = _node_id(node_id)
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source_ref must be a SourceRef")
        _positive_revision(revision)
        await self._validate_source_ref(source_ref)
        await self._register(
            _Node(
                node_id=node_id,
                kind="source",
                reference=source_ref,
                parents=(),
                policy=None,
                revision=revision,
                available=True,
            )
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        node_id = _node_id(node_id)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("citation must be a MemoryCitation")
        normalized_parents = _parents(parents, node_id)
        normalized_policy = _policy(policy)
        await self._validate_citation(citation)
        await self._register(
            _Node(
                node_id=node_id,
                kind="memory",
                reference=citation,
                parents=normalized_parents,
                policy=normalized_policy,
                revision=None,
                available=True,
            )
        )

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        node_id = _node_id(node_id)
        if not isinstance(artifact_ref, ArtifactRef) or artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("artifact_ref must reference topic-memory")
        normalized_parents = _parents(parents, node_id)
        normalized_policy = _policy(policy)
        await self._validate_artifact_ref(artifact_ref)
        await self._register(
            _Node(
                node_id=node_id,
                kind="topic",
                reference=artifact_ref,
                parents=normalized_parents,
                policy=normalized_policy,
                revision=None,
                available=True,
            )
        )

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event must be a SourceLifecycleEvent")
        database = self._require_database()
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            async with database.transaction() as connection:
                nodes = await _read_nodes(connection, self.scope_id)
                source = nodes.get(event.node_id)
                if source is None or source.kind != "source":
                    raise SourceLifecycleError(f"unknown Source: {event.node_id}")
                if source.revision != event.revision:
                    raise SourceLifecycleError("event revision does not match the registered Source")
                latest = await _latest_event(connection, self.scope_id, event.node_id)
                resume = False
                if latest is not None:
                    if event.sequence < latest.event.sequence:
                        raise SourceLifecycleError("event sequence is older than the latest Source event")
                    if event.sequence == latest.event.sequence:
                        if event != latest.event:
                            raise SourceLifecycleError("event sequence already names different content")
                        if latest.completed:
                            return sorted(node.node_id for node in nodes.values() if node.available)
                        resume = True
                    elif not latest.completed:
                        raise SourceLifecycleError("retry the interrupted latest Source event first")
                if not resume:
                    await connection.execute(
                        insert(SOURCE_LIFECYCLE_EVENTS_TABLE).values(
                            scope_id=self.scope_id,
                            node_id=event.node_id,
                            sequence=event.sequence,
                            revision=event.revision,
                            available=event.available,
                            reason=event.reason,
                            completed=False,
                        )
                    )
                previous_availability = {node_id: node.available for node_id, node in nodes.items()}
                nodes[source.node_id] = replace(source, available=event.available)
                availability = _evaluate(nodes)
                updated = {
                    node_id: replace(node, available=availability[node_id]) for node_id, node in nodes.items()
                }
                for node_id, node in updated.items():
                    if node.available == previous_availability[node_id]:
                        continue
                    await connection.execute(
                        update(SOURCE_LIFECYCLE_NODES_TABLE)
                        .where(
                            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id,
                            SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
                        )
                        .values(available=node.available)
                    )
                nodes = updated

            # The event and desired graph state commit before native writes. A
            # failure therefore leaves a resumable event rather than accepting
            # a replay that would incorrectly look like a new sequence.
            for node in nodes.values():
                if node.kind == "memory":
                    await self._sync_memory(node, reason=event.reason)

            async with database.transaction() as connection:
                await connection.execute(
                    update(SOURCE_LIFECYCLE_EVENTS_TABLE)
                    .where(
                        SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == self.scope_id,
                        SOURCE_LIFECYCLE_EVENTS_TABLE.c.node_id == event.node_id,
                        SOURCE_LIFECYCLE_EVENTS_TABLE.c.sequence == event.sequence,
                    )
                    .values(completed=True)
                )
            return sorted(node_id for node_id, node in nodes.items() if node.available)

    async def resolve(self, node_id: str, *, purpose: str) -> _Reference:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError(f"unknown resolve purpose: {purpose}")
        database = self._require_database()
        async with self._runtime._scope_operation(self.scope_id), database.transaction() as connection:
            nodes = await _read_nodes(connection, self.scope_id)
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError(f"unknown lifecycle node: {node_id}")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError(f"lifecycle node is unavailable for generation: {node_id}")
        return node.reference

    async def _register(self, candidate: _Node) -> None:
        database = self._require_database()
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            async with database.transaction() as connection:
                nodes = await _read_nodes(connection, self.scope_id)
                existing = nodes.get(candidate.node_id)
                if existing is not None:
                    if not _same_binding(existing, candidate):
                        raise SourceLifecycleError(f"lifecycle node binding conflicts: {candidate.node_id}")
                    selected = existing
                else:
                    missing = [parent for parent in candidate.parents if parent not in nodes]
                    if missing:
                        raise SourceLifecycleError(f"unregistered lifecycle parents: {', '.join(missing)}")
                    if candidate.kind != "source":
                        support = [nodes[parent].available for parent in candidate.parents]
                        available = all(support) if candidate.policy == "all" else any(support)
                        candidate = replace(candidate, available=available)
                    await connection.execute(
                        insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                            scope_id=self.scope_id,
                            node_id=candidate.node_id,
                            kind=candidate.kind,
                            reference_payload=candidate.reference.model_dump_json(),
                            parents_payload=json.dumps(candidate.parents, separators=(",", ":")),
                            policy=candidate.policy,
                            revision=candidate.revision,
                            available=candidate.available,
                        )
                    )
                    selected = candidate
            if selected.kind == "memory":
                await self._sync_memory(selected, reason="source lifecycle registration")

    async def _sync_memory(self, node: _Node, *, reason: str) -> None:
        citation = cast(MemoryCitation, node.reference)
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(citation.memory_ref.artifact_id)
            entry = next(
                (
                    value
                    for value in await service.entries(current)
                    if value.entry_id == citation.entry_id and value.entry_version_id == citation.entry_version_id
                ),
                None,
            )
            manifest = next(
                (value for value in current.content.manifest.entries if value.entry_id == citation.entry_id), None
            )
            if entry is None or manifest is None:
                raise SourceLifecycleError(f"registered Memory citation is no longer current: {node.node_id}")
            target = "active" if node.available else "inactive"
            if manifest.state == target:
                return
            if node.available:
                await service.reactivate(current, entries=(entry,), reason=reason)
            else:
                await service.forget(current, entries=(entry,), reason=reason)

    async def _validate_source_ref(self, reference: SourceRef) -> None:
        try:
            async with self._runtime._context(self.scope_id) as context:
                for source in await context.sources.list():
                    if context.sources.catalog.as_ref(source) == reference:
                        await context.sources.read(source)
                        return
        except Exception as error:
            raise SourceLifecycleError("SourceRef does not name readable captured evidence") from error
        raise SourceLifecycleError("SourceRef does not name readable captured evidence")

    async def _validate_citation(self, citation: MemoryCitation) -> None:
        try:
            async with self._runtime._context(self.scope_id) as context:
                await context.artifacts.memory.validate_citation(citation)
        except Exception as error:
            raise SourceLifecycleError("MemoryCitation does not name readable Memory evidence") from error

    async def _validate_artifact_ref(self, reference: ArtifactRef) -> None:
        try:
            await self._runtime._records().get_artifact_revision(
                self.scope_id,
                reference.family,
                reference.artifact_id,
                reference.revision,
            )
        except Exception as error:
            raise SourceLifecycleError("ArtifactRef does not name readable Topic Memory") from error

    async def _unavailable_topic_refs(self) -> frozenset[tuple[str, str, int]]:
        database = self._require_database()
        async with self._runtime._scope_operation(self.scope_id), database.transaction() as connection:
            nodes = await _read_nodes(connection, self.scope_id)
        return frozenset(
            (
                cast(ArtifactRef, node.reference).family,
                cast(ArtifactRef, node.reference).artifact_id,
                cast(ArtifactRef, node.reference).revision,
            )
            for node in nodes.values()
            if node.kind == "topic" and not node.available
        )

    def _require_database(self) -> AsyncDatabase:
        if self._database is None:
            raise SourceLifecycleError("Source lifecycle persistence is not configured")
        return self._database


async def _read_nodes(connection: AsyncConnection, scope_id: str) -> dict[str, _Node]:
    rows = (
        await connection.execute(
            select(SOURCE_LIFECYCLE_NODES_TABLE).where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
        )
    ).mappings()
    result: dict[str, _Node] = {}
    for row in rows:
        kind = cast(_NodeKind, row["kind"])
        payload = str(row["reference_payload"])
        reference: _Reference
        if kind == "source":
            reference = SourceRef.model_validate_json(payload)
        elif kind == "memory":
            reference = MemoryCitation.model_validate_json(payload)
        else:
            reference = ArtifactRef.model_validate_json(payload)
        node_id = str(row["node_id"])
        result[node_id] = _Node(
            node_id=node_id,
            kind=kind,
            reference=reference,
            parents=tuple(json.loads(str(row["parents_payload"]))),
            policy=cast(_Policy | None, row["policy"]),
            revision=cast(int | None, row["revision"]),
            available=bool(row["available"]),
        )
    return result


async def _latest_event(connection: AsyncConnection, scope_id: str, node_id: str) -> _EventRecord | None:
    row = (
        await connection.execute(
            select(SOURCE_LIFECYCLE_EVENTS_TABLE)
            .where(
                SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_EVENTS_TABLE.c.node_id == node_id,
            )
            .order_by(SOURCE_LIFECYCLE_EVENTS_TABLE.c.sequence.desc())
            .limit(1)
        )
    ).mappings().one_or_none()
    if row is None:
        return None
    return _EventRecord(
        event=SourceLifecycleEvent(
            node_id=str(row["node_id"]),
            revision=int(row["revision"]),
            sequence=int(row["sequence"]),
            available=bool(row["available"]),
            reason=str(row["reason"]),
        ),
        completed=bool(row["completed"]),
    )


def _evaluate(nodes: dict[str, _Node]) -> dict[str, bool]:
    availability: dict[str, bool] = {}
    visiting: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in availability:
            return availability[node_id]
        if node_id in visiting:
            raise SourceLifecycleError("lifecycle dependency graph contains a cycle")
        visiting.add(node_id)
        node = nodes[node_id]
        if node.kind == "source":
            value = node.available
        else:
            support = [visit(parent) for parent in node.parents]
            value = all(support) if node.policy == "all" else any(support)
        visiting.remove(node_id)
        availability[node_id] = value
        return value

    for node_id in nodes:
        visit(node_id)
    return availability


def _same_binding(left: _Node, right: _Node) -> bool:
    return (
        left.kind == right.kind
        and left.reference == right.reference
        and left.parents == right.parents
        and left.policy == right.policy
        and left.revision == right.revision
    )


def _node_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("node_id must be non-blank text")
    if len(value) > 255:
        raise SourceLifecycleError("node_id must contain at most 255 characters")
    return value


def _positive_revision(value: int) -> None:
    if type(value) is not int or value < 1:
        raise SourceLifecycleError("revision must be a positive integer")


def _parents(values: tuple[str, ...], node_id: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise SourceLifecycleError("parents must be a sequence of node identifiers")
    parents = tuple(values)
    if not parents:
        raise SourceLifecycleError("derived lifecycle nodes require at least one parent")
    normalized = tuple(_node_id(parent) for parent in parents)
    if node_id in normalized:
        raise SourceLifecycleError("lifecycle nodes cannot depend on themselves")
    if len(set(normalized)) != len(normalized):
        raise SourceLifecycleError("lifecycle parents must not contain duplicates")
    return tuple(sorted(normalized))


def _policy(value: str) -> _Policy:
    if value not in {"all", "any"}:
        raise SourceLifecycleError("policy must be 'all' or 'any'")
    return cast(_Policy, value)


__all__ = ["SourceLifecycleError", "SourceLifecycleEvent"]
