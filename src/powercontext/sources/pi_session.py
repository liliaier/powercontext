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

"""Captured Pi v3 session-branch events with durable incremental ingestion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from powercontext.errors import SourceConflictError
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"


class PiSessionInput(BaseModel):
    """One complete UTF-8 Pi v3 JSONL session archive."""

    model_config = ConfigDict(frozen=True)

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event and its complete original Pi graph record."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured event whose identity is independent of its mutable payload."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _ParsedSession:
    session_id: str
    records: tuple[dict[str, JsonValue], ...]
    events: tuple[PiEventInput, ...]


def _identity(session_id: str, event_id: str) -> str:
    identity = json.dumps([session_id, event_id], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"pi:{hashlib.sha256(identity).hexdigest()}"


def _record_digest(record: dict[str, JsonValue]) -> str:
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _json_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _text_parts(content: JsonValue, *, include_tool_calls: bool) -> tuple[str, ...] | None:
    if content is None:
        return None
    if isinstance(content, str):
        return (content,)
    if not isinstance(content, list):
        raise ValueError("Pi visible content must be a string or a list of content blocks")

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Pi content blocks must be JSON objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("Pi text blocks must contain string text")
            parts.append(text)
        elif block_type == "toolCall" and include_tool_calls:
            name, call_id = block.get("name"), block.get("id")
            if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
                raise ValueError("Pi tool calls must contain non-empty string names and IDs")
            if "arguments" not in block:
                raise ValueError("Pi tool calls must contain arguments")
            arguments = json.dumps(
                block["arguments"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            parts.append(f"[tool_call {name} {call_id}] {arguments}")
    return tuple(parts)


def _join_visible_parts(parts: tuple[str, ...] | None) -> str | None:
    if not parts or not any(part != "" for part in parts):
        return None
    return "\n".join(parts)


def _visible_event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_type = record["type"]
    event_id = record["id"]
    parent_id = record["parentId"]
    assert isinstance(event_type, str) and isinstance(event_id, str)
    assert parent_id is None or isinstance(parent_id, str)

    role: str
    body: str | None
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            raise ValueError(f"Pi message {event_id!r} must contain a message object")
        message_role = message.get("role")
        if not isinstance(message_role, str) or not message_role:
            raise ValueError(f"Pi message {event_id!r} must contain a non-empty role")
        role = message_role
        if message_role == "toolResult":
            text = _join_visible_parts(_text_parts(message.get("content"), include_tool_calls=False))
            if text is None:
                return None
            tool_name, call_id = message.get("toolName"), message.get("toolCallId")
            if not isinstance(tool_name, str) or not tool_name or not isinstance(call_id, str) or not call_id:
                raise ValueError("Pi tool results must contain non-empty tool names and call IDs")
            body = f"[tool_result {tool_name} {call_id}] {text}"
        elif message_role == "bashExecution":
            command, output = message.get("command"), message.get("output")
            if not isinstance(command, str) or not isinstance(output, str):
                raise ValueError("Pi bash executions must contain string command and output")
            if command == "" and output == "":
                return None
            body = f"$ {command}\n{output}"
        else:
            body = _join_visible_parts(_text_parts(message.get("content"), include_tool_calls=True))
    elif event_type in {"compaction", "branch_summary"}:
        role = event_type
        summary = record.get("summary")
        if not isinstance(summary, str):
            raise ValueError(f"Pi {event_type} events must contain a string summary")
        body = summary or None
    elif event_type == "custom_message":
        role = "custom"
        body = _join_visible_parts(_text_parts(record.get("content"), include_tool_calls=False))
    else:
        return None

    if body is None:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _parse(value: PiSessionInput) -> _ParsedSession:
    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be strict UTF-8") from error

    records: list[dict[str, JsonValue]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_json_object, parse_constant=_reject_json_constant)
        except ValueError as error:
            raise ValueError(f"invalid Pi JSONL at line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Pi JSONL line {line_number} must contain an object")
        records.append(record)

    if not records:
        raise ValueError("Pi session archive must contain a session header")
    header = records[0]
    session_id = header.get("id")
    if header.get("type") != "session" or type(header.get("version")) is not int or header.get("version") != 3:
        raise ValueError("Pi session archive must start with a version 3 session header")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must contain a non-empty string ID")

    nodes = records[1:]
    if not nodes:
        raise ValueError("Pi session archive must contain at least one event node")
    by_id: dict[str, dict[str, JsonValue]] = {}
    for record in nodes:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Pi event nodes must contain non-empty string IDs")
        if event_id in by_id:
            raise ValueError(f"duplicate Pi event ID {event_id!r}")
        if not isinstance(record.get("type"), str) or not record.get("type"):
            raise ValueError(f"Pi event {event_id!r} must contain a non-empty type")
        if "parentId" not in record:
            raise ValueError(f"Pi event {event_id!r} must contain parentId")
        parent_id = record["parentId"]
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in by_id):
            raise ValueError(f"Pi event {event_id!r} has a missing or forward parent")
        by_id[event_id] = record

    leaf_id = value.leaf_id if value.leaf_id is not None else nodes[-1]["id"]
    if not isinstance(leaf_id, str) or leaf_id not in by_id:
        raise ValueError(f"Pi leaf {leaf_id!r} does not exist")

    selected: list[dict[str, JsonValue]] = []
    visited: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in visited:
            raise ValueError("Pi event graph contains a cycle")
        visited.add(current)
        record = by_id[current]
        selected.append(record)
        parent = record["parentId"]
        assert parent is None or isinstance(parent, str)
        current = parent
    selected.reverse()

    events: list[PiEventInput] = []
    for record in selected:
        event = _visible_event(session_id, record)
        if event is not None:
            events.append(event)
    return _ParsedSession(session_id=session_id, records=tuple(nodes), events=tuple(events))


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Validate a whole Pi graph and return visible events on its selected branch."""

    return _parse(value).events


class PiEventSourceAdapter:
    input_class = PiEventInput
    source_class = PiEventSource
    name = PI_EVENT_SOURCE_NAME

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=_identity(value.session_id, value.event_id),
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    projections=(PiEventTextEvidenceProjection(),),
)


def _checkpoint_events(checkpoint: JsonValue | None) -> dict[str, dict[str, str]]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"version", "events"} or checkpoint["version"] != 1:
        raise ValueError("invalid Pi session checkpoint")
    raw_events = checkpoint["events"]
    if not isinstance(raw_events, dict):
        raise ValueError("invalid Pi session checkpoint events")

    events: dict[str, dict[str, str]] = {}
    for identity, raw in raw_events.items():
        if not isinstance(identity, str) or not isinstance(raw, dict) or set(raw) != {
            "session_id",
            "event_id",
            "record_sha256",
        }:
            raise ValueError("invalid Pi session checkpoint entry")
        session_id, event_id, digest = raw["session_id"], raw["event_id"], raw["record_sha256"]
        if (
            not isinstance(session_id, str)
            or not session_id
            or not isinstance(event_id, str)
            or not event_id
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or identity != _identity(session_id, event_id)
        ):
            raise ValueError("invalid Pi session checkpoint entry")
        events[identity] = {"session_id": session_id, "event_id": event_id, "record_sha256": digest}
    return events


class PiSessionConnector:
    """Submit only previously uncommitted visible events from the selected branch."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        parsed = _parse(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))
        imported = _checkpoint_events(session.checkpoint)
        current_records = {_identity(parsed.session_id, record["id"]): record for record in parsed.records}

        for identity, previous in imported.items():
            if previous["session_id"] != parsed.session_id:
                continue
            record = current_records.get(identity)
            if record is None:
                raise ValueError(f"imported Pi event {previous['event_id']!r} was removed from history")
            if _record_digest(record) != previous["record_sha256"]:
                raise ValueError(f"imported Pi event {previous['event_id']!r} was rewritten")

        for event in parsed.events:
            identity = _identity(event.session_id, event.event_id)
            if identity in imported:
                continue
            try:
                await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            except SourceConflictError as error:
                raise ValueError(f"imported Pi event {event.event_id!r} was rewritten") from error
            imported[identity] = {
                "session_id": event.session_id,
                "event_id": event.event_id,
                "record_sha256": _record_digest(event.record),
            }

        checkpoint = {
            "version": 1,
            "events": {identity: imported[identity] for identity in sorted(imported)},
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
