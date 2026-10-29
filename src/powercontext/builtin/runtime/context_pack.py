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

"""Self-describing context packs and prefix-reuse deltas."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})
_DIGEST_PREFIX = "sha256:"
_HEX_DIGITS = frozenset("0123456789abcdef")


class ContextPackError(ValueError):
    """Raised when a context pack or delta violates the wire protocol."""


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encoded_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _unsigned_pack(items: tuple[bytes, ...]) -> dict[str, Any]:
    return {"schema": _PACK_SCHEMA, "items": _encoded_items(items)}


def _digest(items: tuple[bytes, ...]) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(_json_bytes(_unsigned_pack(items))).hexdigest()


def _validate_items(items: object) -> tuple[bytes, ...]:
    if not isinstance(items, tuple):
        raise ContextPackError("context pack items must be a bytes tuple")
    for item in items:
        if not isinstance(item, bytes):
            raise ContextPackError("context pack items must contain bytes")
    return items


def _decode_json(data: bytes) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise ContextPackError("context pack wire data must be bytes")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContextPackError("context pack wire data contains a duplicate field")
            result[key] = value
        return result

    try:
        payload = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("invalid context pack JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError("context pack wire data must be a JSON object")
    return payload


def _decode_items(values: object, field: str) -> tuple[bytes, ...]:
    if not isinstance(values, list):
        raise ContextPackError(f"{field} must be an array")
    items: list[bytes] = []
    for value in values:
        if not isinstance(value, str):
            raise ContextPackError(f"{field} must contain Base64 strings")
        try:
            item = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ContextPackError(f"{field} contains invalid Base64") from error
        if base64.b64encode(item).decode("ascii") != value:
            raise ContextPackError(f"{field} contains non-canonical Base64")
        items.append(item)
    return _validate_items(tuple(items))


def _validate_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != len(_DIGEST_PREFIX) + 64
        or not value.startswith(_DIGEST_PREFIX)
        or any(character not in _HEX_DIGITS for character in value[len(_DIGEST_PREFIX) :])
    ):
        raise ContextPackError(f"{field} must be a SHA-256 digest")
    return value


@dataclass(frozen=True)
class ContextPack:
    """A complete sequence of UTF-8 context chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        items = _validate_items(items)
        digest = _digest(items)
        byte_count = 0
        while True:
            wire = _json_bytes({**_unsigned_pack(items), "digest": digest, "byte_count": byte_count})
            encoded_count = len(wire)
            if encoded_count == byte_count:
                return cls(items, digest, byte_count)
            byte_count = encoded_count

    @classmethod
    def from_content(cls, content: str | None, *, chunk_bytes: int = 512) -> ContextPack:
        if content is not None and not isinstance(content, str):
            raise ContextPackError("context content must be a string or None")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if not content:
            return cls.build(())

        items: list[bytes] = []
        characters: list[str] = []
        item_bytes = 0
        for character in content:
            try:
                encoded = character.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ContextPackError("context content must be valid UTF-8") from error
            if len(encoded) > chunk_bytes:
                raise ContextPackError("chunk_bytes cannot contain a UTF-8 character")
            if characters and item_bytes + len(encoded) > chunk_bytes:
                items.append("".join(characters).encode("utf-8"))
                characters = []
                item_bytes = 0
            characters.append(character)
            item_bytes += len(encoded)
        if characters:
            items.append("".join(characters).encode("utf-8"))
        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        return _json_bytes({**_unsigned_pack(self.items), "digest": self.digest, "byte_count": self.byte_count})

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        payload = _decode_json(data)
        if set(payload) != _PACK_FIELDS:
            raise ContextPackError("context pack fields do not match the schema")
        if payload["schema"] != _PACK_SCHEMA:
            raise ContextPackError("unsupported context pack schema")
        items = _decode_items(payload["items"], "items")
        pack = cls.build(items)
        if payload["digest"] != pack.digest:
            raise ContextPackError("context pack digest mismatch")
        declared_count = payload["byte_count"]
        if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count != pack.byte_count:
            raise ContextPackError("context pack byte_count mismatch")
        return pack


@dataclass(frozen=True)
class ContextPackDelta:
    """A target suffix applied to a retained prefix of a base pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix_items = 0
        for base_item, target_item in zip(base.items, target.items):
            if base_item != target_item:
                break
            prefix_items += 1
        return cls(base.digest, target.digest, prefix_items, target.items[prefix_items:])

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _json_bytes(
            {
                "schema": _DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": _encoded_items(self.appended),
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        payload = _decode_json(data)
        if set(payload) != _DELTA_FIELDS:
            raise ContextPackError("context pack delta fields do not match the schema")
        if payload["schema"] != _DELTA_SCHEMA:
            raise ContextPackError("unsupported context pack delta schema")
        base_digest = _validate_digest(payload["base_digest"], "base_digest")
        target_digest = _validate_digest(payload["target_digest"], "target_digest")
        prefix_items = payload["prefix_items"]
        if isinstance(prefix_items, bool) or not isinstance(prefix_items, int) or prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        appended = _decode_items(payload["appended"], "appended")
        return cls(base_digest, target_digest, prefix_items, appended)


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Validate and apply a wire delta without relying on sender-side state."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("a context pack and delta are required")
    if delta.base_digest != base.digest:
        raise ContextPackError("context pack delta base digest mismatch")
    if (
        isinstance(delta.prefix_items, bool)
        or not isinstance(delta.prefix_items, int)
        or delta.prefix_items < 0
        or delta.prefix_items > len(base.items)
    ):
        raise ContextPackError("context pack delta prefix is out of bounds")
    appended = _validate_items(delta.appended)
    target = ContextPack.build((*base.items[: delta.prefix_items], *appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("context pack delta target digest mismatch")
    return target
