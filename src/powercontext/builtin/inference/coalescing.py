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

"""Concurrent embedding request coalescing and successful-vector caching."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True, eq=False)
class _Batch:
    entries: tuple[_Entry, ...]
    usage: InferenceUsage | None = None


@dataclass(slots=True, eq=False)
class _Entry:
    text: str
    future: asyncio.Future[EmbeddingVector]
    batch: _Batch | None = None


class CoalescingEmbeddingModel:
    """Share in-flight text embeddings and cache successful vectors in LRU order."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_ms = coalesce_ms
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, _Entry] = {}
        self._queued: list[_Entry] = []
        self._flush_handle: asyncio.Handle | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Embed texts while charging each distinct awaited provider batch once."""
        texts = tuple(texts)
        if not texts:
            return EmbeddingResult(vectors=())

        cached_vectors: dict[str, EmbeddingVector] = {}
        entries: dict[str, _Entry] = {}
        for text in dict.fromkeys(texts):
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                cached_vectors[text] = cached
                continue
            entry = self._inflight.get(text)
            if entry is None:
                entry = _Entry(text=text, future=asyncio.get_running_loop().create_future())
                entry.future.add_done_callback(_consume_exception)
                self._inflight[text] = entry
                self._queued.append(entry)
            entries[text] = entry

        if self._queued and self._flush_handle is None:
            loop = asyncio.get_running_loop()
            if self._coalesce_ms == 0:
                self._flush_handle = loop.call_soon(self._flush)
            else:
                self._flush_handle = loop.call_later(self._coalesce_ms / 1000, self._flush)

        pending = tuple(entries.values())
        if pending:
            await asyncio.gather(*(asyncio.shield(entry.future) for entry in pending))

        batches = {entry.batch for entry in pending if entry.batch is not None}
        usage = _combine_usage(tuple(batch.usage for batch in batches if batch.usage is not None))
        vectors = tuple(
            cached_vectors[text] if text in cached_vectors else entries[text].future.result() for text in texts
        )
        return EmbeddingResult(vectors=vectors, usage=usage)

    def _flush(self) -> None:
        self._flush_handle = None
        queued, self._queued = self._queued, []
        for start in range(0, len(queued), self._max_batch_size):
            entries = tuple(queued[start : start + self._max_batch_size])
            batch = _Batch(entries=entries)
            for entry in entries:
                entry.batch = batch
            task = asyncio.create_task(self._run_batch(batch))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_batch(self, batch: _Batch) -> None:
        entries = batch.entries
        try:
            result = await self._model.embed(tuple(entry.text for entry in entries))
            if len(result.vectors) != len(entries):
                raise ValueError("embedding provider returned a different number of vectors than texts")
            batch.usage = result.usage
            for entry, vector in zip(entries, result.vectors, strict=True):
                self._cache_vector(entry.text, vector)
                entry.future.set_result(vector)
        except BaseException as exc:
            for entry in entries:
                if not entry.future.done():
                    entry.future.set_exception(exc)
        finally:
            for entry in entries:
                if self._inflight.get(entry.text) is entry:
                    del self._inflight[entry.text]

    def _cache_vector(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _consume_exception(future: asyncio.Future[EmbeddingVector]) -> None:
    if not future.cancelled():
        future.exception()


def _combine_usage(usages: tuple[InferenceUsage, ...]) -> InferenceUsage:
    return InferenceUsage(
        requests=sum(usage.requests for usage in usages),
        input_tokens=_sum_optional(tuple(usage.input_tokens for usage in usages)),
        output_tokens=_sum_optional(tuple(usage.output_tokens for usage in usages)),
    )


def _sum_optional(values: tuple[int | None, ...]) -> int | None:
    known = tuple(value for value in values if value is not None)
    return sum(known) if known else None


__all__ = ["CoalescingEmbeddingModel"]
