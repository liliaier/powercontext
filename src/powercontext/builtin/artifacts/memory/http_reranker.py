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

"""HTTP score-based reranking for coarse Memory search candidates."""

from __future__ import annotations

from math import isfinite

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score candidates in bounded HTTP batches and select across the full pool."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("rerank batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return globally ranked original ranks without modifying candidate hits."""

        candidate_count = len(candidates)
        if candidate_count < 1:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if type(limit) is not int or limit < 1 or limit > candidate_count:
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003

        scores: dict[int, int | float] = {}
        requests = 0
        for start in range(0, candidate_count, self._batch_size):
            batch = tuple(enumerate(candidates[start : start + self._batch_size], start=start + 1))
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [{"rank": rank, "text": candidate.text} for rank, candidate in batch],
                },
            )
            requests += 1
            self._raise_for_http_error(response)
            scores.update(self._validated_scores(response, {rank for rank, _ in batch}))

        selected_ranks = tuple(sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected_ranks,
            usage=InferenceUsage(requests=requests),
        )

    @staticmethod
    def _raise_for_http_error(response: httpx.Response) -> None:
        if response.is_success:
            return
        response.raise_for_status()
        # httpx raises only for 4xx/5xx; the reranker contract rejects every non-2xx response.
        raise httpx.HTTPStatusError(
            f"Reranker returned HTTP {response.status_code}",
            request=response.request,
            response=response,
        )

    @staticmethod
    def _validated_scores(response: httpx.Response, expected_ranks: set[int]) -> dict[int, int | float]:
        try:
            payload = response.json()
        except ValueError as error:
            raise ValueError("reranker response was not valid JSON") from error  # noqa: TRY003
        if not isinstance(payload, dict) or set(payload) != {"scores"}:
            raise ValueError("reranker response must contain exactly scores")  # noqa: TRY003
        rows = payload["scores"]
        if not isinstance(rows, list) or len(rows) != len(expected_ranks):
            raise ValueError("reranker scores must cover the request batch exactly")  # noqa: TRY003

        scores: dict[int, int | float] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"rank", "score"}:
                raise ValueError("reranker score records must contain exactly rank and score")  # noqa: TRY003
            rank = row["rank"]
            score = row["score"]
            if type(rank) is not int or rank not in expected_ranks or rank in scores:
                raise ValueError("reranker score ranks must cover the request batch exactly once")  # noqa: TRY003
            if type(score) not in (int, float) or (type(score) is float and not isfinite(score)):
                raise ValueError("reranker scores must be finite numbers")  # noqa: TRY003
            scores[rank] = score
        return scores


__all__ = ["HTTPMemoryReranker"]
