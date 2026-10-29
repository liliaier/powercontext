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

"""Query-driven, literal Memory evidence selection with exact byte accounting."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from powercontext.artifacts import ArtifactAddress, ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.runtime.errors import PreparedContextInvariantError
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY

_BEGIN_MARKER = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
_END_MARKER = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"
_TOKEN_RE = re.compile(r"[a-z]+|\d+")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"')\]]+)?")
_ABBREVIATIONS = {
    "al",
    "approx",
    "ca",
    "dr",
    "e.g",
    "eq",
    "etc",
    "fig",
    "figs",
    "i.e",
    "mr",
    "mrs",
    "ms",
    "no",
    "nos",
    "prof",
    "ref",
    "refs",
    "vs",
}
_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "of",
    "on",
    "or",
    "our",
    "properties",
    "property",
    "show",
    "shown",
    "shows",
    "than",
    "that",
    "the",
    "their",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "which",
    "will",
    "with",
    "would",
}


@dataclass(frozen=True)
class MemoryEntryAddress:
    """Identify one exact Memory entry version across Scope boundaries."""

    memory: ArtifactAddress
    entry_id: str
    entry_version_id: str


@dataclass(frozen=True)
class PreparedContextFragment:
    """A literal half-open character range from one exact Memory entry."""

    address: MemoryEntryAddress
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Document:
    address: MemoryEntryAddress
    text: str
    rank: int


@dataclass
class _Sentence:
    document: int
    index: int
    start: int
    end: int
    tokens: tuple[str, ...]
    stems: frozenset[str]
    concepts: frozenset[str]
    score: float = 0.0


@dataclass(frozen=True)
class _QueryTerms:
    stems: tuple[str, ...]
    weights: dict[str, float]


def build_evidence_context(
    *,
    query: str,
    max_bytes: int,
    current_scope_id: str | None,
    memory_candidates: Sequence[tuple[str, ArtifactRef | None, Sequence[MemoryHit]]],
) -> tuple[str | None, tuple[PreparedContextFragment, ...]]:
    """Select and render evidence without consulting anything beyond the supplied candidates."""

    documents = _evidence_documents(memory_candidates, current_scope_id)
    if not documents:
        return None, ()
    sentences = [
        sentence
        for document_index, document in enumerate(documents)
        for sentence in _document_sentences(document.text, document_index)
    ]
    if not sentences:
        return None, ()
    terms = _score_sentences(query, sentences)
    selected = _fit_sentences(documents, sentences, max_bytes)
    if not selected:
        fragment = _fit_partial_sentence(documents, sentences, terms, max_bytes)
        if fragment is None:
            return None, ()
        return _render((fragment,)), (fragment,)
    fragments = _merge_fragments(documents, selected)
    selected_identities = {id(sentence) for sentence in selected}
    fragment = _fit_partial_sentence(
        documents,
        [sentence for sentence in sentences if id(sentence) not in selected_identities],
        terms,
        max_bytes,
        included=fragments,
    )
    if fragment is not None:
        fragments = (*fragments, fragment)
    return _render(fragments), fragments


def _evidence_documents(
    groups: Sequence[tuple[str, ArtifactRef | None, Sequence[MemoryHit]]],
    current_scope_id: str | None,
) -> tuple[_Document, ...]:
    documents: list[_Document] = []
    seen: set[tuple[str, str, int, str, str]] = set()
    for group_scope_id, memory_ref, hits in groups:
        if hits and memory_ref is None:
            raise PreparedContextInvariantError("memory-ref-missing")
        scope_id = group_scope_id or current_scope_id
        if hits and not scope_id:
            raise PreparedContextInvariantError("memory-scope-missing")
        for hit in hits:
            if hit.memory_ref != memory_ref:
                raise PreparedContextInvariantError("memory-ref-mismatch")
            if not hit.entry_id.strip() or not hit.entry_version_id.strip() or not hit.text.strip():
                continue
            identity = (
                scope_id or "",
                hit.memory_ref.artifact_id,
                hit.memory_ref.revision,
                hit.entry_id,
                hit.entry_version_id,
            )
            if identity in seen:
                continue
            seen.add(identity)
            address = MemoryEntryAddress(
                memory=ArtifactAddress(scope_id=scope_id or "", artifact=hit.memory_ref),
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            documents.append(_Document(address=address, text=hit.text, rank=len(documents)))
    return tuple(documents)


def _document_sentences(text: str, document: int) -> list[_Sentence]:
    boundaries: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.end()
        if end >= len(text) or not text[end].isspace():
            continue
        previous = text[start : match.start()].rstrip()
        word_match = re.search(r"([A-Za-z](?:\.[A-Za-z])?|[A-Za-z]+)$", previous)
        word = word_match.group(1).lower() if word_match is not None else ""
        next_character = text[end:].lstrip()[:1]
        if (match.group() == "." and word in _ABBREVIATIONS) or (
            match.start() > start and text[match.start() - 1].isdigit() and next_character.isdigit()
        ):
            continue
        boundaries.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        boundaries.append((start, len(text)))

    sentences: list[_Sentence] = []
    for start, end in boundaries:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        tokens = _tokens(text[start:end])
        stems = frozenset(_stem(token) for token in tokens)
        sentences.append(
            _Sentence(
                document=document,
                index=len(sentences),
                start=start,
                end=end,
                tokens=tokens,
                stems=stems,
                concepts=frozenset(_concept(stem) for stem in stems),
            )
        )
    return sentences


def _tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii").lower()
    return tuple(_TOKEN_RE.findall(normalized))


def _stem(token: str) -> str:
    if token.isdigit() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    for suffix in ("ization", "ational", "fulness", "ousness", "iveness"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    for suffix in ("ingly", "edly", "ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if len(stem) > 3 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
    for suffix in ("ically", "ality", "ivity", "ment", "ness", "able", "ible"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _concept(stem: str) -> str:
    return _CONCEPTS.get(stem, stem)


def _make_concepts() -> dict[str, str]:
    groups = (
        ("cause", "causal", "induce", "induction", "inductive", "trigger", "promote"),
        ("increase", "increased", "elevate", "elevated", "upregulate", "enhance", "higher"),
        ("decrease", "decreased", "reduce", "reduced", "lower", "inhibit", "downregulate", "suppress"),
        ("associate", "associated", "correlate", "correlated", "related", "link", "linked"),
        ("cancer", "tumor", "tumour", "carcinoma", "neoplasm", "malignancy", "malignant"),
        ("cell", "cells", "cellular"),
        ("gene", "genes", "genetic", "genomic"),
        ("mutation", "mutations", "mutant", "mutated"),
        ("patient", "patients", "subject", "subjects", "participant", "participants"),
        ("child", "children", "childhood", "pediatric", "paediatric"),
        ("mouse", "mice", "murine"),
        ("therapy", "therapeutic", "treatment", "treated"),
        ("effective", "efficacy", "efficacious"),
        ("infection", "infected", "infectious"),
        ("expression", "expressed", "express"),
        ("response", "responsive", "respond"),
        ("risk", "risky", "predispose", "predisposition"),
        ("effect", "effects", "affect", "affected"),
        ("show", "shows", "demonstrate", "demonstrated", "indicate", "indicated"),
        ("property", "properties", "characteristic", "characteristics"),
    )
    return {_stem(variant): _stem(group[0]) for group in groups for variant in group}


_CONCEPTS = _make_concepts()


def _score_sentences(query: str, sentences: Sequence[_Sentence]) -> _QueryTerms:
    query_tokens = tuple(token for token in _tokens(query) if token not in _STOP_WORDS and len(token) > 1)
    query_stems = tuple(dict.fromkeys(_stem(token) for token in query_tokens))
    if not query_stems:
        query_stems = tuple(dict.fromkeys(_stem(token) for token in _tokens(query)))
    document_frequency = Counter(stem for sentence in sentences for stem in sentence.stems)
    count = len(sentences)
    weights = {
        stem: (math.log(1.0 + (count - document_frequency[stem] + 0.5) / (document_frequency[stem] + 0.5)) + 0.25)
        * (1.25 if stem.isdigit() else 1.0)
        for stem in query_stems
    }
    total_weight = sum(weights.values()) or 1.0
    for sentence in sentences:
        exact = {stem for stem in query_stems if stem in sentence.stems}
        exact_weight = sum(weights[stem] for stem in exact)
        related_weight = sum(
            weights[stem] * 0.35
            for stem in query_stems
            if stem not in exact and _concept(stem) in sentence.concepts
        )
        score = (exact_weight + related_weight) / total_weight
        sentence_stems = tuple(_stem(token) for token in sentence.tokens)
        phrase_pairs = sum(
            1
            for left, right in zip(query_stems, query_stems[1:])
            if any(pair == (left, right) for pair in zip(sentence_stems, sentence_stems[1:]))
        )
        if len(query_stems) > 1:
            score += 0.12 * phrase_pairs / (len(query_stems) - 1)
        sentence.score = score
    return _QueryTerms(stems=query_stems, weights=weights)


def _fit_sentences(
    documents: Sequence[_Document],
    sentences: Sequence[_Sentence],
    max_bytes: int,
) -> list[_Sentence]:
    selected: list[_Sentence] = []
    remaining = list(sentences)
    while remaining:
        def priority(sentence: _Sentence) -> tuple[float, float, int, int, int]:
            neighbor_score = max(
                (
                    other.score
                    for other in selected
                    if other.document == sentence.document and abs(other.index - sentence.index) == 1
                ),
                default=0.0,
            )
            adjusted = sentence.score + 0.12 * neighbor_score
            return (
                -adjusted,
                -sentence.score,
                documents[sentence.document].rank,
                sentence.index,
                len(documents[sentence.document].text[sentence.start : sentence.end].encode("utf-8")),
            )

        added = False
        for sentence in sorted(remaining, key=priority):
            trial = (*selected, sentence)
            if _rendered_bytes(_merge_fragments(documents, trial)) <= max_bytes:
                selected.append(sentence)
                remaining.remove(sentence)
                added = True
                break
        if not added:
            break
    return selected


def _merge_fragments(
    documents: Sequence[_Document],
    sentences: Sequence[_Sentence],
) -> tuple[PreparedContextFragment, ...]:
    order = {id(sentence): index for index, sentence in enumerate(sentences)}
    ranges: list[tuple[int, int, int, int]] = []
    for sentence in sorted(sentences, key=lambda item: (item.document, item.start)):
        if ranges:
            document, start, end, selected_order = ranges[-1]
            gap = documents[document].text[end : sentence.start]
            if document == sentence.document and sentence.start >= end and not gap.strip():
                ranges[-1] = (document, start, sentence.end, min(selected_order, order[id(sentence)]))
                continue
        ranges.append((sentence.document, sentence.start, sentence.end, order[id(sentence)]))
    return tuple(
        PreparedContextFragment(
            address=documents[document].address,
            start=start,
            end=end,
            text=documents[document].text[start:end],
        )
        for document, start, end, _ in sorted(ranges, key=lambda item: item[3])
    )


def _fit_partial_sentence(
    documents: Sequence[_Document],
    sentences: Sequence[_Sentence],
    terms: _QueryTerms,
    max_bytes: int,
    *,
    included: Sequence[PreparedContextFragment] = (),
) -> PreparedContextFragment | None:
    for sentence in sorted(sentences, key=lambda item: (-item.score, documents[item.document].rank, item.index)):
        document = documents[sentence.document]
        text = document.text[sentence.start : sentence.end]
        starts = [sentence.start]
        starts.extend(
            sentence.start + match.start()
            for match in _WORD_RE.finditer(text)
            if _stem_tokens(match.group()) in terms.stems
        )
        best: PreparedContextFragment | None = None
        best_key = (0, 0)
        for start in dict.fromkeys(starts):
            lower = start + 1
            upper = sentence.end
            candidate: PreparedContextFragment | None = None
            while lower <= upper:
                end = (lower + upper) // 2
                fragment = PreparedContextFragment(
                    address=document.address,
                    start=start,
                    end=end,
                    text=document.text[start:end],
                )
                if _rendered_bytes((*included, fragment)) <= max_bytes:
                    candidate = fragment
                    lower = end + 1
                else:
                    upper = end - 1
            if candidate is not None:
                coverage = len(set(terms.stems) & {_stem(token) for token in _tokens(candidate.text)})
                key = (coverage, len(candidate.text.encode("utf-8")))
                if key > best_key:
                    best = candidate
                    best_key = key
        if best is not None:
            return best
    return None


def _stem_tokens(token: str) -> str:
    tokens = _tokens(token)
    return _stem(tokens[0]) if tokens else ""


def _render(fragments: Sequence[PreparedContextFragment]) -> str:
    envelope = {
        "trust": "untrusted_history",
        "items": [
            {
                "citation": {
                    "scope_id": fragment.address.memory.scope_id,
                    "memory_ref": fragment.address.memory.artifact.model_dump(mode="json"),
                    "entry_id": fragment.address.entry_id,
                    "entry_version_id": fragment.address.entry_version_id,
                },
                "content": fragment.text,
                "start": fragment.start,
                "end": fragment.end,
            }
            for fragment in fragments
        ],
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


def _rendered_bytes(fragments: Sequence[PreparedContextFragment]) -> int:
    return len(_render(fragments).encode("utf-8"))
