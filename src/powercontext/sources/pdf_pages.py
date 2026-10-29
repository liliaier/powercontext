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

from __future__ import annotations

import hashlib
import io
import re
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime.models import SearchMemoryRequest
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization


class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")
        return value


class PDFPageInput(BaseModel):
    document_id: str
    document_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    page_number: Annotated[int, Field(ge=1)]
    text: str

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")
        return value


class PDFPageSource(Source):
    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSourceAdapter:
    input_class = PDFPageInput
    name = "pdf-page"
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        document_digest = hashlib.sha256(value.document_id.encode("utf-8")).hexdigest()
        return PDFPageSource(
            name=f"pdf:{document_digest}:{value.document_sha256}:{value.page_number}",
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    projections=(PDFPageTextEvidenceProjection(),),
)


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    document_sha256 = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(value.content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a password")
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=document_sha256,
                page_number=page_number,
                text=page.extract_text() or "",
            )
            for page_number, page in enumerate(reader.pages, start=1)
        )
    except Exception as error:
        raise ValueError("invalid or unreadable PDF") from error
    if not pages:
        raise ValueError("PDF contains no pages")
    return pages


def _query_variants(question: str) -> tuple[tuple[str, float], ...]:
    normalized = re.sub(r"\bfy\s*(\d{4})\b", r"\1", question, flags=re.IGNORECASE)
    normalized = re.sub(r"\b([A-Za-z]+)[’']s\b", r"\1", normalized)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    retained = []
    for sentence in sentences:
        stripped = sentence.strip()
        if re.match(r"^(?:give a response|respond to the question|answer in\b|if\b.*\bstate\s+0)", stripped, re.I):
            continue
        if re.match(r"^assume\b", stripped, re.I) and ":" not in stripped:
            continue
        if re.match(r"^(?:answer|respond)\b.*:", stripped, re.I):
            stripped = stripped.split(":", 1)[1].strip()
        if stripped:
            retained.append(stripped)
    core = " ".join(retained) or normalized
    variants: list[tuple[str, float]] = [(question, 0.6), (core, 1.0)]
    lowered = core.casefold()
    targeted = (
        (r"geograph|countries|regions", "business operations geographic regions countries divisions"),
        (r"legal|litigation", "legal proceedings material pending litigation"),
        (r"restructur", "restructuring impairment charges consolidated statements of income"),
        (
            r"ebitda|operating income|income statement|p\s*&\s*l",
            "consolidated statements of income operating profit net revenues",
        ),
        (
            r"d\s*&\s*a|depreciation|amortization|cash flow",
            "consolidated statements of cash flows depreciation amortization",
        ),
        (
            r"capex|capital expenditure",
            "consolidated statements of cash flows capital spending purchases of property plant and equipment",
        ),
        (
            r"capital[- ]intensive|capital intensity",
            "consolidated statements of income balance sheet cash flows capital expenditures "
            "property plant equipment total assets net sales",
        ),
        (
            r"segment|business segment|divisions?",
            "worldwide sales change by business segment organic sales acquisitions divestitures translation",
        ),
        (
            r"margin|profitab",
            "operating margin gross margin percent of sales operating profit net sales",
        ),
        (
            r"ratio|liquidity|working capital|current assets",
            "consolidated balance sheets current assets current liabilities cash and cash equivalents",
        ),
        (
            r"margin|gross profit|operating profit|return on",
            "consolidated statements of income net sales cost of sales gross profit operating income",
        ),
        (
            r"m\s*&\s*a|acquisition|divestiture|organic",
            "acquisitions divestitures organic sales growth translation",
        ),
        (r"balance sheet|pp\s*&\s*e|ppne", "consolidated balance sheet property plant and equipment net"),
    )
    additions = [replacement for pattern, replacement in targeted if re.search(pattern, lowered, re.I)]
    if additions:
        variants.append((f"{core} {' '.join(additions)}", 0.7))
        variants.extend((replacement, 0.9) for replacement in additions)
    if re.search(r"geograph|countries|regions", lowered):
        variants.append(("countries territories operations", 1.1))
    if re.search(r"ebitda|operating income", lowered):
        variants.append(("operating profit", 1.2))
    if re.search(r"d\s*&\s*a|depreciation|amortization", lowered):
        variants.append(("depreciation amortization", 1.1))
    if re.search(r"capex|capital expenditure", lowered):
        variants.append(("capital spending", 1.1))
    if re.search(r"capital[- ]intensive|capital intensity|total assets|return on assets", lowered):
        variants.append(("consolidated statement of income", 3.0))
        variants.append(("consolidated balance sheet", 3.0))
        variants.append(("consolidated statement of cash flows", 3.0))
        variants.append(("net sales operating income", 2.0))
        variants.append(("cash flows from operating activities", 2.0))
        variants.append(("total assets total liabilities", 2.0))
    if re.search(r"ratio|liquidity|working capital", lowered):
        variants.append(("consolidated balance sheets current assets current liabilities", 3.0))
    if re.search(r"margin|gross profit|operating profit|return on", lowered):
        variants.append(("consolidated statements of income net sales cost of sales gross profit", 3.0))
    if re.search(
        r"margin|profitab|operating income|gross profit|revenue|earnings|tax rate|effective tax",
        lowered,
    ):
        variants.append(("consolidated statements of income", 3.0))
    if re.search(r"balance sheet|liabilit|liabilities|deposit|borrowing|debt|securities", lowered):
        variants.append(("consolidated balance sheets", 3.0))
    if re.search(r"effective tax rate|tax rate", lowered):
        variants.append(("effective tax rate", 3.0))
    unique: dict[str, tuple[str, float]] = {}
    for text, weight in variants:
        key = " ".join(text.casefold().split())
        if key:
            unique.setdefault(key, (text, weight))
    return tuple(unique.values())


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    if limit < 0:
        raise ValueError("limit must not be negative")
    if limit == 0:
        return ()
    scores: dict[tuple[str, int, str, str], float] = {}
    citations: dict[tuple[str, int, str, str], MemoryCitation] = {}
    for query, weight in _query_variants(question):
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=max(limit, 32)))
        for rank, hit in enumerate(page.hits, start=1):
            citation = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            identity = (
                hit.memory_ref.artifact_id,
                hit.memory_ref.revision,
                hit.entry_id,
                hit.entry_version_id,
            )
            citations.setdefault(identity, citation)
            scores[identity] = scores.get(identity, 0.0) + weight / (60 + rank)
            if rank <= 2:
                scores[identity] += weight
    ordered = sorted(scores, key=lambda identity: (-scores[identity], identity))
    return tuple(citations[identity] for identity in ordered[:limit])


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "search_pdf_pages",
]
