"""Policy corpus parsing and scoring. No configured paths, no I/O policy.

This module knows how to turn markdown into chunks and how to score a chunk
against a query. It does **not** know where the corpus lives or which corpus is
in use — that belongs to `core.sources.CorpusPolicySource`, so a live policy
source can reuse the scoring without inheriting a fixture path.

Chunks are delimited by an h2 of the form `## <domain>:<id>`, e.g.
`## disruption_care:SQ-CANCEL-ACCOM-01`. That id is what an option cites and what
the citation auditor resolves against, so it has to be stable across re-parses —
hence a hand-rolled parser over a fixed convention rather than a text splitter.

**Retrieval is lexical, not embedded, and that is a choice.** Embedding the
corpus needs an API key, which would put the offline gates behind a credential
and a bill and make every citation-audit rule untestable in CI. The corpus is a
few dozen short, densely-labelled chunks whose discriminating terms are literal
— "hotel accommodation overnight", "meal voucher", `SQ-CANCEL-ACCOM-01`. IDF over
unigrams handles those at least as well as a dense retriever and never returns a
plausible-but-unrelated neighbour. It is also deterministic, so a scenario
replays. A dense backend can be added as another `PolicySource`; the envelope's
`source` field is how callers tell which one answered.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_H2_RE = re.compile(r"^##\s+([A-Za-z_][A-Za-z0-9_\-]*):([A-Za-z0-9_\-]+)\s*$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str      # "<domain>:<id>"
    domain: str
    local_id: str
    text: str
    source_file: str


class ChunkParseError(ValueError):
    """Raised when two chunks share an id — caught at load, not at citation time."""


# ===============================================================
# Parsing
# ===============================================================
def _chunks_in(text: str, source_file: str) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    current: tuple[str, str] | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is None:
            return
        domain, local_id = current
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(PolicyChunk(
                chunk_id=f"{domain}:{local_id}", domain=domain, local_id=local_id,
                text=body, source_file=source_file,
            ))

    for line in text.splitlines():
        if match := _H2_RE.match(line):
            flush()
            current, buffer = (match.group(1), match.group(2)), []
            continue
        if current is not None:
            buffer.append(line)
    flush()
    return chunks


def load_chunks(policies_dir: Path) -> list[PolicyChunk]:
    """Load every `## domain:id` block under `policies_dir`.

    Preamble prose is discarded by design: the files open with a disclaimer and
    an explanation of the convention, neither of which is a retrievable
    entitlement.
    """
    root = Path(policies_dir)
    if not root.exists():
        return []

    seen: dict[str, str] = {}
    out: list[PolicyChunk] = []
    for md in sorted(root.glob("*.md")):
        for chunk in _chunks_in(md.read_text(encoding="utf-8"), source_file=md.name):
            if chunk.chunk_id in seen:
                raise ChunkParseError(
                    f"duplicate chunk_id {chunk.chunk_id!r}: first in "
                    f"{seen[chunk.chunk_id]}, then in {chunk.source_file}"
                )
            seen[chunk.chunk_id] = chunk.source_file
            out.append(chunk)
    return out


# ===============================================================
# Scoring
# ===============================================================
def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def idf(chunks: tuple[PolicyChunk, ...] | list[PolicyChunk]) -> dict[str, float]:
    """Inverse document frequency over a given corpus.

    Takes the corpus rather than reading a global one, so two different policy
    sources in the same process cannot contaminate each other's statistics.
    """
    if not chunks:
        return {}
    total = len(chunks)
    counts: Counter[str] = Counter()
    for chunk in chunks:
        for token in set(tokenize(f"{chunk.chunk_id} {chunk.text}")):
            counts[token] += 1
    return {t: math.log((total + 1) / (n + 1)) + 1.0 for t, n in counts.items()}


def score_chunk(
    query: str,
    chunk: PolicyChunk,
    corpus: tuple[PolicyChunk, ...] | list[PolicyChunk],
) -> float:
    """Relevance of one chunk to a query, in [0, ~1].

    The chunk id is scored alongside the body so that a query naming an id
    directly retrieves it. The critic re-retrieves by id to confirm a cited
    chunk says what a draft claims, and that check is useless if an exact id
    does not rank first.
    """
    wanted = set(tokenize(query))
    if not wanted:
        return 0.0
    weights = _cached_idf(tuple(corpus))
    present = set(tokenize(f"{chunk.chunk_id} {chunk.text}"))
    overlap = wanted & present
    if not overlap:
        return 0.0
    raw = sum(weights.get(t, 1.0) for t in overlap)
    denom = sum(weights.get(t, 1.0) for t in wanted) or 1.0
    return round(raw / denom, 4)


@lru_cache(maxsize=8)
def _cached_idf(corpus: tuple[PolicyChunk, ...]) -> dict[str, float]:
    """IDF is O(corpus) and would otherwise be recomputed for every scored chunk.

    Keyed on the corpus tuple itself, which works because `PolicyChunk` is a
    frozen dataclass and therefore hashable. An earlier version keyed on
    `id(corpus)`; that is a latent bug, since CPython reuses ids after garbage
    collection and a new corpus could inherit a stale one's statistics.
    """
    return idf(corpus)
