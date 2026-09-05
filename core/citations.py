"""Citation audit — the highest-value grounding check in this domain.

"Does the chunk id exist?" is not enough. The failure this corpus was written to
catch is a citation that is real, resolvable, **and wrong**: quoting the £520
EC 261 figure from a *delay* rule to justify compensation on a *cancellation*. The
citation resolves, so it reads as rigorous, and it is false.

So there are five checks of increasing strength:

1. **Resolvability** — every `[domain:ID]` names a chunk that exists.
2. **Figure containment** — every monetary amount and duration appears in a chunk
   cited in the same section.
3. **Phone containment** — every phone number comes from the contact fixture,
   never from the model.
4. **Kind matching** — a cancellation does not rest on a DELAY chunk.
5. **Category completeness** — an entitlements document states all six categories,
   including the ones the corpus does not cover.

Ported from tripsure's `advisor/tools_extra.py:audit_citations`, which earned each
of its exemptions against live output. Two are worth keeping in mind while reading:
a flight price carries `source`/`fetched_at` instead of a citation and is exempt,
but that exemption does not apply in a compensation context, or "you are owed £520
[source: …]" would smuggle a policy claim through as tool data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from core.policy import PolicyChunk

_CITE_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_\-]*):([A-Za-z0-9_\-]+)\]")
_MONEY_RE = re.compile(
    r"(?:[£$€]\s?([\d,]+(?:\.\d{1,2})?))"
    r"|(?:\b([\d,]+(?:\.\d{1,2})?)\s?(?:GBP|EUR|USD|SGD)\b)"
)
_DURATION_RE = re.compile(
    r"\b(\d{1,3})[\s-]*(hours?|hrs?|minutes?|mins?|nights?|days?)\b", re.I
)
_HEADING_RE = re.compile(
    r"^\s{0,3}(?:"
    r"#{1,6}\s+"              # ## Compensation
    r"|\*\*[A-Z]"             # **Compensation**
    r"|\d{1,2}[.)]\s+[A-Z]"   # 4) Cash compensation
    r"|[-*]\s+\*\*[A-Z]"      # - **Meals**
    r")",
    re.M,
)
_PHONE_RE = re.compile(r"\+\d[\d\s().\-]{6,}\d")
#: Tool-sourced data carries its own provenance (G8) rather than a policy chunk. A
#: flight price is a fetched fact; a compensation amount is a policy claim. Only
#: the second must trace to the corpus.
_PROVENANCE_RE = re.compile(r"source\s*[:=]|fetched[_\s]*at", re.I)
_COMPENSATION_CTX = re.compile(r"compensat|owed|entitle|payable|reimburs", re.I)

NOT_COVERED = "NOT COVERED BY POLICY CORPUS"
REQUIRED_CATEGORIES = (
    "rebooking", "accommodation", "meals", "compensation", "refund", "contact",
)


@dataclass
class Audit:
    ok: bool = True
    unresolved_citations: list[str] = field(default_factory=list)
    uncited_money: list[str] = field(default_factory=list)
    uncited_durations: list[str] = field(default_factory=list)
    uncited_phones: list[str] = field(default_factory=list)
    kind_mismatches: list[str] = field(default_factory=list)
    missing_categories: list[str] = field(default_factory=list)

    def report(self) -> str:
        if self.ok:
            return "PASS — every citation resolves, every figure is corpus-backed."
        lines = ["FAIL"]
        for label, items in (
            ("Unresolved / invented citations", self.unresolved_citations),
            ("Figures with no supporting cited chunk", self.uncited_money),
            ("Durations with no supporting cited chunk", self.uncited_durations),
            ("Phone numbers absent from the contact fixture", self.uncited_phones),
            ("Citation contradicts the disruption kind", self.kind_mismatches),
            ("Required categories missing entirely", self.missing_categories),
        ):
            if items:
                lines.append(f"- {label}:")
                lines.extend(f"    - {item}" for item in items)
        return "\n".join(lines)


def _sections(text: str) -> list[str]:
    """Split on headings / bold labels; the whole document if there are none.

    Sectioning is what makes figure containment meaningful: a citation three
    paragraphs away does not support a number, but a document-wide check would
    accept it.
    """
    idxs = [m.start() for m in _HEADING_RE.finditer(text)]
    if not idxs:
        return [text]
    if idxs[0] != 0:
        idxs = [0] + idxs
    return [text[a:b] for a, b in zip(idxs, idxs[1:] + [len(text)])]


def _phone_digits(contacts: dict | None) -> str:
    """Every phone number the contact source knows, digits only.

    A number in generated text that is not in here did not come from a lookup,
    which means a model produced it from memory. Taking the contacts as an
    argument means a test can audit against its own directory rather than the
    project fixture.
    """
    if contacts is None:
        from core.sources import sources
        contacts = _all_contacts(sources())
    raw = json.dumps(contacts, default=str)
    return "".join(re.sub(r"\D", "", tok) for tok in _PHONE_RE.findall(raw))


def _all_contacts(src) -> dict:
    """Pull every carrier the source will answer for. Used only by the auditor."""
    from config import DATA_DIR
    try:
        return json.loads((DATA_DIR / "airline_contacts.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _num(raw: str) -> str:
    plain = raw.replace(",", "")
    return plain.rstrip("0").rstrip(".") if "." in plain else plain


def audit_citations(
    text: str,
    disruption_kind: str = "cancelled",
    *,
    require_categories: bool = False,
    chunks: tuple[PolicyChunk, ...] | None = None,
    contacts: dict | None = None,
) -> Audit:
    """Audit a generated document against the live corpus. Pure function.

    `require_categories` applies only to the ENTITLEMENTS document. Demanding all
    six of every file reports the impact analysis as failing for "missing
    compensation" — a false alarm, and false alarms train you to ignore the
    checker, which is worse than not having one.
    """
    if chunks is None:
        from core.sources import sources
        chunks = sources().policy.chunks()
    index = {c.chunk_id: c for c in chunks}
    audit = Audit()

    # (1) Resolvability.
    for match in _CITE_RE.finditer(text):
        cid = f"{match.group(1)}:{match.group(2)}"
        if cid not in index:
            audit.unresolved_citations.append(cid)

    fixture_digits = _phone_digits(contacts)
    kind = (disruption_kind or "").strip().lower()
    opposite = "DELAY" if kind in {"cancelled", "canceled", "diverted"} else "CANCEL"

    for section in _sections(text):
        cited = [f"{m.group(1)}:{m.group(2)}" for m in _CITE_RE.finditer(section)]
        corpus_text = " ".join(
            index[c].text for c in cited if c in index
        ).lower().replace(",", "")

        # (2) Figure containment, with the tool-data exemption.
        for match in _MONEY_RE.finditer(section):
            amount = _num(match.group(1) or match.group(2) or "")
            if not amount or amount in corpus_text:
                continue
            start = section.rfind("\n", 0, match.start()) + 1
            end = section.find("\n", match.end())
            line = section[start: end if end != -1 else len(section)]
            tool_sourced = bool(_PROVENANCE_RE.search(line))
            policy_claim = bool(_COMPENSATION_CTX.search(line))
            if tool_sourced and not policy_claim:
                continue
            why = (
                "policy claim — provenance does not substitute for a citation"
                if policy_claim
                else "no cited chunk and no source/fetched_at provenance"
            )
            audit.uncited_money.append(
                f"{match.group(0).strip()} ({why}; cited here: {cited or 'NOTHING'})"
            )

        for match in _DURATION_RE.finditer(section):
            n, unit = match.group(1), match.group(2).lower().rstrip("s")
            if not re.search(rf"\b{n}[\s-]*{unit}", corpus_text):
                audit.uncited_durations.append(
                    f"{match.group(0).strip()} (cited here: {cited or 'NOTHING'})"
                )

        # (3) Phone containment.
        for match in _PHONE_RE.finditer(section):
            digits = re.sub(r"\D", "", match.group(0))
            if digits and digits not in fixture_digits:
                audit.uncited_phones.append(
                    f"{match.group(0).strip()} is not in data/airline_contacts.json"
                )

        # (4) Kind matching — the error that resolves and is still wrong.
        if "compensat" in section.lower():
            for cid in cited:
                if opposite in cid.split(":", 1)[-1].upper():
                    audit.kind_mismatches.append(
                        f"{cid} supports a {opposite} rule but the disruption is {kind!r}"
                    )

    # (5) Silence is a failure mode — an absent row reads as a clean answer.
    if require_categories:
        # Citations are stripped first. `[disruption_care:SQ-CANCEL-REFUND-01]`
        # lowercases to contain "refund", so scanning raw text let a document
        # delete its whole Refund row and pass on the strength of a citation.
        prose = _CITE_RE.sub(" ", text).lower()
        for category in REQUIRED_CATEGORIES:
            if category not in prose:
                audit.missing_categories.append(category)

    audit.ok = not any((
        audit.unresolved_citations, audit.uncited_money, audit.uncited_durations,
        audit.uncited_phones, audit.kind_mismatches, audit.missing_categories,
    ))
    return audit


def unresolved_citation_ids(
    citations: list[str], chunks: tuple[PolicyChunk, ...] | None = None
) -> list[str]:
    """Which of these chunk ids do not exist. Used by the plan compiler (C10)."""
    if chunks is None:
        from core.sources import sources
        chunks = sources().policy.chunks()
    known = {c.chunk_id for c in chunks}
    return [c for c in citations if c.strip("[]") not in known]


def contact_for(carrier_iata: str) -> dict:
    """A carrier's disruption desk, via the configured contact source.

    Kept as a function because several callers want the default source; anything
    needing its own directory should use `ContactSource.contact` directly.
    """
    from core.sources import sources
    return sources().contacts.contact(carrier_iata)
