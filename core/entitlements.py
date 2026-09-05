"""What the airline owes, derived from the corpus — six categories, no model call.

The grounding layer existed from R2 but nothing surfaced it: an option showed a
chunk id like `disruption_care:SQ-CANCEL-REBOOK-01` and nothing a traveller could
read. A citation nobody can follow is decoration.

This is C1 of the context-compiler plan brought forward, in the same spirit as
`core.planner`: retrieval here is lexical and local, so building the full
entitlement picture costs nothing and needs no key.

Two selection rules do the work that a model was previously asked to get right,
and repeatedly did not:

**Carrier scoping.** A chunk is only usable if it is about this carrier or is
explicitly generic. `SQ-CANCEL-ACCOM-01` must not be quoted at a BA passenger.

**Kind matching.** A cancellation entitlement must rest on a CANCEL chunk, never
a DELAY one. This was the error the corpus was written to catch — the citation
resolves, so it reads as rigorous, and it is wrong. Here it is impossible: chunks
carrying the opposite kind are filtered out before ranking.

Everything produced is put through `audit_citations`, so the rendering is checked
against the corpus the same way a model's would be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.policy import PolicyChunk, score_chunk
from core.sources import Sources, sources
from models import DisruptionKind

#: The six categories a traveller needs an answer for, in the order they matter
#: when a flight has just been cancelled: (key, heading, query, id token).
#:
#: The id token exploits the corpus's own naming convention — `SQ-CANCEL-MEALS-01`,
#: `BA-CANCEL-REFUND-01`, `airline_rebooking:GENERIC-01`. Lexical scoring alone
#: picked the wrong chunk twice: a query about meal vouchers matched
#: `CONTACT-SQ-01`, because that chunk genuinely says "handles accommodation and
#: meal vouchers directly", and BA's refund fell back to the generic chunk while
#: `BA-CANCEL-REFUND-01` sat right there. A chunk whose id names the category is
#: about the category, and that is a stronger signal than word overlap.
CATEGORIES: tuple[tuple[str, str, str, str], ...] = (
    ("rebooking", "Getting you there",
     "cancellation free rebooking next available flight endorsement partner",
     "REBOOK"),
    ("accommodation", "Somewhere to stay",
     "hotel accommodation overnight stranded transfers nights", "ACCOM"),
    ("meals", "Food while you wait",
     "meal vouchers refreshments waiting hours", "MEALS"),
    ("compensation", "Cash compensation",
     "cash compensation payable conditions of carriage", "COMP"),
    ("refund", "Getting your money back",
     "refund unflown portion taxes surcharges instead of rebooking", "REFUND"),
    ("contact", "Who to talk to",
     "disruption desk contact telephone service desk", "CONTACT"),
)

NOT_COVERED = "NOT COVERED BY POLICY CORPUS"
_OPPOSITE = {
    DisruptionKind.CANCELLED: "DELAY",
    DisruptionKind.DIVERTED: "DELAY",
    DisruptionKind.DELAYED: "CANCEL",
}


@dataclass
class Entitlement:
    category: str
    label: str
    covered: bool
    text: str
    chunk_id: str | None = None
    source_file: str | None = None


@dataclass
class EntitlementsReport:
    carrier: str
    disruption_kind: str
    items: list[Entitlement] = field(default_factory=list)
    contact: dict = field(default_factory=dict)
    #: Present when the contact fixture carries a warning. It always does, and it
    #: always needs showing: the numbers are drama-range placeholders, and a
    #: number displayed without that caveat reads as verified.
    contact_warning: str = ""

    def by_category(self, category: str) -> Entitlement | None:
        return next((i for i in self.items if i.category == category), None)

    @property
    def covered_count(self) -> int:
        return sum(1 for i in self.items if i.covered)

    @property
    def chunk_ids(self) -> list[str]:
        return [i.chunk_id for i in self.items if i.chunk_id]

    def as_markdown(self) -> str:
        """The entitlements document, rendered rather than written by a model.

        Sectioned per category, because the citation auditor checks figures
        against the citations in the *same* section — a compensation figure under
        an accommodation heading is reported as uncited, correctly.
        """
        lines = [
            f"# What {self.carrier} owes you — {self.disruption_kind}",
            "",
            "> Demo fixture corpus. Confirm anything you intend to act on with "
            "the carrier.",
            "",
        ]
        for item in self.items:
            lines += [f"## {item.label} ({item.category})", ""]
            if item.covered and item.chunk_id:
                lines += [f"{item.text} [{item.chunk_id}]", ""]
            else:
                lines += [
                    f"{NOT_COVERED} — confirm this with the carrier directly.", ""
                ]
        if self.contact.get("disruption_desk"):
            lines += [
                "## Contact (contact)",
                "",
                f"Disruption desk: {self.contact['disruption_desk']} "
                f"[{self.by_category('contact').chunk_id}]"
                if self.by_category("contact")
                and self.by_category("contact").chunk_id
                else f"Disruption desk: {self.contact['disruption_desk']}",
                "",
            ]
        return "\n".join(lines)


def _usable(chunk: PolicyChunk, carrier: str, kind: DisruptionKind) -> bool:
    """Is this chunk about the right carrier, and the right kind of disruption?"""
    local = chunk.local_id.upper()
    prefix = local.split("-", 1)[0]
    if prefix not in (carrier.upper(), "GENERIC", "CONTACT"):
        return False
    # `CONTACT-SQ-01` puts the carrier second, so check the whole id too.
    if prefix == "CONTACT" and carrier.upper() not in local and "GENERIC" not in local:
        return False
    opposite = _OPPOSITE.get(kind)
    return not (opposite and opposite in local)


def _best(
    query: str, carrier: str, kind: DisruptionKind, token: str,
    chunks: tuple[PolicyChunk, ...],
) -> tuple[PolicyChunk, bool] | None:
    """Best usable chunk for a category, and whether its id actually names it.

    Ranking is (does the id name this category, is it carrier-specific rather
    than generic, lexical score). The first key matters most — see `CATEGORIES`
    for the two live misses that motivated it.
    """
    ranked = []
    for chunk in chunks:
        if not _usable(chunk, carrier, kind):
            continue
        score = score_chunk(query, chunk, chunks)
        names_category = token in chunk.chunk_id.upper()
        carrier_specific = chunk.local_id.upper().startswith(carrier.upper())
        if score <= 0 and not names_category:
            continue
        ranked.append((names_category, carrier_specific, score, chunk))

    if not ranked:
        return None
    ranked.sort(key=lambda row: (-row[0], -row[1], -row[2]))
    return ranked[0][3], bool(ranked[0][0])


def _extract(chunk: PolicyChunk, limit: int = 260) -> str:
    """A readable lead from the chunk, cut on a sentence where possible.

    The chunk text is quoted rather than paraphrased. Paraphrasing an entitlement
    is how a figure drifts, and the whole point of the citation is that the words
    can be checked against it.
    """
    text = " ".join(chunk.text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: stop + 1] if stop > limit // 2 else cut.rstrip() + "…")


def build_entitlements(
    carrier: str,
    kind: DisruptionKind | str = DisruptionKind.CANCELLED,
    *,
    src: Sources | None = None,
) -> EntitlementsReport:
    """The six-category entitlement picture for one carrier and disruption kind.

    `src` defaults to the configured source set; pass one to build against a
    different corpus or contact directory without touching the environment.
    """
    if isinstance(kind, str):
        try:
            kind = DisruptionKind(kind)
        except ValueError:
            kind = DisruptionKind.UNKNOWN

    src = src or sources()
    chunks = src.policy.chunks()
    contact = src.contacts.contact(carrier)
    report = EntitlementsReport(
        carrier=carrier.upper(),
        disruption_kind=kind.value,
        contact={} if "error" in contact else contact,
        contact_warning=contact.get("_warning", "") if "error" not in contact else "",
    )

    for category, label, query, token in CATEGORIES:
        found = _best(query, carrier, kind, token, chunks)
        # A chunk whose id does not name this category is not an answer for it.
        # Lexical overlap alone put a *refund* chunk under "compensation" for a
        # delayed SQ flight, which is a wrong answer wearing the clothes of a
        # right one. An explicit gap is worse to read and better to trust.
        chunk = found[0] if found and found[1] else None
        if chunk is None:
            report.items.append(Entitlement(
                category=category, label=label, covered=False,
                text=(
                    f"{NOT_COVERED}. Confirm with the carrier — this demo's "
                    f"corpus has no chunk for {carrier.upper()} on this."
                ),
            ))
            continue
        report.items.append(Entitlement(
            category=category, label=label, covered=True,
            text=_extract(chunk), chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
        ))

    return report


def cancellation_advice(report: EntitlementsReport) -> list[str]:
    """Plain next steps implied by what the carrier owes.

    Derived from which categories the corpus actually covers, so the advice
    cannot recommend something the policy does not support — and says so plainly
    where the corpus is silent.
    """
    advice: list[str] = []

    rebooking = report.by_category("rebooking")
    if rebooking and rebooking.covered:
        advice.append(
            "You are entitled to be rebooked rather than charged a new fare — "
            "say so before paying anything."
        )

    accommodation = report.by_category("accommodation")
    if accommodation and accommodation.covered:
        advice.append(
            "If a recovery keeps you overnight, the carrier arranges and pays "
            "for the hotel. Ask at the service desk rather than booking your "
            "own, which is usually only reimbursed with prior authorisation."
        )

    meals = report.by_category("meals")
    if meals and meals.covered:
        advice.append("Ask for meal vouchers if you are waiting at the airport.")

    refund = report.by_category("refund")
    if refund and refund.covered:
        advice.append(
            "You can decline rebooking and take a refund instead — but that "
            "ends the carrier's duty of care, so hotels and meals stop with it."
        )

    compensation = report.by_category("compensation")
    if compensation and compensation.covered:
        advice.append(
            "Cash compensation depends on the carrier's own determination of "
            "cause. Do not assume a figure — this demo's corpus does not state "
            "one for a cancellation."
        )

    if report.contact.get("in_airport"):
        advice.append(f"At the airport: {report.contact['in_airport']}")

    return advice
