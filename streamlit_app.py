"""TripRecovery UI.

A guided flow: welcome → your itinerary (sample or uploaded) → the two options →
the action receipt.

**The UI holds no copy of the recovery's truth.** Which recovery state to render
is derived from the SQLite store on every rerun — the approval row, the handoff
rows, the working itinerary. `st.session_state` holds which *screen* the visitor
is on and which itinerary they supplied, both of which are inputs rather than
derived state.

That split matters because the recovery outlives the page: a traveller may
confirm the transfer today and the payment tomorrow from a different tab. Any
recovery state the UI kept for itself would be a second, staler answer to "where
is this trip up to", and the whole point of the ledger is that there is one.

**Nothing here contacts anyone.** There is no hotel integration, no transfer
provider, no airline, no calendar. Every outward-facing action is recorded as
`prepared`, every button says what it actually does, and no control implies a
navigation that does not exist.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st

from config import IS_MOCK_DATA, LLM_MODEL, SQLITE_DB, summary
from core.approval import Refusal, approve, authorize
from core.entitlements import build_entitlements, cancellation_advice
from core.executor import ConfirmRefusal, confirm_handoff, execute, outbox, revalidate
from core.planner import NoRecovery
from core.plan_compiler import itinerary_version
from core.store import (
    clear_trip,
    current_approval,
    load_handoffs,
    load_ledger,
    load_plan,
    save_approval,
    save_plan,
)
from demo import ITINERARY_PATH, Scenario
from recovery.agent import flight_designator
from itinerary_doc import DEMO_PNR, booking_rows, build_pdf
from models import ActionClass, ActionKind, BookingType, Itinerary, RecoveryPlan

st.set_page_config(
    page_title="TripRecovery",
    page_icon=":material/flight_takeoff:",
    layout="centered",
)

logger = logging.getLogger(__name__)

RISK_COLOUR = {"low": "green", "medium": "orange", "high": "red"}
SEVERITY = {
    "broken": ("Broken", "red"),
    "at_risk": ("At risk", "orange"),
    "degraded": ("Tight", "gray"),
}
ACTION_ICON = {
    ActionKind.UPDATE_ITINERARY: ":material/edit_calendar:",
    ActionKind.UPDATE_CALENDAR: ":material/event:",
    ActionKind.SEND_HOTEL_MESSAGE: ":material/mail:",
    ActionKind.PAY_FOR_FLIGHT: ":material/credit_card:",
    ActionKind.REBOOK_TRANSFER: ":material/local_taxi:",
    ActionKind.RESCHEDULE_ACTIVITY: ":material/confirmation_number:",
    ActionKind.CANCEL_ACTIVITY: ":material/cancel:",
}
#: What each handoff button honestly does. None of these navigate anywhere —
#: there is no booking integration — so none of them say "open" any more.
#: The quiet "this is done now" control under each pending action. Every label
#: describes marking a record, not performing the act — the previous set ("I have
#: paid", "I have rebooked it") asked the traveller to attest to something they
#: had not been prompted to do.
HANDOFF_CTA = {
    ActionKind.PAY_FOR_FLIGHT: "Mark as booked",
    ActionKind.REBOOK_TRANSFER: "Mark as rebooked",
    ActionKind.RESCHEDULE_ACTIVITY: "Mark as moved",
    ActionKind.CANCEL_ACTIVITY: "Mark as cancelled",
}
TRAVELLER_ASK = (
    "My flight is cancelled and this is my itinerary — tell me what the best "
    "options are to make sure the rest of the trip goes smoothly."
)


# ===============================================================
# The active trip. Uploaded itinerary wins; otherwise the sample.
# ===============================================================
@st.cache_data(show_spinner=False)
def sample_itinerary_json() -> str:
    return ITINERARY_PATH.read_text("utf-8")


def active_itinerary_json() -> str:
    return st.session_state.get("uploaded_itinerary") or sample_itinerary_json()


@st.cache_resource(show_spinner=False)
def scenario_for(itinerary_json: str) -> Scenario:
    """Everything the deterministic core derives from a trip. Pure, so cacheable."""
    return Scenario.from_itinerary(Itinerary.model_validate_json(itinerary_json))


@st.cache_data(show_spinner=False)
def planner_plan_json(itinerary_json: str) -> tuple[str | None, list[str]]:
    """The deterministic planner's compiled plan for this trip, or why there isn't one."""
    sc = scenario_for(itinerary_json)
    if sc.disruption is None:
        return None, ["No flight in this itinerary is disrupted."]
    if not sc.disruption.verified:
        return None, [sc.disruption.note or "The disruption could not be verified."]
    try:
        result = sc.builder_plan()
    except NoRecovery as exc:
        return None, [str(exc)]
    if result.plan is None:
        return None, result.violations
    return result.plan.model_dump_json(), []


def build_plan(itinerary_json: str) -> tuple[RecoveryPlan | None, list[str]]:
    """Recover this trip with the deep agent.

    The rule-based planner is kept as a safety net rather than a mode: if the
    agent cannot run — no API key, exhausted credits, a provider limit — the
    demo continues rather than dead-ending on a screen nobody wants to see live.
    The fallback is logged, not announced, because a two-minute demo does not
    need a lecture about which code path produced the plan.
    """
    sc = scenario_for(itinerary_json)
    if sc.disruption is None:
        return None, ["No flight in this itinerary is disrupted."]
    if not sc.disruption.verified:
        return None, [sc.disruption.note or "The disruption could not be verified."]

    from recovery.llm import available

    if available(LLM_MODEL):
        try:
            from recovery.agent import run_recovery

            result = run_recovery(sc.itinerary)
            if result.ok and result.plan is not None:
                logger.info("plan built by the deep agent: %s", result.cost)
                return result.plan, []
            logger.warning(
                "agent run did not produce a plan (%s); using the planner",
                result.stopped_reason or result.violations,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent run failed (%s); using the planner", exc)

    payload, problems = planner_plan_json(itinerary_json)
    if payload is None:
        return None, problems
    return RecoveryPlan.model_validate_json(payload), []


@st.cache_data(show_spinner=False)
def entitlements_for(carrier: str, kind: str):
    """What the carrier owes. Deterministic and local, so free to compute."""
    return build_entitlements(carrier, kind)


def operating_carrier(sc: Scenario) -> str:
    """The carrier of the disrupted flight, from the itinerary rather than parsed
    out of a designator string."""
    if sc.disruption is None:
        return ""
    booking = next(
        (b for b in sc.itinerary.bookings
         if b.booking_id == sc.disruption.booking_id and b.legs), None
    )
    return booking.legs[0].carrier if booking else ""


@st.cache_data(show_spinner=False)
def pdf_for(itinerary_json: str) -> bytes:
    return build_pdf(Itinerary.model_validate_json(itinerary_json))


def fmt(dt: datetime) -> str:
    return dt.strftime("%a %d %b, %H:%M")


def duration(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} min"
    hours, mins = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h" if not mins else f"{hours}h {mins}m"
    days, rem = divmod(hours, 24)
    return f"{days}d" if not rem else f"{days}d {rem}h"


def flight_search_url(sc: Scenario) -> str | None:
    """A real, generic flight search for the disrupted route.

    Deliberately a *search*, and labelled as one. There is no held fare and no
    booking integration, so a link claiming to be "the booking page" would be
    the same false affordance as the button that used to say "Open booking page"
    and do nothing at all.
    """
    if sc.disruption is None:
        return None
    booking = next(
        (b for b in sc.itinerary.bookings
         if b.booking_id == sc.disruption.booking_id and b.legs), None
    )
    if booking is None:
        return None
    leg = booking.legs[0]
    query = (
        f"Flights from {leg.origin_iata} to {leg.destination_iata} "
        f"on {leg.scheduled_departure.date().isoformat()}"
    )
    return f"https://www.google.com/travel/flights?q={quote_plus(query)}"


itinerary_json = active_itinerary_json()
sc = scenario_for(itinerary_json)
trip_id = sc.itinerary.trip_id
is_uploaded = bool(st.session_state.get("uploaded_itinerary"))
st.session_state.setdefault("stage", "welcome")


# ===============================================================
# Sidebar
# ===============================================================
with st.sidebar:
    st.subheader(":material/flight_takeoff: TripRecovery")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"`{trip_id}`")
        if IS_MOCK_DATA:
            # Stated once, quietly. Flight status, availability, airline
            # contacts and policy all come from fixtures in this mode.
            st.badge("Mock data", icon=":material/science:", color="gray")

    if st.button("Start over", icon=":material/restart_alt:", width="stretch"):
        clear_trip(trip_id)
        st.cache_data.clear()
        st.cache_resource.clear()
        for key in ("plan_version_override", "plan_version",
                    "uploaded_itinerary", "pdf_fallback"):
            st.session_state.pop(key, None)
        st.session_state["stage"] = "welcome"
        st.rerun()

    with st.expander("What has actually happened", icon=":material/policy:"):
        # The one place this is spelled out. It used to be repeated on the
        # welcome screen, the itinerary screen, the options screen and the
        # receipt; saying it five times made it wallpaper rather than
        # information.
        st.markdown(
            "**Not contacted:** hotel, transfer provider, activity operator, "
            "airline. No integration behind any of them.\n\n"
            "**Real either way:** the dependency graph, the two compiled "
            "options, the approval hash, and the changes to *your* copy of the "
            "itinerary."
        )
        if sent := outbox():
            st.caption(f"{len(sent)} message drafted, none delivered:")
            for message in sent:
                st.text(message["body"])

    with st.expander("Config", icon=":material/settings:"):
        st.json(summary(), expanded=False)
        st.caption(f"Store: `{SQLITE_DB}`")


# ===============================================================
# The plan on screen, if one has been built yet
# ===============================================================
# Built by the deep agent when "Find my options" is pressed; see `build_plan`.
# A recompiled plan (the "change something first" panel) takes precedence.
plan: RecoveryPlan | None = None
plan_problems: list[str] = []
for key in ("plan_version_override", "plan_version"):
    if version := st.session_state.get(key):
        plan = load_plan(version)
        if plan is not None:
            break

if plan is not None:
    save_plan(plan)

# --- Recovery state, derived entirely from the store -------------------
# The approval is looked up regardless of whether this session has a plan in
# hand, and supplies the plan when it does not. Without that, a new visitor
# could not be offered the recovery already under way — the plan was only
# loaded from session state, which a fresh tab does not have.
approval = current_approval(trip_id)
if plan is None and approval is not None:
    plan = load_plan(approval.plan_version)
handoffs = load_handoffs(trip_id, approval.plan_version) if approval else []
ledger = load_ledger(trip_id, approval.plan_version) if approval else []
validation = revalidate(trip_id, approval.plan_version) if approval else None
approval_is_stale = bool(
    approval and plan is not None and approval.plan_version != plan.plan_version
)

# Only advance to the receipt from the options screen — approving is what moves
# you on. This used to fire unconditionally, which meant a stored approval from
# any earlier run hijacked a *fresh* session: open the app and land on a
# completed recovery, having never seen the welcome screen. Recovery state
# rightly lives in the store, but which screen a new visitor starts on does not.
if approval and not approval_is_stale and st.session_state["stage"] == "plan":
    st.session_state["stage"] = "receipt"
stage = st.session_state["stage"]


# ===============================================================
# SCREEN 1 — welcome
# ===============================================================
if stage == "welcome":
    st.title("Travel plans break. This puts them back together.")
    st.markdown(
        "#### When a flight is cancelled, the flight is rarely the only problem.\n"
        "The transfer meeting you, the hotel expecting you, the tickets booked "
        "for that evening — each was timed around an arrival that is no longer "
        "happening."
    )
    st.space("small")

    with st.container(horizontal=True):
        with st.container(border=True):
            st.markdown(":material/account_tree: **See what actually breaks**")
            st.caption(
                "Your bookings depend on each other. TripRecovery walks that "
                "chain and tells you which are broken, which are merely tight, "
                "and by how long."
            )
            st.badge("Checked, not guessed", color="blue")
        with st.container(border=True):
            st.markdown(":material/alt_route: **Get two real options**")
            st.caption(
                "Not a list of flights. Two complete plans, each checked to "
                "leave a trip that actually holds together, with the trade-off "
                "between them spelled out."
            )
            st.badge("Cost · delay · what changes", color="violet")
        with st.container(border=True):
            st.markdown(":material/lock: **Nothing happens until you say so**")
            st.caption(
                "You approve one specific plan. Anything involving money or a "
                "third party stays yours to do, and you can see which is which "
                "before you decide."
            )
            st.badge("You stay in control", color="green")

    st.space("medium")
    st.markdown("##### Where would you like to start?")

    if approval is not None and not approval_is_stale:
        # A recovery is already under way for this trip. Say so and offer it,
        # rather than either hiding it or forcing the visitor into it.
        with st.container(border=True, horizontal=True,
                          vertical_alignment="center"):
            st.markdown(
                f"**A recovery is already in progress**  \n"
                f":small[Option {approval.option_id}, approved "
                f"{fmt(approval.approved_at)}]"
            )
            with st.container(horizontal_alignment="right"):
                if st.button("Pick it up", icon=":material/resume:",
                             type="primary"):
                    st.session_state["stage"] = "receipt"
                    st.rerun()

    with st.container(horizontal=True):
        if st.button("My flight was cancelled", type="primary",
                     icon=":material/flight_class:", width="stretch"):
            st.session_state["stage"] = "itinerary"
            st.rerun()
        if st.button("Upload my booking confirmation",
                     icon=":material/upload_file:",
                     width="stretch"):
            st.session_state["stage"] = "itinerary"
            st.session_state["show_upload"] = True
            st.rerun()

    st.stop()


# ===============================================================
# SCREEN 2 — your itinerary, sample or uploaded
# ===============================================================
def render_uploader() -> None:
    """The upload control.

    Accepts a booking-confirmation **PDF**, because that is what a traveller
    actually has in their inbox. Parsing one is not implemented yet — the
    uploader says so plainly rather than accepting a file and quietly doing
    nothing with it, which is the same class of false affordance as a button
    that navigates nowhere. See the TODO in the README.

    A JSON path is kept alongside it for now, unlabelled as the primary route,
    because removing it outright would leave "upload your own trip" with no
    working implementation at all.
    """
    upload = st.file_uploader(
        "Your booking confirmation", type=["pdf", "json"],
        key="itinerary_upload",
        help="A booking-confirmation PDF from your airline or agent.",
    )
    with st.container(horizontal=True):
        st.download_button(
            "Example confirmation", data=pdf_for(sample_itinerary_json()),
            file_name="example-booking-confirmation.pdf",
            mime="application/pdf", icon=":material/download:",
        )
        if is_uploaded and st.button("Use the prepared trip",
                                     icon=":material/undo:"):
            st.session_state.pop("uploaded_itinerary", None)
            st.session_state.pop("show_upload", None)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    if upload is None:
        return

    if (upload.name or "").lower().endswith(".pdf"):
        # PDF parsing is not built yet, so fall through to the sample trip and
        # carry on rather than stopping the flow with a dead end.
        #
        # The `!=` guard is load-bearing. A file_uploader keeps its file in
        # widget state across reruns, so without it this branch fired on every
        # single run — set the flag, rerun, see the same file again, rerun —
        # an infinite loop that looked like a frozen page and made the Back
        # button appear broken, because no click ever got processed.
        if st.session_state.get("pdf_fallback") != upload.name:
            st.session_state["pdf_fallback"] = upload.name
            st.session_state.pop("uploaded_itinerary", None)
            st.session_state.pop("show_upload", None)
            st.rerun()
        return

    raw = upload.getvalue().decode("utf-8", errors="replace")
    try:
        candidate = Itinerary.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001
        st.error("That file could not be read as an itinerary.",
                 icon=":material/error:")
        st.caption(f"{type(exc).__name__}: {str(exc)[:300]}")
        return
    # Same shape of guard, for the same reason: the file stays in widget state,
    # so this must only fire when the active trip is not already this one.
    if candidate.model_dump_json() != sc.itinerary.model_dump_json():
        st.session_state["uploaded_itinerary"] = candidate.model_dump_json()
        st.session_state.pop("show_upload", None)
        st.session_state.pop("pdf_fallback", None)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.success(
        f"Loaded {len(candidate.bookings)} bookings for `{candidate.trip_id}`.",
        icon=":material/check:",
    )


if stage == "itinerary":
    st.title("Let's look at your trip")

    # Arriving via "Upload my own itinerary" shows ONLY the uploader. Rendering
    # the sample trip underneath it read as though that trip was already yours,
    # which is worse than an empty screen: it invites approving someone else's
    # itinerary.
    awaiting_upload = bool(st.session_state.get("show_upload")) and not is_uploaded
    if awaiting_upload:
        st.markdown("#### Upload your booking confirmation")
        st.caption("Every screen after this follows from your bookings.")
        with st.container(border=True):
            render_uploader()
        if st.button("Back", icon=":material/arrow_back:"):
            for key in ("show_upload", "pdf_fallback", "itinerary_upload"):
                st.session_state.pop(key, None)
            st.session_state["stage"] = "welcome"
            st.rerun()
        st.stop()

    with st.expander(
        "Use my own itinerary",
        icon=":material/upload_file:",
        expanded=is_uploaded,
    ):
        st.caption("Every screen after this follows from your bookings.")
        render_uploader()

    with st.chat_message("user"):
        st.markdown(TRAVELLER_ASK)
        st.caption(
            f":material/attach_file: "
            + (f"{trip_id}" if is_uploaded
               else f"booking-confirmation-{DEMO_PNR}.pdf")
            + f" · {len(sc.itinerary.bookings)} bookings · "
            f"{sc.itinerary.party_size} travellers"
        )

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            route = ""
            flight = next(
                (b for b in sc.itinerary.bookings
                 if b.type is BookingType.FLIGHT and b.legs), None
            )
            if flight is not None:
                route = (f" · {flight.legs[0].origin_iata} → "
                         f"{flight.legs[0].destination_iata}")
            st.markdown(
                ("**Your itinerary**" if is_uploaded
                 else f"**Booking confirmation {DEMO_PNR}**")
                + f"  \n:small[{sc.itinerary.traveller_first_name} and party of "
                f"{sc.itinerary.party_size}{route}]"
            )
            with st.container(horizontal_alignment="right"):
                st.download_button(
                    "Download PDF", data=pdf_for(itinerary_json),
                    file_name=f"itinerary-{trip_id}.pdf",
                    mime="application/pdf", icon=":material/download:",
                )

        if fallback_name := st.session_state.get("pdf_fallback"):
            st.caption(
                f":orange[`{fallback_name}` could not be read, so the bookings "
                f"below are not from your file.]"
            )

        rows = booking_rows(sc.itinerary)
        if st.session_state.get("pdf_fallback"):
            # The file could not be read, so these are not the visitor's dates.
            # Showing them anyway invited exactly the confusion they cause —
            # concrete times for a trip that is not theirs.
            st.table({
                "": [r["kind"] for r in rows],
                "Booking": [r["title"] for r in rows],
            })
        else:
            st.table({
                "": [r["kind"] for r in rows],
                "Booking": [r["title"] for r in rows],
                "Starts": [r["starts"] for r in rows],
                "Ends": [r["ends"] for r in rows],
            })
            st.caption(
                "Everything after the flight is timed around its arrival. That "
                "is the chain TripRecovery checks."
            )

    with st.spinner("Checking your bookings…"):
        payload, problems = planner_plan_json(itinerary_json)
    if payload is None:
        st.warning(
            "This trip cannot be recovered here:", icon=":material/warning:"
        )
        for problem in problems:
            st.write(f"- {problem}")
        st.caption(
            "Replacement inventory is only available for BOM → SIN on "
            "2026-09-15."
        )
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f"I can confirm the flight with the carrier, work out which of "
                f"these {len(sc.itinerary.bookings)} bookings the cancellation "
                f"breaks, and put together two recovery plans for you to choose "
                f"between."
            )

    with st.container(horizontal=True):
        if payload is not None and st.button(
            "Find my options", type="primary", icon=":material/travel_explore:"
        ):
            with st.spinner("Checking the flight, your bookings and the policy…"):
                built, problems = build_plan(itinerary_json)
            if built is None:
                st.error("This trip could not be recovered:")
                for problem in problems:
                    st.write(f"- {problem}")
                st.stop()
            save_plan(built)
            st.session_state["plan_version"] = built.plan_version
            st.session_state.pop("plan_version_override", None)
            st.session_state["stage"] = "plan"
            st.rerun()
        if st.button("Back", icon=":material/arrow_back:"):
            for key in ("pdf_fallback", "itinerary_upload"):
                st.session_state.pop(key, None)
            st.session_state["stage"] = "welcome"
            st.rerun()
    st.stop()


if plan is None:
    st.title("No plan for this trip")
    for problem in plan_problems:
        st.write(f"- {problem}")
    if st.button("Back to my itinerary", icon=":material/arrow_back:"):
        st.session_state["stage"] = "itinerary"
        st.rerun()
    st.stop()


# ===============================================================
# SCREEN 3 — the disruption, the damage, the two options
# ===============================================================
disruption = plan.verified_disruption
ent = entitlements_for(operating_carrier(sc) or disruption.flight_iata[:2],
                       disruption.kind.value)

if stage == "plan":
    st.title(f"Flight {disruption.flight_iata} has been cancelled")

    with st.container(horizontal=True):
        st.badge(
            "Confirmed with the carrier" if disruption.verified else "Not verified",
            icon=":material/verified:" if disruption.verified else ":material/error:",
            color="green" if disruption.verified else "red",
        )
        st.badge(f"{sc.itinerary.party_size} travellers",
                 icon=":material/group:", color="gray")
    st.caption(f"Checked {fmt(disruption.fetched_at)} · plan `{plan.plan_version}`")

    st.header("What this breaks")
    blocking = [i for i in plan.impacts if i.severity in ("broken", "at_risk")]
    st.caption(f"{len(blocking)} of {len(plan.impacts)} bookings need attention")

    for impact in plan.impacts:
        label, colour = SEVERITY.get(impact.severity, (impact.severity, "gray"))
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(
                f"**{impact.label or impact.node_id}**  \n:small[{impact.reason}]"
            )
            with st.container(horizontal_alignment="right"):
                st.badge(label, color=colour)

    if approval_is_stale and approval is not None:
        st.warning(
            f"This plan changed after it was approved. The approval is bound to "
            f"`{approval.plan_version}` and this plan is `{plan.plan_version}`, "
            f"so it has to be approved again — nothing will execute until it is.",
            icon=":material/lock_reset:",
        )

    # ---- what the airline owes, before the options ------------------
    # Entitlements come first deliberately: they change which option is
    # acceptable and what it really costs. If the carrier owes a hotel, an
    # overnight option stops being the expensive one.
    st.header(f"What {ent.carrier} owes you")
    st.caption(
        f"{ent.covered_count} of 6 categories are covered by the policy corpus "
        f"for a {ent.disruption_kind} flight."
    )

    if ent.contact.get("disruption_desk"):
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(
                    f":material/call: **{ent.contact.get('name', ent.carrier)} "
                    f"disruption desk**  \n"
                    f"### {ent.contact['disruption_desk']}"
                )
                with st.container(horizontal_alignment="right"):
                    regional = ent.contact.get("regional") or {}
                    if regional:
                        st.caption("**By region**  \n" + "  \n".join(
                            f"{code}: {number}" for code, number in regional.items()
                        ))
            if ent.contact.get("in_airport"):
                st.caption(f":material/storefront: {ent.contact['in_airport']}")
    else:
        st.warning(
            f"No verified contact number is on file for {ent.carrier}. Use the "
            f"number printed on your ticket — TripRecovery will not guess one.",
            icon=":material/phone_disabled:",
        )

    advice = cancellation_advice(ent)
    if advice:
        with st.expander("What to do about it", icon=":material/tips_and_updates:",
                         expanded=True):
            for line in advice:
                st.markdown(f"- {line}")

    # Six categories of quoted policy is a lot of reading on the screen where
    # someone is choosing a flight, and as bordered cards it pushed the options
    # off the page. Collapsed, and as a list rather than cards, so the chunk id
    # is not truncated — a citation you cannot read is no citation at all.
    with st.expander(
        f"The policy behind this ({ent.covered_count} of 6 categories covered)",
        icon=":material/gavel:",
    ):
        for item in ent.items:
            if item.covered:
                st.markdown(
                    f"**{item.label}** — {item.text}  \n"
                    f":small[`{item.chunk_id}`]"
                )
            else:
                st.markdown(
                    f"**{item.label}** — :orange[not covered by the policy "
                    f"corpus.] Confirm with the carrier."
                )

    st.header("Your two options")
    st.caption("Ranked by whole-trip recovery, then arrival, then cost.")

    for index, option in enumerate(plan.options):
        recommended = index == 0
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.subheader(f"Option {option.option_id}")
                if recommended:
                    st.badge("Recommended", color="blue")
                with st.container(horizontal_alignment="right"):
                    st.badge(
                        f"{option.risk.title()} risk",
                        icon=":material/shield:" if option.risk == "low"
                        else ":material/warning:",
                        color=RISK_COLOUR[option.risk],
                    )

            st.write(option.headline)
            flight = option.flight
            # `flight_designator` rather than concatenation: the provider stores
            # SQ425 in `flight_number`, so carrier + number rendered "SQSQ425".
            st.markdown(
                f":material/flight_takeoff: "
                f"**{flight_designator(flight.carrier, flight.flight_number)}** "
                f"{flight.origin_iata} → {flight.destination_iata} · departs "
                f"{fmt(flight.departure)} · arrives {fmt(flight.arrival)}"
            )

            with st.container(horizontal=True):
                st.metric(
                    "Extra cost",
                    f"{option.additional_cost_currency} "
                    f"{option.additional_cost_amount:,.0f}", border=True,
                )
                st.metric("Arrival delay", duration(option.arrival_delay_minutes),
                          border=True)
                st.metric("Bookings handled", f"{len(option.actions)}", border=True)

            with st.container(horizontal=True):
                for action in option.actions:
                    is_agent = action.action_class is ActionClass.AGENT_SAFE
                    st.badge(
                        action.description.rstrip("."),
                        icon=ACTION_ICON.get(action.kind, ":material/task_alt:"),
                        color="green" if is_agent else "orange",
                    )
            st.caption(
                ":green[Green] — TripRecovery can prepare this for you. "
                ":orange[Amber] — needs you, because money moves or a provider "
                "must confirm."
            )

            overnight = option.flight.arrival.date() > disruption.fetched_at.date()
            accommodation = ent.by_category("accommodation")
            if overnight and accommodation and accommodation.covered:
                st.caption(
                    f":green[:material/hotel: This option keeps you overnight — "
                    f"and {ent.carrier} owes you the hotel, so the real cost to "
                    f"you is lower than the fare suggests.] "
                    f"`{accommodation.chunk_id}`"
                )
            elif overnight:
                st.caption(
                    ":orange[:material/hotel: This option keeps you overnight and "
                    "the corpus does not confirm accommodation cover — assume the "
                    "hotel is yours to pay until the carrier says otherwise.]"
                )

            with st.expander("Why this option", icon=":material/help:"):
                st.write(option.rationale)
                st.caption(f"Risk: {option.risk_reason}")
                st.caption("Grounded in " + ", ".join(f"`{c}`" for c in option.citations))
                st.caption(
                    f"Fare and availability from `{flight.source}`, fetched "
                    f"{fmt(flight.fetched_at)}."
                )

            with st.expander(f"Exactly what you are approving "
                             f"({len(option.actions)} actions)",
                             icon=":material/checklist:"):
                st.caption(
                    "Approving covers these and nothing else. The message below "
                    "is approved as written — those exact words are the only "
                    "ones that could be sent."
                )
                for action in option.actions:
                    is_agent = action.action_class is ActionClass.AGENT_SAFE
                    when = (f" → {fmt(action.new_start)}"
                            if action.new_start else "")
                    st.markdown(
                        f"{ACTION_ICON.get(action.kind, ':material/task_alt:')} "
                        f"**{action.description}**{when} "
                        + (":green-badge[TripRecovery]" if is_agent
                           else ":orange-badge[You]")
                    )
                    if action.message_body:
                        st.code(action.message_body, language=None, wrap_lines=True)

            if st.button(
                f"Approve Option {option.option_id}"
                if recommended else f"Choose Option {option.option_id}",
                key=f"approve_{option.option_id}",
                type="primary" if recommended else "secondary",
                width="stretch", icon=":material/check:",
            ):
                record = approve(plan, option.option_id, approved_by="traveller")
                if isinstance(record, Refusal):
                    st.error(f"{record.reason.value}: {record.detail}")
                    st.stop()
                save_approval(record)
                auth = authorize(plan, record, sc.itinerary)
                if isinstance(auth, Refusal):
                    st.error(f"{auth.reason.value}: {auth.detail}")
                    st.stop()
                execute(auth)
                st.session_state["stage"] = "receipt"
                st.rerun()

    with st.expander("I would rather handle it myself",
                     icon=":material/self_improvement:"):
        st.caption("What needs doing, with the times that make the trip work.")
        for option in plan.options:
            st.markdown(f"**Option {option.option_id}** — {option.headline}")
            for action in option.actions:
                when = f" → {fmt(action.new_start)}" if action.new_start else ""
                st.markdown(f"- {action.description}{when}")

    if st.button("Back to my itinerary", icon=":material/arrow_back:"):
        st.session_state["stage"] = "itinerary"
        st.rerun()
    st.stop()


# ===============================================================
# SCREEN 4 — the receipt
# ===============================================================
assert approval is not None and validation is not None
option = plan.option(approval.option_id)
assert option is not None

pending = [h for h in handoffs if h.status == "pending"]
entries = {e.action_id: e for e in ledger}
settled = [e for e in ledger if e.settled]
prepared = [e for e in ledger if e.status == "prepared"]
total = len(option.actions)

if validation.valid:
    st.title("Your itinerary holds together again")
    if prepared:
        # "Valid" means the *schedule* is consistent. It does not mean everyone
        # who needs telling has been told, and "nothing is outstanding" here
        # would be the same false commit in a friendlier sentence.
        st.success(
            f"Option {approval.option_id} is in place and every dependency in "
            f"the trip works again — the flight, the transfer, the hotel and "
            f"the activity all line up. "
            f"{len(prepared)} message{'s' if len(prepared) != 1 else ''} "
            f"{'are' if len(prepared) != 1 else 'is'} still waiting to go out.",
            icon=":material/check_circle:",
        )
    else:
        st.success(
            f"Option {approval.option_id} is in place. Every dependency in the "
            f"trip works again, and nothing is left outstanding.",
            icon=":material/check_circle:",
        )
else:
    st.title("Recovery in progress")
    st.caption(
        f"Option {approval.option_id} is approved. "
        + (f"{len(pending)} of {total} still need you."
           if pending else "Working through the plan.")
    )

if prepared:
    st.info(
        f"**{len(prepared)} action{'s' if len(prepared) != 1 else ''} drafted "
        f"and ready — nothing sent yet.** TripRecovery can send "
        f"{'them' if len(prepared) != 1 else 'it'} for you.",
        icon=":material/drafts:",
    )

st.progress(
    len(settled) / total if total else 0.0,
    text=f"{len(settled)} of {total} actually settled"
         + (f" · {len(prepared)} ready to send" if prepared else ""),
)

search_url = flight_search_url(sc)

for action in option.actions:
    entry = entries.get(action.action_id)
    handoff = next((h for h in handoffs if h.action_id == action.action_id), None)

    with st.container(border=True):
        if entry is None:
            with st.container(horizontal=True, vertical_alignment="center"):
                if action.kind is ActionKind.PAY_FOR_FLIGHT:
                    # The one action that genuinely needs the traveller to go
                    # somewhere. So it gets a link that goes there, rather than
                    # a button asking them to assert they already did it.
                    st.markdown(
                        f"{ACTION_ICON[action.kind]} **{action.description}**  \n"
                        f":small[Opens a flight search for this route.]"
                    )
                    with st.container(horizontal_alignment="right"):
                        if search_url:
                            st.link_button(
                                "Rebook", search_url,
                                icon=":material/open_in_new:", type="primary",
                            )
                else:
                    # Prepared, exactly like the agent-safe actions: TripRecovery
                    # holds the new time and has told nobody. Reading "I have
                    # rebooked it" implied the traveller had already done
                    # something they had not been asked to do yet.
                    st.markdown(
                        f":material/drafts: **{action.description}**  \n"
                        f":small[The new time is ready — the provider has not "
                        f"been contacted.]"
                    )
                    with st.container(horizontal_alignment="right"):
                        st.badge("Ready — not sent", color="orange")

            if handoff is not None and st.button(
                HANDOFF_CTA.get(action.kind, "Mark as done"),
                key=f"confirm_{action.action_id}",
                icon=":material/check:", type="tertiary",
            ):
                outcome = confirm_handoff(
                    trip_id, approval.plan_version, action.action_id
                )
                if isinstance(outcome, ConfirmRefusal):
                    st.error(outcome.detail)
                else:
                    st.rerun()
            continue

        if entry.status == "prepared":
            icon, badge, colour = ":material/drafts:", "Ready — not sent", "orange"
        elif entry.status == "done" and entry.performed_by == "human":
            icon, badge, colour = ":material/check_circle:", "You confirmed", "blue"
        elif entry.status == "done":
            icon, badge, colour = ":material/check_circle:", "In your itinerary", "green"
        else:
            icon, badge, colour = ":material/error:", entry.status.title(), "red"

        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"{icon} **{action.description}**  \n:small[{entry.detail}]")
            with st.container(horizontal_alignment="right"):
                st.badge(badge, color=colour)

if pending:
    st.warning(
        f"Your turn: {len(pending)} action"
        f"{'s' if len(pending) != 1 else ''} needs you. TripRecovery keeps this "
        f"state and rechecks the whole itinerary after each confirmation.",
        icon=":material/hourglass_top:",
    )

if validation.remaining:
    st.error("Something in the itinerary still does not work:", icon=":material/error:")
    for node in validation.remaining:
        st.write(f"- **{node.label or node.node_id}** — {node.reason}")

if validation.warnings:
    with st.expander(f"Tight connections ({len(validation.warnings)})",
                     icon=":material/schedule:"):
        st.caption(
            "Feasible, but with little slack. These do not block anything — the "
            "trip was originally booked with one of them."
        )
        for node in validation.warnings:
            st.write(f"- **{node.label or node.node_id}** — {node.reason}")

with st.expander("Your itinerary as it now stands", icon=":material/list:"):
    from core.executor import working_itinerary

    working = working_itinerary(trip_id)
    if working is None:
        st.caption("Nothing applied yet.")
    else:
        ordered = sorted(working.bookings, key=lambda b: b.start)
        st.caption(
            f"Version `{itinerary_version(working)}` · was "
            f"`{plan.itinerary_version}` when the plan was built"
        )
        st.table({
            "Booking": [b.title for b in ordered],
            "Starts": [fmt(b.start) for b in ordered],
            "Ends": [fmt(b.end) for b in ordered],
        })
