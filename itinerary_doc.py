"""Render an `Itinerary` as the booking confirmation a traveller would actually hold.

This exists for the demo: the flow starts from "here is my itinerary", and a JSON
file is not what anyone has in their inbox. The PDF is generated from the same
`data/sample_itinerary_sin.json` the gates and the agent use, so what the demo
shows and what the system reasons about cannot drift apart.

One detail is deliberate and worth noticing during a demo: **this document carries
a PNR and the agent's copy does not.** A booking confirmation is exactly where a
record locator belongs. `recovery.agent.redact` strips it before the itinerary is
mounted into the agent's workspace, so the rule "never echo a PNR" holds because
the PNR was never there to echo — not because a prompt asked nicely.
"""

from __future__ import annotations

import io
from datetime import datetime

from models import Booking, BookingType, Itinerary

#: A plausible record locator for the demo document. Airlines issue six
#: alphanumerics; this one is fabricated and belongs only to the fixture.
DEMO_PNR = "K7QW2M"
_AIRLINE = {"SQ": "Singapore Airlines", "AI": "Air India", "TG": "Thai Airways"}

_KIND_LABEL = {
    BookingType.FLIGHT: "Flight",
    BookingType.TRANSFER: "Airport transfer",
    BookingType.HOTEL: "Hotel",
    BookingType.ACTIVITY: "Activity",
}


def _fmt(dt: datetime) -> str:
    return dt.strftime("%a %d %b %Y, %H:%M %Z").strip()


def _short(dt: datetime) -> str:
    return dt.strftime("%d %b, %H:%M")


def booking_rows(itinerary: Itinerary) -> list[dict[str, str]]:
    """The itinerary as display rows. Shared by the PDF and the in-app render, so
    the two cannot disagree about what the trip contains."""
    rows: list[dict[str, str]] = []
    for booking in sorted(itinerary.bookings, key=lambda b: b.start):
        detail = booking.location_name or booking.location_iata or ""
        if booking.type == BookingType.FLIGHT and booking.legs:
            leg = booking.legs[0]
            detail = (
                f"{_AIRLINE.get(leg.carrier, leg.carrier)} {leg.flight_number} · "
                f"{leg.origin_iata} → {leg.destination_iata}"
                + (f" · {leg.cabin}" if leg.cabin else "")
            )
        rows.append({
            "kind": _KIND_LABEL.get(booking.type, booking.type.value),
            "title": booking.title,
            "detail": detail,
            "starts": _fmt(booking.start),
            "ends": _fmt(booking.end),
            "reference": booking.legs[0].booking_reference or DEMO_PNR
            if booking.type == BookingType.FLIGHT and booking.legs
            else DEMO_PNR,
        })
    return rows


def build_pdf(itinerary: Itinerary, *, pnr: str = DEMO_PNR) -> bytes:
    """Render the itinerary to a booking-confirmation PDF."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    ink = colors.HexColor("#14213D")
    accent = colors.HexColor("#1F6FEB")
    muted = colors.HexColor("#5B6472")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=ink,
                        fontSize=20, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=muted,
                         fontSize=9.5, spaceAfter=14)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=accent,
                        fontSize=11.5, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9,
                          textColor=ink, leading=13)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=7.8,
                           textColor=muted, leading=11)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=14 * mm,
        title=f"Booking confirmation {pnr}",
        author="TripRecovery demo",
    )

    flow: list = [
        Paragraph("Booking confirmation", h1),
        Paragraph(
            f"Reference <b>{pnr}</b> &nbsp;·&nbsp; {itinerary.traveller_first_name} "
            f"and party of {itinerary.party_size} &nbsp;·&nbsp; trip "
            f"<font face='Courier'>{itinerary.trip_id}</font>",
            sub,
        ),
    ]

    rows = [["", "Booking", "Starts", "Ends"]]
    for row in booking_rows(itinerary):
        rows.append([
            Paragraph(f"<b>{row['kind']}</b>", small),
            Paragraph(f"<b>{row['title']}</b><br/><font size=8 color='#5B6472'>"
                      f"{row['detail']}</font>", body),
            Paragraph(row["starts"], small),
            Paragraph(row["ends"], small),
        ])

    table = Table(rows, colWidths=[26 * mm, 78 * mm, 33 * mm, 33 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF3FB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), accent),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#DCE3EF")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#DCE3EF")),
    ]))
    flow += [table, Spacer(1, 6 * mm)]

    flight = next(
        (b for b in itinerary.bookings if b.type == BookingType.FLIGHT and b.legs),
        None,
    )
    if flight is not None:
        leg = flight.legs[0]
        flow += [
            Paragraph("Your outbound flight", h2),
            Paragraph(
                f"<b>{_AIRLINE.get(leg.carrier, leg.carrier)} {leg.flight_number}</b> "
                f"departs {leg.origin_iata} at {_fmt(leg.scheduled_departure)} and "
                f"arrives {leg.destination_iata} at {_fmt(leg.scheduled_arrival)}. "
                f"Check in at least three hours before departure.",
                body,
            ),
        ]

    flow += [
        Paragraph("Important", h2),
        Paragraph(
            "Your airport transfer, hotel check-in and activity are timed around "
            "the arrival of your flight. If that flight is delayed or cancelled, "
            "each of them may be affected in turn.",
            body,
        ),
        Spacer(1, 8 * mm),
        Paragraph(
            "Generated by TripRecovery. Not a real booking.", small,
        ),
    ]

    doc.build(flow)
    return buffer.getvalue()
