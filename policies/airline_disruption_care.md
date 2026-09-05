# Airline Disruption Care & Entitlements

> **DEMO FIXTURE DATA — NOT LEGAL ADVICE.**
> These chunks exist so the TripSure advisor agent has a grounded, citable corpus for
> duty-of-care questions. Figures and thresholds here are illustrative and are **not** a
> reliable statement of any carrier's obligations or of EU/UK/US passenger-rights law.
> Real entitlements vary by departure country, fare class, ticket conditions, and the
> carrier's own determination of the cause of disruption. Anything a traveller intends to
> act on must be confirmed with the operating carrier.

Retrievable chunks, same convention as `airline_rebooking.md`: rationales cite the chunk id
in square brackets, e.g. `[disruption_care:SQ-CANCEL-ACCOM-01]`.

Coverage: Singapore Airlines (SQ) and British Airways (BA) across six entitlement categories
— rebooking, accommodation, meals, compensation, refund, contact. Carriers outside SQ/BA fall
through to the `GENERIC-*` chunks below.

## disruption_care:SQ-CANCEL-REBOOK-01
Singapore Airlines, cancellation: free rebooking onto the next available SQ service to the
ticketed destination is provided at no additional fare cost. Where no SQ service operates
within 24 hours, endorsement onto a Star Alliance partner carrier is permitted. See also
[airline_rebooking:SQ-CANCEL-01] for the primary rebooking rule. No fare difference is
collected when the replacement cabin matches or is below the original cabin.

## disruption_care:SQ-CANCEL-ACCOM-01
Singapore Airlines, cancellation requiring an overnight stay: hotel accommodation and
return airport transfers are arranged and paid for by the carrier for each night the
traveller is stranded away from their home city, up to a maximum of 2 nights. The traveller
should present at the SQ transfer or service desk to have the hotel booked for them.
Independently self-booked hotel stays are reimbursed only where prior authorisation was
obtained. Travellers stranded in their home city are not eligible for accommodation.

## disruption_care:SQ-CANCEL-MEALS-01
Singapore Airlines, cancellation: meal vouchers are issued for airport waits exceeding
3 hours attributable to the cancellation. Vouchers are collected at the service desk and
scale with the length of the wait. Travellers already departed on a replacement service
receive standard onboard meal service instead.

## disruption_care:SQ-CANCEL-COMP-01
Singapore Airlines, cancellation: **no fixed cash compensation** is payable under SQ's own
conditions of carriage for a cancellation. Care obligations (rebooking, accommodation,
meals) apply instead, per [disruption_care:SQ-CANCEL-ACCOM-01] and
[disruption_care:SQ-CANCEL-MEALS-01]. Services departing the EU or UK may separately fall
under statutory passenger-rights regimes, which the carrier assesses case by case; the
advisor must not state a figure for such a claim.

## disruption_care:SQ-CANCEL-REFUND-01
Singapore Airlines, cancellation: the traveller may decline rebooking and instead take a
full refund of the unflown portion of the ticket, including taxes and carrier surcharges.
Choosing a refund ends the carrier's care obligations, so onward accommodation and meals
cease from the point the refund is accepted.

## disruption_care:SQ-DELAY-MEALS-01
Singapore Airlines, delay: meal vouchers are issued once the departure delay exceeds
3 hours. Delays beyond 6 hours overnight additionally attract accommodation on the same
terms as [disruption_care:SQ-CANCEL-ACCOM-01].

## disruption_care:BA-CANCEL-REBOOK-01
British Airways, cancellation: free rebooking within a 72-hour window either side of the
original departure. Re-routing to a nearby airport in the same metropolitan area (for
example LHR to LGW) is permitted where arrival falls inside the same 24-hour period, and
BA covers surface transfer between the substituted airports. See also
[airline_rebooking:BA-CANCEL-01].

## disruption_care:BA-CANCEL-ACCOM-01
British Airways, cancellation requiring an overnight stay: hotel accommodation plus
transfers are provided for each night the traveller is delayed away from home, with no
fixed night cap while the traveller remains awaiting re-routing. Where BA cannot arrange a
hotel directly, reasonable self-booked accommodation is reimbursed on production of
receipts.

## disruption_care:BA-CANCEL-MEALS-01
British Airways, cancellation: refreshments and meals are provided proportionate to the
waiting time, issued as vouchers at the airport. Reasonable receipted meal expenses are
reimbursed where vouchers were not made available.

## disruption_care:BA-CANCEL-COMP-01
British Airways, cancellation: cash compensation may be payable in addition to re-routing
and care where the cancellation was within the carrier's control and notice was short.
The governing figures are those of the EC 261 regime already recorded in
[airline_rebooking:BA-DELAY-01] — the advisor must cite that chunk for any amount rather
than restating a figure here, and must note that eligibility depends on BA's
cause-of-cancellation determination.

## disruption_care:BA-CANCEL-REFUND-01
British Airways, cancellation: a full refund of the unused ticket portion is available as
an alternative to re-routing, payable to the original form of payment. Accepting a refund
terminates BA's onward duty of care, including accommodation and meals.

## disruption_care:CONTACT-SQ-01
Singapore Airlines disruption contact: the traveller should call the SQ disruption and
rebooking desk, whose telephone number for their region is held in the advisor's contact
fixture and must be surfaced from that fixture rather than recalled. In-airport, the SQ
transfer desk handles accommodation and meal vouchers directly and is faster than the
phone line during a mass-disruption event.

## disruption_care:CONTACT-BA-01
British Airways disruption contact: the traveller should call the BA delay and cancellation
line for their region, whose telephone number is held in the advisor's contact fixture.
Expense reimbursement claims and EC 261 compensation claims are filed through BA's online
claim form rather than by phone.

## disruption_care:CONTACT-GENERIC-01
For any carrier without a dedicated contact chunk: direct the traveller to the customer
service telephone number printed on their ticket or booking confirmation, and note that
the advisor holds no verified contact number on file for that carrier. Never state a
telephone number that did not come from the contact fixture.

## disruption_care:GENERIC-CANCEL-CARE-01
Carriers without specific care chunks: apply the traveller's ticket fare rules per
[airline_rebooking:GENERIC-01] for rebooking, and treat accommodation, meals, and cash
compensation as unverified. State that these are unconfirmed for the carrier rather than
assuming either that they apply or that they do not.

## disruption_care:GENERIC-REFUND-01
Carriers without specific refund chunks: a refund of the unflown portion is commonly
available as an alternative to rebooking, but is governed by the fare rules of the
specific ticket. Basic or restricted economy fares may forfeit value on cancellation.
Treat the refund route as requiring confirmation with the carrier.

## disruption_care:GENERIC-NOT-COVERED-01
Reporting rule for the advisor: where the policy corpus contains no chunk supporting an
entitlement category for the carrier and disruption type in question, the advisor must
write the exact phrase `NOT COVERED BY POLICY CORPUS` for that category, followed by a
direction to confirm with the carrier. Omitting the category entirely is not permitted,
because an absent row reads as a clean answer. Never substitute general knowledge of
passenger-rights law for a missing chunk.
