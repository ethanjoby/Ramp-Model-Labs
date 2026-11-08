"""Procedural generators for common financial workflows.

These let the platform produce realistic, messy training data with zero LLM spend —
the "cheap" path. When an LLM is configured, the data agent blends these with
LLM-synthesized examples for broader coverage.

Every entity list (merchants, vendors, counterparties, people, phrasings of a risk flag, a support request
or a reconciliation note) is split into two pools that never overlap: ``"train"`` feeds training rows,
``"eval"`` feeds an "unseen" evaluation slice. A model scored on the eval pool meets merchants it has never
read and sentences it has never seen, which is what production looks like a week after launch. Formatting
noise (processor prefixes, casing, store numbers, cities, amounts, memos) is shared by both pools because it
is not what a model should be learning from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Callable, Sequence

POOLS = ("train", "eval")


def _pool(items: Sequence[str], pool: str) -> list[str]:
    """Deterministic split of one list: every fifth item (positions 4, 9, 14, ...) is held out for
    evaluation, the rest train. Lists shorter than five keep their last item for evaluation so neither
    pool is ever empty. The same rule serves the generators and ``Template.entities``."""
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    if len(items) < 5:
        ev, tr = [items[-1]], list(items[:-1])
    else:
        ev = [x for i, x in enumerate(items) if i % 5 == 4]
        tr = [x for i, x in enumerate(items) if i % 5 != 4]
    return ev if pool == "eval" else tr


CITIES = [
    ("SAN FRANCISCO", "CA"), ("NEW YORK", "NY"), ("AUSTIN", "TX"), ("SEATTLE", "WA"),
    ("CHICAGO", "IL"), ("BOSTON", "MA"), ("DENVER", "CO"), ("MIAMI", "FL"),
    ("LOS ANGELES", "CA"), ("ATLANTA", "GA"), ("PORTLAND", "OR"), ("BROOKLYN", "NY"),
    ("PHOENIX", "AZ"), ("DALLAS", "TX"), ("SAN JOSE", "CA"), ("NASHVILLE", "TN"),
    ("MINNEAPOLIS", "MN"), ("PHILADELPHIA", "PA"), ("SALT LAKE CITY", "UT"), ("RALEIGH", "NC"),
]


def _city(r: random.Random) -> str:
    c, s = r.choice(CITIES)
    style = r.random()
    if style < 0.4:
        return f"{c} {s}"
    if style < 0.7:
        return f"{c[:13]} {s}"
    return s


def _ref(r: random.Random, n: int = 6) -> str:
    return "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(n))


def _amount(r: random.Random, lo: float, hi: float) -> str:
    return f"${r.uniform(lo, hi):,.2f}"


# ----------------------------------------------------------------------------------
# Expense categorization
# ----------------------------------------------------------------------------------

# Card descriptors as they arrive from the processor. Deliberate near-misses are labeled the way a controller
# would code them: UBER *EATS and DD *DOORDASH are meals, UBER *TRIP is ground transport; APPLE.COM/BILL is a
# subscription, APPLE STORE is hardware; AMZN MKTP is office supplies by default, AMAZON KINDLE is books.
EXPENSE_MERCHANTS: dict[str, tuple[list[str], tuple[float, float]]] = {
    "Software & SaaS": (["FIGMA", "NOTION LABS", "SLACK TECHNOLOGIES", "GITHUB", "LINEAR", "ATLASSIAN",
                          "ZOOM.US", "DOCUSIGN", "1PASSWORD", "LOOM", "VERCEL", "RETOOL", "CALENDLY",
                          "ADOBE *CREATIVE CLD", "GOOGLE *WORKSPACE", "MSFT *M365", "OPENAI *CHATGPT",
                          "ANTHROPIC", "INTERCOM", "HUBSPOT", "APPLE.COM/BILL", "ASANA.COM", "MIRO.COM",
                          "AIRTABLE", "DROPBOX*BUSINESS", "ZAPIER.COM", "JETBRAINS", "CANVA* PRO",
                          "GRAMMARLY", "POSTMAN"], (8, 1800)),
    "Cloud Infrastructure": (["AWS", "AMAZON WEB SERVICES", "GOOGLE *CLOUD", "MSFT *AZURE", "DIGITALOCEAN",
                               "CLOUDFLARE", "SNOWFLAKE", "DATADOG", "MONGODB CLOUD", "HEROKU",
                               "SUPABASE", "MODAL LABS", "TWILIO SENDGRID", "FLY.IO", "RENDER.COM",
                               "PLANETSCALE", "CONFLUENT CLOUD", "FASTLY", "LINODE*AKAMAI", "SENTRY.IO",
                               "NEW RELIC", "PAGERDUTY", "ELASTIC CLOUD", "HETZNER ONLINE"], (20, 24000)),
    "Travel - Airfare": (["UNITED AIRLINES", "DELTA AIR", "AMERICAN AIR", "SOUTHWEST AIR", "ALASKA AIR",
                           "JETBLUE", "LUFTHANSA", "AIR CANADA", "EXPEDIA *AIR", "UA INFLT WIFI",
                           "BRITISH AIRWAYS", "AIR FRANCE", "SPIRIT AIRL", "FRONTIER AIRLINES", "KLM ROYAL DUTCH",
                           "EMIRATES", "VIRGIN ATLANTIC", "KAYAK *FLIGHT", "CATHAY PACIFIC", "SINGAPORE AIRLINES"], (89, 2400)),
    "Travel - Lodging": (["MARRIOTT", "HILTON HOTELS", "HYATT", "AIRBNB *HM", "ACE HOTEL", "KIMPTON",
                           "FOUR SEASONS", "HOTEL TONIGHT", "BOOKING.COM", "W HOTELS", "SONDER", "WESTIN",
                           "SHERATON", "HAMPTON INN", "HOLIDAY INN EXPRESS", "VRBO", "EXPEDIA *HOTEL",
                           "THE STANDARD", "FAIRMONT", "RITZ-CARLTON"], (120, 3200)),
    "Ground Transportation": (["UBER *TRIP", "LYFT *RIDE", "CURB TAXI", "AMTRAK", "HERTZ RENT-A-CAR",
                                "ENTERPRISE RENT", "CLIPPER TRANSIT", "MTA*NYCT PAYGO", "SFO PARKING", "CHEVRON",
                                "SHELL OIL", "AVIS RENT A CAR", "NATIONAL CAR RENTAL", "LAZ PARKING", "SPOTHERO",
                                "EXXONMOBIL", "BP FUEL", "76 FUEL", "CALTRAIN", "NJ TRANSIT", "YELLOW CAB", "TURO",
                                "ZIPCAR", "GETAROUND"], (3, 450)),
    "Meals & Entertainment": (["BLUE BOTTLE COFFEE", "SWEETGREEN", "CHIPOTLE", "STARBUCKS", "UBER *EATS",
                                "NOBU", "SHAKE SHACK", "BALTHAZAR", "TARTINE BAKERY", "GRUBHUB",
                                "THE MODERN", "JOE'S PIZZA", "RITUAL COFFEE", "SPIN NYC", "DD *DOORDASH",
                                "CAVIAR", "SEAMLESS", "PANERA BREAD", "DIG INN", "CHICK-FIL-A", "OLIVE GARDEN",
                                "PHILZ COFFEE", "PRET A MANGER", "TOPGOLF", "DAVE & BUSTERS", "LEVAIN BAKERY"], (4, 1400)),
    "Office Supplies": (["STAPLES", "OFFICE DEPOT", "AMZN MKTP US", "BEST BUY", "ULINE", "IKEA",
                          "THE CONTAINER STORE", "B&H PHOTO", "APPLE STORE", "AMAZON.COM*", "COSTCO WHSE",
                          "TARGET", "HOME DEPOT", "WALMART", "CDW DIRECT", "DELL MARKETING", "LOGITECH",
                          "WAYFAIR", "MICRO CENTER", "QUILL CORP"], (6, 2600)),
    "Advertising & Marketing": (["GOOGLE *ADS", "FACEBK *ADS", "LINKEDIN *ADS", "TWITTER ADS", "REDDIT ADS",
                                  "VISTAPRINT", "STICKER MULE", "MAILCHIMP", "TIKTOK ADS", "META *ADS",
                                  "BING ADS", "SEMRUSH", "AHREFS", "HOOTSUITE", "KLAVIYO", "SPROUT SOCIAL",
                                  "PRINTFUL", "MOO.COM", "4IMPRINT", "YELP ADS", "AMAZON ADS"], (25, 15000)),
    "Professional Services": (["COOLEY LLP", "WILSON SONSINI", "UPWORK", "FIVERR", "DELOITTE", "PILOT.COM",
                                "GUSTO", "CARTA", "LEGALZOOM", "TOPTAL", "CLERKY", "KPMG", "PWC", "ERNST & YOUNG",
                                "ORRICK", "GUNDERSON DETTMER", "BENCH ACCOUNTING", "JUSTWORKS", "RIPPLING",
                                "TRINET", "ADP PAYROLL"], (99, 25000)),
    "Utilities & Telecom": (["COMCAST BUSINESS", "VERIZON WRLS", "AT&T", "T-MOBILE", "PG&E",
                              "CON EDISON", "GOOGLE FIBER", "SONIC.NET", "RINGCENTRAL", "SPECTRUM BUSINESS",
                              "LUMEN", "DUKE ENERGY", "SDG&E", "NATIONAL GRID", "COX BUSINESS", "VONAGE",
                              "DIALPAD", "8X8 INC", "WASTE MANAGEMENT", "XFINITY"], (30, 900)),
    "Shipping & Postage": (["FEDEX", "UPS", "USPS PO", "DHL EXPRESS", "SHIPPO", "STAMPS.COM",
                             "SHIPBOB", "PIRATE SHIP", "EASYPOST", "ONTRAC", "LASERSHIP", "PITNEY BOWES",
                             "SENDLE", "FLEXPORT", "USPS.COM CLICKNSHIP", "DHL ECOMMERCE"], (5, 900)),
    "Training & Education": (["COURSERA", "UDEMY", "O'REILLY MEDIA", "MASTERCLASS", "LINKEDIN LEARNING",
                               "MAVEN", "EVENTBRITE", "SAASTR", "DREAMFORCE REG", "AMAZON KINDLE", "PLURALSIGHT",
                               "DATACAMP", "SKILLSHARE", "GENERAL ASSEMBLY", "AUDIBLE", "HOPIN", "CVENT", "EDX",
                               "CODECADEMY", "FRONTEND MASTERS"], (12, 2500)),
}
EXPENSE_LABELS = list(EXPENSE_MERCHANTS)

_PREFIXES = ["", "", "", "SQ *", "TST* ", "PAYPAL *", "SP ", "PY *", "IC* "]
_MEMOS = {
    "Software & SaaS": ["annual plan", "seat upgrade", "team plan", "monthly sub", "", "", "renewal", "api credits"],
    "Cloud Infrastructure": ["prod usage", "compute", "monthly bill", "", "", "storage + egress", "staging env"],
    "Travel - Airfare": ["offsite flight", "customer visit", "conf travel", "", "seat upgrade", "bag fee"],
    "Travel - Lodging": ["2 nights", "offsite hotel", "client visit", "", "conference stay", "late checkout"],
    "Ground Transportation": ["to airport", "client mtg", "", "", "late night", "rental for site visit"],
    "Meals & Entertainment": ["team lunch", "client dinner", "coffee w/ candidate", "", "", "offsite snacks", "happy hour"],
    "Office Supplies": ["monitors", "desk stuff", "new hire kit", "", "chairs", "printer toner"],
    "Advertising & Marketing": ["q3 campaign", "launch ads", "swag", "", "retargeting", "brand refresh"],
    "Professional Services": ["legal retainer", "contractor", "bookkeeping", "", "audit", "payroll fees"],
    "Utilities & Telecom": ["office internet", "phone plan", "", "", "electric", "water"],
    "Shipping & Postage": ["ship laptops", "returns", "", "customer samples", "postage"],
    "Training & Education": ["course", "conference ticket", "books", "", "workshop", "certification"],
}


def gen_expense(r: random.Random, noise: bool = True, pool: str = "train") -> tuple[str, str]:
    label = r.choice(EXPENSE_LABELS)
    merchants, (lo, hi) = EXPENSE_MERCHANTS[label]
    m = r.choice(_pool(merchants, pool))
    # realistic ambiguity: Amazon marketplace can be office supplies or training (kindle)
    _alt = r.choice(EXPENSE_LABELS)  # drawn in both modes so noise on/off share one text sequence
    flip = r.random() < 0.035 and noise  # applied after the text is built: noise flips the label, never the row
    prefix = r.choice(_PREFIXES) if "*" not in m else ""
    store = f" #{r.randint(100, 9999)}" if r.random() < 0.3 else ""
    ref = f"*{_ref(r)}" if r.random() < 0.25 else ""
    desc = f"{prefix}{m}{ref}{store} {_city(r)}".strip()
    if r.random() < 0.15:
        desc = desc.title()
    memo = r.choice(_MEMOS[label])
    text = f"{desc} | {_amount(r, lo, hi)}" + (f" | memo: {memo}" if memo else "")
    return text, (_alt if flip else label)


# ----------------------------------------------------------------------------------
# Fraud / anomaly screening
# ----------------------------------------------------------------------------------

FRAUD_RISKY = ["ELECTRONICS-DEPOT-ONLINE", "GIFTCARDMALL", "CRYPTO EXCHANGE*COINFLOW", "LUXE WATCHES INTL",
               "PREPAID TOPUP*", "WIRE*INTL TRANSFER", "GOLDBAR DIRECT", "VPN-PAY*ANON", "SNEAKERDROP RESALE",
               "MONEYGRAM*ONLINE", "BITPAY*MERCH", "EGIFT*DIGITAL CODES"]
FRAUD_NORMAL = ["WHOLE FOODS", "DELTA AIR", "AWS", "UBER *TRIP", "STAPLES", "SWEETGREEN", "MARRIOTT", "FEDEX",
                "SLACK TECHNOLOGIES", "SHELL OIL", "COSTCO WHSE", "HOME DEPOT", "LYFT *RIDE", "HILTON HOTELS",
                "UNITED AIRLINES"]
# Each risk flag keeps one weight and several wordings; the pool picks the wording, never the weight.
FRAUD_FLAGS: list[tuple[list[str], float]] = [
    (["new device fingerprint", "device not seen before on this account", "unrecognized device"], 0.9),
    (["shipping address != billing", "ship-to differs from bill-to", "shipping and billing addresses do not match"], 1.0),
    (["{n} declined attempts in last 10 min", "{n} declines in the past 10 minutes", "{n} failed auths within 10 min"], 1.4),
    (["IP geolocates to {country}", "IP address resolves to {country}", "login IP located in {country}"], 1.1),
    (["first purchase at merchant", "merchant never used before", "no prior history with this merchant"], 0.3),
    (["card added to wallet {h}h ago", "card provisioned to wallet {h} hours ago", "wallet token created {h}h before purchase"], 1.0),
    (["velocity: {n} txns in 5 min", "{n} transactions inside 5 minutes", "burst of {n} charges in 5 min"], 1.2),
    (["employee traveling (matches itinerary)", "cardholder on booked trip, location matches", "itinerary on file covers this city"], -1.2),
    (["recurring vendor, 14 prior payments", "established vendor, paid 14 times before", "regular monthly charge from this vendor"], -1.5),
    (["amount matches PO #{po}", "amount equals open PO #{po}", "ties out to purchase order {po}"], -1.3),
    (["billing zip verified", "AVS zip match", "postal code confirmed by issuer"], -0.6),
    (["3DS challenge passed", "3-D Secure authentication succeeded", "cardholder passed step-up auth"], -1.0),
]
FRAUD_NO_SIGNAL = ["no prior alerts", "nothing unusual on file", "no risk flags raised"]


def gen_fraud(r: random.Random, noise: bool = True, pool: str = "train") -> tuple[str, str]:
    fraud = r.random() < 0.42
    signals: list[str] = []
    score = 0.0
    channel = r.choice(["card-present", "card-not-present", "card-not-present", "contactless"])
    if channel == "card-not-present":
        score += 0.4
    hour = r.randint(0, 23)
    if hour < 5:
        score += 0.7
    amt = r.lognormvariate(4.2, 1.3)
    if amt > 1500:
        score += 0.8
    risky = r.choice(_pool(FRAUD_RISKY, pool))
    normal = r.choice(_pool(FRAUD_NORMAL, pool))
    merchant = risky if r.random() < (0.75 if fraud else 0.08) else normal
    if merchant == risky:
        score += 1.2
    k = r.randint(1, 4)
    for wordings, w in r.sample(FRAUD_FLAGS, k):
        phrase = r.choice(_pool(wordings, pool))
        if (w > 0) == fraud or r.random() < 0.18:
            signals.append(phrase.format(n=r.randint(2, 7), country=r.choice(["RO", "NG", "VN", "BR", "RU"]),
                                         h=r.randint(1, 6), po=r.randint(1000, 9999)))
            score += w
    if not signals:
        signals.append(r.choice(_pool(FRAUD_NO_SIGNAL, pool)))
    label = "fraud" if (score > 1.9) else "legitimate"
    if r.random() < 0.04 and noise:  # the draw always happens, so noise on/off share one text sequence
        label = "legitimate" if label == "fraud" else "fraud"
    text = (f"{channel} {'$'}{amt:,.2f} at {merchant} {hour:02d}:{r.randint(0, 59):02d} local; "
            + "; ".join(signals))
    return text, label


# ----------------------------------------------------------------------------------
# Support ticket routing
# ----------------------------------------------------------------------------------

# Ten phrasings per intent in mixed registers. Positions 4 and 9 (the eval pool) always carry a placeholder,
# so the unseen slice has as much surface variety as the training one.
SUPPORT_INTENTS: dict[str, list[str]] = {
    "card_declined": [
        "my card got declined at {merchant} and i'm standing at the register",
        "why was my corporate card declined? it says {amount} was rejected",
        "Transaction at {merchant} failed even though I have budget left",
        "card keeps getting denied for online purchases",
        "declined again?? this is the third time today at {merchant}",
        "Card was rejected at checkout, the error just says do not honor",
        "my virtual card won't go through on {merchant}, is it blocked?",
        "got a decline text for {amount} at {merchant} but I'm travelling and need this to work",
        "physical card declined twice at the hotel front desk this morning",
        "trying to pay {merchant} {amount} and it fails every time, what gives",
    ],
    "reimbursement": [
        "how do I get reimbursed for a {merchant} receipt I paid with personal card",
        "submitted an out of pocket expense for {amount} two weeks ago, when do I get paid",
        "my reimbursement was rejected, can you tell me why",
        "need to file mileage reimbursement for client visits",
        "personal card used for {merchant}, want to be paid back",
        "paid {amount} out of pocket for team dinner, how do i claim it",
        "reimbursement shows approved but nothing hit my bank account yet",
        "can i get paid back for a {merchant} purchase from last month or is it too late",
        "where do I upload receipts for out-of-pocket expenses",
        "expensed {amount} for {merchant} on my own card, still waiting on repayment",
    ],
    "bill_pay": [
        "vendor invoice from {vendor} is stuck in approval",
        "how do I schedule an ACH payment to {vendor} for {amount}",
        "we paid {vendor} twice for the same invoice",
        "need to change the remittance bank account for {vendor}",
        "can I pay an international invoice in EUR to {vendor}",
        "{vendor} says they never received the {amount} wire we sent last week",
        "how do I cancel a scheduled bill payment before it goes out",
        "invoice from {vendor} needs to be paid by check, not ACH",
        "who approves vendor bills over {amount} on our account",
        "{vendor} invoice for {amount} is overdue, can we expedite the payment",
    ],
    "accounting_integration": [
        "netsuite sync is failing with an error on GL codes",
        "transactions aren't showing up in QuickBooks",
        "how do I map {merchant} to a different GL account",
        "Xero integration disconnected after we changed admins",
        "class and department fields not syncing to the ERP for {merchant} charges",
        "the Sage Intacct export has every {merchant} transaction coded to suspense",
        "duplicate journal entries after re-running the accounting sync",
        "how do I set the default GL account for {merchant} transactions",
        "quickbooks shows the {amount} charge twice, once from the bank feed and once from you",
        "need to re-map {merchant} to the right expense account before close",
    ],
    "account_access": [
        "locked out of my account after resetting my phone",
        "can't log in, 2fa code never arrives",
        "need to add {coworker} as a new admin on our business account",
        "SSO with okta isn't working for new hires like {coworker}",
        "how do i change the email on {coworker}'s login",
        "{coworker} left the company, how do we remove their access and cards",
        "password reset link expired before I could use it",
        "my authenticator app was wiped, need 2fa reset",
        "{coworker} is getting 'access denied' on the admin dashboard",
        "can you unlock {coworker}'s account, too many failed login attempts",
    ],
    "limits_and_policy": [
        "can you raise my card limit to {amount} for the offsite",
        "what's our policy on alcohol at team dinners",
        "need a temporary limit increase for travel next week",
        "is {merchant} an allowed merchant category under our policy",
        "can I spend {amount} at {merchant} without pre-approval",
        "my manager says my spend limit resets monthly, is that right",
        "is there a per-transaction cap on my card? {merchant} blocked a {amount} purchase",
        "what's the limit on client gifts under the expense policy",
        "need my monthly limit bumped above {amount} for the conference sponsorship",
        "does policy allow booking {merchant} for business travel, or do i need approval",
    ],
    "receipts": [
        "receipt for {merchant} won't attach to the transaction",
        "how do I forward receipts by text message",
        "missing receipt reminder for a {amount} charge i already uploaded",
        "the OCR read my receipt total wrong",
        "can I submit an itemized receipt for the {merchant} charge later",
        "lost the paper receipt from {merchant}, what do i do",
        "receipt matching put my {amount} receipt on the wrong transaction",
        "do i need a receipt for charges under {amount}",
        "email receipts from {merchant} aren't auto-matching anymore",
        "uploaded a receipt for {amount} but it still shows as missing",
    ],
}
SUPPORT_LABELS = list(SUPPORT_INTENTS)
SUPPORT_MERCHANTS = ["Delta", "Uber", "Home Depot", "AWS", "a hotel", "Costco", "WeWork", "Amazon", "Marriott", "Lyft",
                     "Staples", "Zoom", "the airport parking", "Starbucks", "Best Buy"]
SUPPORT_VENDORS = ["Acme Logistics", "Brightline Legal", "Northwind Supply", "Globex", "Initech", "Stark Facilities",
                   "Umbrella Staffing", "Hooli Cloud", "Vandelay Industries", "Wonka Supply Co"]
SUPPORT_PEOPLE = ["Priya", "Marcus", "Elena", "Tom", "Aisha", "Diego", "Hannah", "Kenji", "Sofia", "Omar", "Grace", "Liam"]


def gen_support(r: random.Random, noise: bool = True, pool: str = "train") -> tuple[str, str]:
    label = r.choice(SUPPORT_LABELS)
    t = r.choice(_pool(SUPPORT_INTENTS[label], pool)).format(
        merchant=r.choice(_pool(SUPPORT_MERCHANTS, pool)),
        vendor=r.choice(_pool(SUPPORT_VENDORS, pool)),
        coworker=r.choice(_pool(SUPPORT_PEOPLE, pool)),
        amount=_amount(r, 40, 9000),
    )
    style = r.random()
    if style < 0.25:
        t = t.lower()
    elif style < 0.37:
        t = "Hi team — " + t + ". Thanks!"
    elif style < 0.46:
        t = t + " asap please"
    elif style < 0.54:
        t = "hey, " + t
    elif style < 0.6:
        t = "URGENT: " + t
    elif style < 0.66:
        t = t + " thx"
    _alt = r.choice(SUPPORT_LABELS)  # drawn in both modes so noise on/off share one text sequence
    if r.random() < 0.05 and noise:
        label = _alt
    return t, label


# ----------------------------------------------------------------------------------
# AP invoice triage
# ----------------------------------------------------------------------------------

INVOICE_VENDORS = ["Acme Logistics", "Brightline Legal LLP", "Northwind Supply", "Globex Corp", "Initech Consulting",
                   "Stark Facilities", "Umbrella Staffing", "Hooli Cloud", "Vandelay Industries", "Pied Piper Hosting",
                   "Wayne Industrial", "Tyrell Office Systems", "Cyberdyne Systems", "Soylent Catering", "Oscorp Labs",
                   "Dunder Mifflin Paper", "Prestige Worldwide", "Gekko Advisors", "Massive Dynamic", "Sterling Cooper"]
INVOICE_NOTES: dict[str, list[str]] = {
    "duplicate": [
        "same invoice # seen on {d} for same amount", "amount + vendor match payment 3 days ago",
        "invoice number INV-{n} previously paid", "identical total already settled on {d}",
        "exact duplicate of a bill approved last week", "vendor resubmitted INV-{n} after it was paid",
        "matches a cleared payment from {d}, same reference", "this invoice number appears twice in the AP queue",
        "same PO, same amount, already paid on {d}", "second copy of INV-{n}, original paid in full",
    ],
    "needs_review": [
        "amount {p}% over PO", "no PO attached", "new vendor, bank details changed", "line items don't sum to total",
        "unusual GL code for vendor", "quantity billed exceeds quantity received", "tax rate differs from vendor master",
        "PO closed, invoice still arriving", "missing approver signature on the backup",
        "total {p}% above last month's bill from this vendor",
    ],
    "reject": [
        "vendor not in approved list", "services dated before contract start", "invoice addressed to a different entity",
        "sanctioned country remit-to", "bill-to name does not match any of our entities",
        "contract terminated, no services authorized", "charges for a cancelled order",
        "remit-to account flagged by bank as fraudulent", "invoice is for a competitor's subsidiary, sent in error",
        "work never authorized, no PO and no contract",
    ],
    "approve": [
        "matches PO and receipt (3-way match)", "recurring monthly retainer", "within contract terms", "approved budget line",
        "3-way match clean, goods received in full", "standing order, amount unchanged from prior months",
        "matches signed SOW milestone", "receipt confirmed by warehouse, PO open", "pre-approved by department head",
        "quantities and unit prices agree with PO",
    ],
}
INVOICE_LABELS = ["approve", "needs_review", "duplicate", "reject"]
INVOICE_TERMS = ["net 30", "due in 5 days", "early pay discount 2%", "ACH", "net 45", "wire", "due on receipt"]


def gen_invoice(r: random.Random, noise: bool = True, pool: str = "train") -> tuple[str, str]:
    vendor = r.choice(_pool(INVOICE_VENDORS, pool))
    amt = r.lognormvariate(7.6, 1.1)
    po = r.random() < 0.7
    roll = r.random()
    label = "duplicate" if roll < 0.18 else "needs_review" if roll < 0.36 else "reject" if roll < 0.46 else "approve"
    note = r.choice(_pool(INVOICE_NOTES[label], pool))
    if label == "approve" and r.random() < 0.2:
        note = ""  # a clean invoice often arrives with no note at all
    notes = [note] if note else []
    if r.random() < 0.2:
        notes.append(r.choice(INVOICE_TERMS))
    notes = [n.format(d=f"10/{r.randint(1, 28)}", n=r.randint(10000, 99999), p=r.randint(8, 60)) for n in notes]
    text = (f"{vendor} INV-{r.randint(10000, 99999)} {'$'}{amt:,.2f}"
            + (f" PO-{r.randint(1000, 9999)}" if po else "") + ("; " + "; ".join(notes) if notes else ""))
    _alt = r.choice(INVOICE_LABELS)  # drawn in both modes so noise on/off share one text sequence
    if r.random() < 0.04 and noise:
        label = _alt
    return text, label


# ----------------------------------------------------------------------------------
# Transaction reconciliation
# ----------------------------------------------------------------------------------

RECON_PARTIES = ["ACME LOGISTICS", "NORTHWIND SUPPLY", "GLOBEX CORP", "STARK FACILITIES", "HOOLI CLOUD",
                 "UMBRELLA STAFFING", "BRIGHTLINE LEGAL", "AWS", "GUSTO PAYROLL", "WEWORK",
                 "VANDELAY INDUSTRIES", "PIED PIPER HOSTING", "TYRELL OFFICE SYSTEMS", "SOYLENT CATERING", "OSCORP LABS",
                 "DUNDER MIFFLIN", "MASSIVE DYNAMIC", "STERLING COOPER", "CYBERDYNE SYSTEMS", "WAYNE INDUSTRIAL"]
# The engine-computed part of a delta ("diff $12.40; 2d apart") is shared vocabulary; the reason a reviewer
# writes next to it is what the pools split.
RECON_NOTES: dict[str, list[str]] = {
    "matched": ["exact match", "auto-matched on ref", "same ref, same amount", "cleared as expected", "clean match",
                "ref and amount agree", "no exceptions", "matched by engine", "ties out", "straight match"],
    "amount_mismatch": ["wire fee?", "short pay", "fx variance", "partial payment", "bank fee deducted",
                        "possible discount taken", "paid less than booked", "rounding on fx", "vendor credit applied",
                        "disputed line item"],
    "timing_difference": ["crosses period end", "outstanding check", "deposit in transit", "month-end cutoff", "check float",
                          "bank posted next period", "ACH settled after close", "cutoff timing", "in transit at period end",
                          "outstanding at close"],
    "duplicate": ["second bank line for same ref", "posted twice", "ref already cleared", "already matched to another bank line",
                  "vendor charged twice", "repeat of a cleared transaction", "original matched earlier",
                  "two bank lines, one ledger entry", "already reconciled under same ref", "first occurrence already cleared"],
    "missing_in_ledger": ["no ledger match above threshold", "unrecorded bank fee", "no open item for payee",
                          "nothing booked for this payee in the period", "bank line has no counterpart in GL",
                          "no candidate within tolerance; likely unbooked charge", "payee unknown to the ledger",
                          "no GL entry on or near this date", "unposted: no invoice or bill on file", "no match; bank-only transaction"],
}
RECON_LABELS = ["matched", "amount_mismatch", "timing_difference", "duplicate", "missing_in_ledger"]


def gen_reconciliation(r: random.Random, noise: bool = True, pool: str = "train") -> tuple[str, str]:
    """A bank line against its closest ledger entry, with the deltas a matching engine computes."""
    party = r.choice(_pool(RECON_PARTIES, pool))
    method = r.choice(["ACH", "WIRE", "CHECK", "CARD"])
    amt = r.lognormvariate(7.2, 1.2)
    ref = f"{r.randint(10000, 99999)}"
    label = r.choice(["matched", "matched", "amount_mismatch", "timing_difference", "duplicate", "missing_in_ledger"])
    bank = f"BANK {method} {party} ${amt:,.2f} REF {ref}"
    note = r.choice(_pool(RECON_NOTES[label], pool))
    if label == "matched":
        gl = f"GL {party.title()} ${amt:,.2f} REF {ref}"
        delta = f"diff $0.00; {r.randint(0, 2)}d apart" + (f"; {note}" if r.random() < 0.6 else "")
    elif label == "amount_mismatch":
        d = amt * r.uniform(0.005, 0.2)
        gl = f"GL {party.title()} ${amt + d:,.2f} REF {ref}"
        delta = f"diff ${d:,.2f}; {r.randint(0, 2)}d apart; {note}"
    elif label == "timing_difference":
        gl = f"GL {party.title()} ${amt:,.2f} REF {ref}"
        delta = f"diff $0.00; {r.randint(6, 40)}d apart; {note}"
    elif label == "duplicate":
        gl = f"GL {party.title()} ${amt:,.2f} REF {ref}"
        delta = f"diff $0.00; {r.randint(0, 3)}d apart; {note}"
    else:
        gl = "GL no candidate entry"
        delta = note
    _alt = r.choice(RECON_LABELS)  # drawn in both modes so noise on/off share one text sequence
    if r.random() < 0.04 and noise:
        label = _alt
    return f"{bank} | {gl} | {delta}", label


@dataclass
class Template:
    key: str
    name: str
    description: str
    labels: list[str]
    input_hint: str
    generator: Callable[[random.Random, bool, str], tuple[str, str]]
    keywords: list[str] = field(default_factory=list)
    # Every list the generator pools: merchants, counterparties, people, phrasings. Each is split on its own.
    entity_sources: list[Sequence[str]] = field(default_factory=list)

    def entities(self, pool: str) -> set[str]:
        """The surface strings rows from `pool` may contain and rows from the other pool never do."""
        return {e for src in self.entity_sources for e in _pool(src, pool) if e}


TEMPLATES: dict[str, Template] = {
    "transaction_reconciliation": Template(
        "transaction_reconciliation", "Transaction Reconciler",
        "Compare a bank line with its closest ledger entry and decide how the pair reconciles.",
        RECON_LABELS,
        "bank line | ledger entry | computed deltas", gen_reconciliation,
        ["reconcil", "recon ", "bank statement", "bank feed", "unmatched", "three-way", "close the books"],
        entity_sources=[RECON_PARTIES, *RECON_NOTES.values()],
    ),
    "expense_categorization": Template(
        "expense_categorization", "Expense Categorizer",
        "Classify raw card transactions into GL expense categories.",
        list(EXPENSE_MERCHANTS), "card descriptor | amount | memo", gen_expense,
        ["expense", "categor", "gl ", "gl code", "transaction", "merchant", "spend", "bookkeep", "ledger", "coding"],
        entity_sources=[ms for ms, _ in EXPENSE_MERCHANTS.values()],
    ),
    "fraud_detection": Template(
        "fraud_detection", "Fraud Screener",
        "Flag card transactions as fraud or legitimate from risk signals.",
        ["fraud", "legitimate"], "transaction + risk signals", gen_fraud,
        ["fraud", "suspicious", "anomal", "risk", "chargeback", "scam", "stolen"],
        entity_sources=[FRAUD_RISKY, FRAUD_NORMAL, *[w for w, _ in FRAUD_FLAGS], FRAUD_NO_SIGNAL],
    ),
    "support_routing": Template(
        "support_routing", "Support Router",
        "Route customer support messages to the right financial-ops queue.",
        list(SUPPORT_INTENTS), "customer message", gen_support,
        ["support", "ticket", "route", "routing", "customer", "intent", "help desk", "inbox"],
        entity_sources=[*SUPPORT_INTENTS.values(), SUPPORT_MERCHANTS, SUPPORT_VENDORS, SUPPORT_PEOPLE],
    ),
    "invoice_triage": Template(
        "invoice_triage", "AP Invoice Triage",
        "Decide whether an incoming vendor invoice should be approved, reviewed, flagged duplicate, or rejected.",
        INVOICE_LABELS, "invoice summary", gen_invoice,
        ["invoice", "accounts payable", " ap ", "bill", "vendor", "duplicate", "payable"],
        entity_sources=[INVOICE_VENDORS, *INVOICE_NOTES.values()],
    ),
}


# Vocabulary shared across workflows can only break ties; every other keyword, and each template's own label
# names, pins the prompt to one workflow. Scored on presence, so repeating "transaction" never beats "fraud".
GENERIC_KEYWORDS = frozenset({
    "transaction", "bill", "vendor", "customer", "route", "routing", "merchant", "spend", "ledger", "coding",
    "risk", "intent", "inbox", "duplicate", "bank", "card", "payment",
})
MIN_SCORE, MIN_MARGIN = 2, 1


def _weight(keyword: str) -> int:
    return 1 if keyword.strip() in GENERIC_KEYWORDS else 3


def template_keywords(t: Template) -> dict[str, int]:
    words = {k: _weight(k) for k in t.keywords}
    for label in t.labels:
        phrase = label.lower().replace("_", " ")
        words.setdefault(phrase, _weight(phrase))
    return words


def match_template(prompt: str) -> Template | None:
    """The one built-in workflow a prompt clearly asks for, or None when nothing fits or two fit equally."""
    p = f" {prompt.lower()} "
    ranked = sorted(((sum(w for k, w in template_keywords(t).items() if k in p), t) for t in TEMPLATES.values()),
                    key=lambda st: st[0], reverse=True)
    best_score, best = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0
    if best_score < MIN_SCORE or best_score < runner_up + MIN_MARGIN:
        return None
    return best


def generate(template: Template, n: int, seed: int = 7, noise: bool = True, pool: str = "train") -> list[dict[str, str]]:
    """Rows from a generator.

    `noise=False` turns off the planted label noise, which is what held-out rows need: a test set with planted
    mislabels punishes correct answers and caps every candidate at the same ceiling. For one seed and pool the
    noisy and clean draws share one text sequence and differ only in labels.

    `pool="eval"` draws only from the entities and phrasings the train pool never uses, for an unseen slice.
    """
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    r = random.Random(seed)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 5:
        attempts += 1
        text, label = template.generator(r, noise, pool)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "label": label, "source": "procedural"})
    return rows
