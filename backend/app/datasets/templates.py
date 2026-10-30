"""Procedural generators for common financial workflows.

These let the platform produce realistic, messy training data with zero LLM spend —
the "cheap" path. When an LLM is configured, the data agent blends these with
LLM-synthesized examples for broader coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Callable

CITIES = [
    ("SAN FRANCISCO", "CA"), ("NEW YORK", "NY"), ("AUSTIN", "TX"), ("SEATTLE", "WA"),
    ("CHICAGO", "IL"), ("BOSTON", "MA"), ("DENVER", "CO"), ("MIAMI", "FL"),
    ("LOS ANGELES", "CA"), ("ATLANTA", "GA"), ("PORTLAND", "OR"), ("BROOKLYN", "NY"),
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

EXPENSE_MERCHANTS: dict[str, tuple[list[str], tuple[float, float]]] = {
    "Software & SaaS": (["FIGMA", "NOTION LABS", "SLACK TECHNOLOGIES", "GITHUB", "LINEAR", "ATLASSIAN",
                          "ZOOM.US", "DOCUSIGN", "1PASSWORD", "LOOM", "VERCEL", "RETOOL", "CALENDLY",
                          "ADOBE *CREATIVE CLD", "GOOGLE *WORKSPACE", "MSFT *M365", "OPENAI *CHATGPT",
                          "ANTHROPIC", "INTERCOM", "HUBSPOT"], (8, 1800)),
    "Cloud Infrastructure": (["AWS", "AMAZON WEB SERVICES", "GOOGLE *CLOUD", "MSFT *AZURE", "DIGITALOCEAN",
                               "CLOUDFLARE", "SNOWFLAKE", "DATADOG", "MONGODB CLOUD", "HEROKU",
                               "SUPABASE", "MODAL LABS", "TWILIO SENDGRID"], (20, 24000)),
    "Travel - Airfare": (["UNITED AIRLINES", "DELTA AIR", "AMERICAN AIR", "SOUTHWEST AIR", "ALASKA AIR",
                           "JETBLUE", "LUFTHANSA", "AIR CANADA", "EXPEDIA *AIR", "UA INFLT WIFI"], (89, 2400)),
    "Travel - Lodging": (["MARRIOTT", "HILTON HOTELS", "HYATT", "AIRBNB *HM", "ACE HOTEL", "KIMPTON",
                           "FOUR SEASONS", "HOTEL TONIGHT", "BOOKING.COM", "W HOTELS", "SONDER"], (120, 3200)),
    "Ground Transportation": (["UBER *TRIP", "LYFT *RIDE", "UBER *EATS PENDING", "CURB TAXI", "AMTRAK",
                                "HERTZ RENT-A-CAR", "ENTERPRISE RENT", "CLIPPER TRANSIT", "MTA*NYCT PAYGO",
                                "SFO PARKING", "CHEVRON", "SHELL OIL"], (3, 450)),
    "Meals & Entertainment": (["BLUE BOTTLE COFFEE", "SWEETGREEN", "CHIPOTLE", "STARBUCKS", "DOORDASH",
                                "NOBU", "SHAKE SHACK", "BALTHAZAR", "TARTINE BAKERY", "GRUBHUB",
                                "THE MODERN", "JOE'S PIZZA", "RITUAL COFFEE", "SPIN NYC"], (4, 1400)),
    "Office Supplies": (["STAPLES", "OFFICE DEPOT", "AMZN MKTP US", "BEST BUY", "UPLINE PRODUCTS",
                          "UNIQLO", "IKEA", "THE CONTAINER STORE", "B&H PHOTO", "APPLE STORE"], (6, 2600)),
    "Advertising & Marketing": (["GOOGLE *ADS", "FACEBK *ADS", "LINKEDIN *ADS", "TWITTER ADS", "REDDIT ADS",
                                  "VISTAPRINT", "STICKER MULE", "CANVA", "MAILCHIMP", "TIKTOK ADS"], (25, 15000)),
    "Professional Services": (["COOLEY LLP", "WILSON SONSINI", "UPWORK", "FIVERR", "DELOITTE", "PILOT.COM",
                                "GUSTO", "CARTA", "LEGALZOOM", "TOPTAL", "CLERKY"], (99, 25000)),
    "Utilities & Telecom": (["COMCAST BUSINESS", "VERIZON WRLS", "AT&T", "T-MOBILE", "PG&E",
                              "CON EDISON", "GOOGLE FIBER", "SONIC.NET", "RINGCENTRAL"], (30, 900)),
    "Shipping & Postage": (["FEDEX", "UPS", "USPS PO", "DHL EXPRESS", "SHIPPO", "STAMPS.COM",
                             "SHIPBOB"], (5, 900)),
    "Training & Education": (["COURSERA", "UDEMY", "O'REILLY MEDIA", "MASTERCLASS", "LINKEDIN LEARNING",
                               "MAVEN", "EVENTBRITE", "SAASTR", "DREAMFORCE REG", "AMAZON KINDLE"], (12, 2500)),
}

_PREFIXES = ["", "", "", "SQ *", "TST* ", "PAYPAL *", "SP ", "PY *", "IC* "]
_MEMOS = {
    "Software & SaaS": ["annual plan", "seat upgrade", "team plan", "monthly sub", "", "", "renewal"],
    "Cloud Infrastructure": ["prod usage", "compute", "monthly bill", "", "", "storage + egress"],
    "Travel - Airfare": ["offsite flight", "customer visit", "conf travel", "", "seat upgrade"],
    "Travel - Lodging": ["2 nights", "offsite hotel", "client visit", "", "conference stay"],
    "Ground Transportation": ["to airport", "client mtg", "", "", "late night"],
    "Meals & Entertainment": ["team lunch", "client dinner", "coffee w/ candidate", "", "", "offsite snacks"],
    "Office Supplies": ["monitors", "desk stuff", "new hire kit", "", "chairs"],
    "Advertising & Marketing": ["q3 campaign", "launch ads", "swag", "", "retargeting"],
    "Professional Services": ["legal retainer", "contractor", "bookkeeping", "", "audit"],
    "Utilities & Telecom": ["office internet", "phone plan", "", "", "electric"],
    "Shipping & Postage": ["ship laptops", "returns", "", "customer samples"],
    "Training & Education": ["course", "conference ticket", "books", "", "workshop"],
}


def gen_expense(r: random.Random) -> tuple[str, str]:
    label = r.choice(list(EXPENSE_MERCHANTS))
    merchants, (lo, hi) = EXPENSE_MERCHANTS[label]
    m = r.choice(merchants)
    # realistic ambiguity: Amazon marketplace can be office supplies or training (kindle)
    if r.random() < 0.035:
        label = r.choice(list(EXPENSE_MERCHANTS))
    prefix = r.choice(_PREFIXES) if "*" not in m else ""
    store = f" #{r.randint(100, 9999)}" if r.random() < 0.3 else ""
    ref = f"*{_ref(r)}" if r.random() < 0.25 else ""
    desc = f"{prefix}{m}{ref}{store} {_city(r)}".strip()
    if r.random() < 0.15:
        desc = desc.title()
    memo = r.choice(_MEMOS[label])
    text = f"{desc} | {_amount(r, lo, hi)}" + (f" | memo: {memo}" if memo else "")
    return text, label


# ----------------------------------------------------------------------------------
# Fraud / anomaly screening
# ----------------------------------------------------------------------------------

def gen_fraud(r: random.Random) -> tuple[str, str]:
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
    risky = r.choice(["ELECTRONICS-DEPOT-ONLINE", "GIFTCARDMALL", "CRYPTO EXCHANGE*COINFLOW",
                      "LUXE WATCHES INTL", "PREPAID TOPUP*", "WIRE*INTL TRANSFER"])
    normal = r.choice(["WHOLE FOODS", "DELTA AIR", "AWS", "UBER *TRIP", "STAPLES", "SWEETGREEN",
                       "MARRIOTT", "FEDEX", "SLACK TECHNOLOGIES", "SHELL OIL"])
    merchant = risky if r.random() < (0.75 if fraud else 0.08) else normal
    if merchant == risky:
        score += 1.2
    flags = [
        ("new device fingerprint", 0.9), ("shipping address != billing", 1.0),
        ("{n} declined attempts in last 10 min", 1.4), ("IP geolocates to {country}", 1.1),
        ("first purchase at merchant", 0.3), ("card added to wallet {h}h ago", 1.0),
        ("velocity: {n} txns in 5 min", 1.2), ("employee traveling (matches itinerary)", -1.2),
        ("recurring vendor, 14 prior payments", -1.5), ("amount matches PO #{po}", -1.3),
        ("billing zip verified", -0.6), ("3DS challenge passed", -1.0),
    ]
    k = r.randint(1, 4)
    for f, w in r.sample(flags, k):
        if (w > 0) == fraud or r.random() < 0.18:
            signals.append(f.format(n=r.randint(2, 7), country=r.choice(["RO", "NG", "VN", "BR", "RU"]),
                                    h=r.randint(1, 6), po=r.randint(1000, 9999)))
            score += w
    if not signals:
        signals.append("no prior alerts")
    label = "fraud" if (score > 1.9) else "legitimate"
    if r.random() < 0.04:
        label = "legitimate" if label == "fraud" else "fraud"
    text = (f"{channel} {'$'}{amt:,.2f} at {merchant} {hour:02d}:{r.randint(0, 59):02d} local; "
            + "; ".join(signals))
    return text, label


# ----------------------------------------------------------------------------------
# Support ticket routing
# ----------------------------------------------------------------------------------

SUPPORT_INTENTS: dict[str, list[str]] = {
    "card_declined": [
        "my card got declined at {merchant} and i'm standing at the register",
        "why was my corporate card declined? it says {amount} was rejected",
        "Transaction at {merchant} failed even though I have budget left",
        "card keeps getting denied for online purchases",
        "declined again?? this is the third time today at {merchant}",
    ],
    "reimbursement": [
        "how do I get reimbursed for a {merchant} receipt I paid with personal card",
        "submitted an out of pocket expense for {amount} two weeks ago, when do I get paid",
        "my reimbursement was rejected, can you tell me why",
        "need to file mileage reimbursement for client visits",
        "personal card used for {merchant}, want to be paid back",
    ],
    "bill_pay": [
        "vendor invoice from {vendor} is stuck in approval",
        "how do I schedule an ACH payment to {vendor} for {amount}",
        "we paid {vendor} twice for the same invoice",
        "need to change the remittance bank account for {vendor}",
        "can I pay an international invoice in EUR",
    ],
    "accounting_integration": [
        "netsuite sync is failing with an error on GL codes",
        "transactions aren't showing up in QuickBooks",
        "how do I map {merchant} to a different GL account",
        "Xero integration disconnected after we changed admins",
        "class and department fields not syncing to the ERP",
    ],
    "account_access": [
        "locked out of my account after resetting my phone",
        "can't log in, 2fa code never arrives",
        "need to add a new admin to our business account",
        "SSO with okta isn't working for new hires",
        "how do i change the email on my login",
    ],
    "limits_and_policy": [
        "can you raise my card limit to {amount} for the offsite",
        "what's our policy on alcohol at team dinners",
        "need a temporary limit increase for travel next week",
        "is {merchant} an allowed merchant category under our policy",
        "my manager says my spend limit resets monthly, is that right",
    ],
    "receipts": [
        "receipt for {merchant} won't attach to the transaction",
        "how do I forward receipts by text message",
        "missing receipt reminder for a {amount} charge i already uploaded",
        "the OCR read my receipt total wrong",
        "can I submit an itemized receipt later",
    ],
}


def gen_support(r: random.Random) -> tuple[str, str]:
    label = r.choice(list(SUPPORT_INTENTS))
    t = r.choice(SUPPORT_INTENTS[label]).format(
        merchant=r.choice(["Delta", "Uber", "Home Depot", "AWS", "a hotel", "Costco", "WeWork", "Amazon"]),
        vendor=r.choice(["Acme Logistics", "Brightline Legal", "Northwind Supply", "Globex", "Initech"]),
        amount=_amount(r, 40, 9000),
    )
    style = r.random()
    if style < 0.3:
        t = t.lower()
    elif style < 0.45:
        t = "Hi team — " + t + ". Thanks!"
    elif style < 0.55:
        t = t + " asap please"
    if r.random() < 0.05:
        label = r.choice(list(SUPPORT_INTENTS))
    return t, label


# ----------------------------------------------------------------------------------
# AP invoice triage
# ----------------------------------------------------------------------------------

def gen_invoice(r: random.Random) -> tuple[str, str]:
    vendor = r.choice(["Acme Logistics", "Brightline Legal LLP", "Northwind Supply", "Globex Corp",
                       "Initech Consulting", "Stark Facilities", "Umbrella Staffing", "Hooli Cloud"])
    amt = r.lognormvariate(7.6, 1.1)
    po = r.random() < 0.7
    label = "approve"
    notes = []
    roll = r.random()
    if roll < 0.18:
        label = "duplicate"
        notes.append(r.choice(["same invoice # seen on {d} for same amount", "amount + vendor match payment 3 days ago",
                               "invoice number INV-{n} previously paid"]))
    elif roll < 0.36:
        label = "needs_review"
        notes.append(r.choice(["amount {p}% over PO", "no PO attached", "new vendor, bank details changed",
                               "line items don't sum to total", "unusual GL code for vendor"]))
    elif roll < 0.46:
        label = "reject"
        notes.append(r.choice(["vendor not in approved list", "services dated before contract start",
                               "invoice addressed to a different entity", "sanctioned country remit-to"]))
    else:
        notes.append(r.choice(["matches PO and receipt (3-way match)", "recurring monthly retainer",
                               "within contract terms", "approved budget line", ""]))
    if r.random() < 0.2:
        notes.append(r.choice(["net 30", "due in 5 days", "early pay discount 2%", "ACH"]))
    notes = [n.format(d=f"10/{r.randint(1, 28)}", n=r.randint(10000, 99999), p=r.randint(8, 60)) for n in notes if n]
    text = (f"{vendor} INV-{r.randint(10000, 99999)} {'$'}{amt:,.2f}"
            + (f" PO-{r.randint(1000, 9999)}" if po else "") + ("; " + "; ".join(notes) if notes else ""))
    if r.random() < 0.04:
        label = r.choice(["approve", "needs_review", "duplicate", "reject"])
    return text, label


# ----------------------------------------------------------------------------------
# Transaction reconciliation
# ----------------------------------------------------------------------------------

def gen_reconciliation(r: random.Random) -> tuple[str, str]:
    """A bank line against its closest ledger entry, with the deltas a matching engine computes."""
    party = r.choice(["ACME LOGISTICS", "NORTHWIND SUPPLY", "GLOBEX CORP", "STARK FACILITIES", "HOOLI CLOUD",
                      "UMBRELLA STAFFING", "BRIGHTLINE LEGAL", "AWS", "GUSTO PAYROLL", "WEWORK"])
    method = r.choice(["ACH", "WIRE", "CHECK", "CARD"])
    amt = r.lognormvariate(7.2, 1.2)
    ref = f"{r.randint(10000, 99999)}"
    label = r.choice(["matched", "matched", "amount_mismatch", "timing_difference", "duplicate", "missing_in_ledger"])
    bank = f"BANK {method} {party} ${amt:,.2f} REF {ref}"
    if label == "matched":
        gl = f"GL {party.title()} ${amt:,.2f} REF {ref}"
        delta = f"diff $0.00; {r.randint(0, 2)}d apart"
    elif label == "amount_mismatch":
        d = amt * r.uniform(0.005, 0.2)
        gl = f"GL {party.title()} ${amt + d:,.2f} REF {ref}"
        delta = f"diff ${d:,.2f}; {r.randint(0, 2)}d apart; " + r.choice(["wire fee?", "short pay", "fx variance", "partial payment"])
    elif label == "timing_difference":
        gl = f"GL {party.title()} ${amt:,.2f} REF {ref}"
        delta = f"diff $0.00; {r.randint(6, 40)}d apart; " + r.choice(["crosses period end", "outstanding check", "deposit in transit"])
    elif label == "duplicate":
        gl = f"GL {party.title()} ${amt:,.2f} REF {ref}"
        delta = f"diff $0.00; {r.randint(0, 3)}d apart; " + r.choice(["second bank line for same ref", "posted twice", "ref already cleared"])
    else:
        gl = "GL no candidate entry"
        delta = r.choice(["no ledger match above threshold", "unrecorded bank fee", "no open item for payee"])
    if r.random() < 0.04:
        label = r.choice(["matched", "amount_mismatch", "timing_difference", "duplicate", "missing_in_ledger"])
    return f"{bank} | {gl} | {delta}", label


@dataclass
class Template:
    key: str
    name: str
    description: str
    labels: list[str]
    input_hint: str
    generator: Callable[[random.Random], tuple[str, str]]
    keywords: list[str] = field(default_factory=list)


TEMPLATES: dict[str, Template] = {
    "transaction_reconciliation": Template(
        "transaction_reconciliation", "Transaction Reconciler",
        "Compare a bank line with its closest ledger entry and decide how the pair reconciles.",
        ["matched", "amount_mismatch", "timing_difference", "duplicate", "missing_in_ledger"],
        "bank line | ledger entry | computed deltas", gen_reconciliation,
        ["reconcil", "recon ", "bank statement", "bank feed", "unmatched", "three-way", "close the books"],
    ),
    "expense_categorization": Template(
        "expense_categorization", "Expense Categorizer",
        "Classify raw card transactions into GL expense categories.",
        list(EXPENSE_MERCHANTS), "card descriptor | amount | memo", gen_expense,
        ["expense", "categor", "gl ", "gl code", "transaction", "merchant", "spend", "bookkeep", "ledger", "coding"],
    ),
    "fraud_detection": Template(
        "fraud_detection", "Fraud Screener",
        "Flag card transactions as fraud or legitimate from risk signals.",
        ["fraud", "legitimate"], "transaction + risk signals", gen_fraud,
        ["fraud", "suspicious", "anomal", "risk", "chargeback", "scam", "stolen"],
    ),
    "support_routing": Template(
        "support_routing", "Support Router",
        "Route customer support messages to the right financial-ops queue.",
        list(SUPPORT_INTENTS), "customer message", gen_support,
        ["support", "ticket", "route", "routing", "customer", "intent", "help desk", "inbox"],
    ),
    "invoice_triage": Template(
        "invoice_triage", "AP Invoice Triage",
        "Decide whether an incoming vendor invoice should be approved, reviewed, flagged duplicate, or rejected.",
        ["approve", "needs_review", "duplicate", "reject"], "invoice summary", gen_invoice,
        ["invoice", "accounts payable", " ap ", "bill", "vendor", "duplicate", "payable"],
    ),
}


def match_template(prompt: str) -> Template | None:
    p = f" {prompt.lower()} "
    best, best_score = None, 0
    for t in TEMPLATES.values():
        score = sum(p.count(k) for k in t.keywords)
        if score > best_score:
            best, best_score = t, score
    return best


def generate(template: Template, n: int, seed: int = 7) -> list[dict[str, str]]:
    r = random.Random(seed)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 5:
        attempts += 1
        text, label = template.generator(r)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "label": label, "source": "procedural"})
    return rows
