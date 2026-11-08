"""Label-preserving noise: the ways one real row shows up twice in a finance system.

Bank feeds truncate merchant names, card processors change casing and separators, people fat-finger memos,
exports reformat amounts, and processors prepend POS codes. None of that changes what the row *is*, so a
model that changes its answer under these edits is brittle. The same operations feed two places:

- evaluation: perturbed copies of held-out rows give a robustness slice, scored next to the clean one
- training: variants of the user's own rows, instead of exact copies (the Kimi K2 report found rephrased
  data beats repeating the same tokens for more epochs; the same holds for a 60-row upload)

Operations that could plausibly change a label (dropping a token that carries the signal) are training-only.
"""
from __future__ import annotations

import random
import re

_AMOUNT = re.compile(r"\$\s?(\d[\d,]*)(\.\d+)?")
_WORD = re.compile(r"[A-Za-z]{6,}")
_PREFIXES = ["POS DEBIT ", "PURCHASE ", "CARD 4411 ", "PENDING ", "DEBIT CARD PURCHASE ", "CHECKCARD "]
_SUFFIXES = [" 11/03", " RECURRING", " US", " 0312", " AUTH"]
_MSG_PREFIXES = ["Fwd: ", "Re: ", "Hi, ", "Quick one: ", "FYI ", "hey - "]
_MSG_SUFFIXES = [" Thanks!", " Sent from my iPhone", " (see attached)", " - thx", " ??"]


def _casing(t: str, rng: random.Random) -> str:
    return rng.choice([t.upper(), t.lower(), t.title()])


def _separators(t: str, rng: random.Random) -> str:
    sep = rng.choice([" - ", ", ", "  ", " / ", " ; "])
    out = re.sub(r"\s*\|\s*", sep, t)
    if out == t:
        out = re.sub(r"[#*]+", rng.choice(["", " "]), t)
    return out


def _typo(t: str, rng: random.Random) -> str:
    words = list(_WORD.finditer(t))
    if not words:
        return t
    m = rng.choice(words)
    w = m.group()
    i = rng.randrange(1, len(w) - 1)
    kind = rng.random()
    if kind < 0.4:      # swap neighbours
        w2 = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    elif kind < 0.7:    # drop a letter
        w2 = w[:i] + w[i + 1:]
    else:               # double a letter
        w2 = w[:i] + w[i] + w[i:]
    return t[:m.start()] + w2 + t[m.end():]


def _truncate(t: str, rng: random.Random) -> str:
    """Bank descriptors cut long names: 'SLACK TECHNOLOGIES' arrives as 'SLACK TECHNOLO'."""
    words = [m for m in _WORD.finditer(t) if len(m.group()) >= 8]
    if not words:
        return t
    m = rng.choice(words)
    keep = rng.randint(5, len(m.group()) - 2)
    return t[:m.start()] + m.group()[:keep] + t[m.end():]


def _amount(t: str, rng: random.Random) -> str:
    def fmt(m: re.Match) -> str:
        whole = m.group(1).replace(",", "")
        cents = (m.group(2) or ".00")[1:]
        style = rng.random()
        if style < 0.3:
            return f"{whole}.{cents} USD"
        if style < 0.55:
            return f"USD {int(whole):,}.{cents}"
        if style < 0.8:
            return f"${whole}.{cents}"
        return f"${int(whole):,}.{cents[:1] if cents.endswith('0') else cents}"
    return _AMOUNT.sub(fmt, t, count=1)


def _looks_like_descriptor(t: str) -> bool:
    letters = [c for c in t if c.isalpha()]
    return "$" in t or (bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.5)


def _affix(t: str, rng: random.Random) -> str:
    """Processor codes around a card descriptor; mail-client clutter around a message."""
    pre, suf = (_PREFIXES, _SUFFIXES) if _looks_like_descriptor(t) else (_MSG_PREFIXES, _MSG_SUFFIXES)
    return rng.choice(pre) + t if rng.random() < 0.6 else t + rng.choice(suf)


def _drop_memo(t: str, rng: random.Random) -> str:
    return re.sub(r"\s*\|\s*memo:[^|]*$", "", t)


def _drop_token(t: str, rng: random.Random) -> str:
    toks = t.split()
    if len(toks) < 4:
        return t
    toks.pop(rng.randrange(1, len(toks)))
    return " ".join(toks)


# Safe everywhere: the row still says what it said. Training-only ops may remove a signal-bearing token.
SAFE_OPS = [_casing, _separators, _typo, _truncate, _amount, _affix, _drop_memo]
TRAIN_OPS = SAFE_OPS + [_drop_token]


def perturb(text: str, rng: random.Random, strength: int = 1, train: bool = False) -> str:
    """`strength` edits drawn from the safe set (or the wider training set). Returns the input unchanged only
    when no operation applies, which the callers treat as "skip this row"."""
    ops = TRAIN_OPS if train else SAFE_OPS
    out = text
    for op in rng.sample(ops, min(strength, len(ops))):
        out = op(out, rng)
    out = re.sub(r"\s+", " ", out).strip()
    if out == text.strip():
        for op in rng.sample(ops, len(ops)):  # one more try with every op before giving up
            cand = re.sub(r"\s+", " ", op(text, rng)).strip()
            if cand and cand != text.strip():
                return cand
    return out


def perturb_rows(rows: list[dict], rng: random.Random, n: int, slice_name: str = "perturbed") -> list[dict]:
    """Up to `n` perturbed copies of distinct rows, labels carried over, each tagged with its slice and origin."""
    out: list[dict] = []
    seen = {r["text"] for r in rows}
    order = list(range(len(rows)))
    rng.shuffle(order)
    for i in order:
        if len(out) >= n:
            break
        r = rows[i]
        t = perturb(r["text"], rng, strength=rng.choice([1, 1, 2]))
        if t == r["text"] or t in seen:
            continue
        seen.add(t)
        out.append({"text": t, "label": r["label"], "source": slice_name, "slice": slice_name, "origin": r["text"]})
    return out
