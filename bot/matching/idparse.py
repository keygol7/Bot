"""Deterministic, convention-general market-id parsing — the LLM-free matcher core.

Both venues encode the settlement-deciding facts in their identifiers and titles using
PLATFORM-WIDE conventions (not per-category formats):

- Kalshi tickers: ``KX<SERIES>-<middle>-<outcome>`` — one date code (%y%b%d), one
  threshold vocabulary (T3.6 / B2450 / bare integers), outcome codes. Series semantics
  (what is measured) come from the venue's own /series metadata (title + category +
  tags), synced to the db — DATA-DRIVEN, so a new series parses without code changes.
- Polymarket slugs: ``<family>-<league>-<event tokens>-<ISO date>-<qualifiers>-<code>``
  — ISO dates, gtNptM-style thresholds, single-letter metric qualifiers (w/g/a/tg...),
  trailing outcome codes; titles carry full names.

The tokenizers here are POSITION-INDEPENDENT: they find dates, thresholds and period
markers wherever they appear and treat the residue as entity/outcome evidence. A brand
new category that follows the venues' standard conventions parses with zero new code —
that is the future-proof property. Anything that doesn't parse fails CLOSED and shows
up in the coverage report; it is never guessed.

Alignment between venues uses one mechanism for every category (teams, drivers,
candidates, companies): token containment plus deterministic code generation
(first3+last3 person codes: "Chase Elliott" -> chaell; last-name suffix: COM matched by
fracom; team-code-as-token: NY in ny-sea), guarded by the sub-org rule (BESTIA vs
BESTIA Academy are different teams).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from bot.matching.scope import id_scope_tags

# ---------------------------------------------------------------- vocabulary

# Generic metric vocabulary, derived from words in a series title / slug qualifiers.
# This is a bounded KEYWORD map (tested), not a per-series curation: any series whose
# title says "…Winner…" is a winner market, whatever the sport or category.
_METRIC_KEYWORDS = (
    ("fastest lap", "fastlap"), ("fastlap", "fastlap"),
    ("total games", "total"), ("total", "total"),
    ("winner", "winner"), ("race", "winner"), ("champion", "winner"),
    ("wins", "winner"), ("win", "winner"), ("game", "winner"), ("match", "winner"),
    ("podium", "podium"), ("top 10", "top10"), ("top 5", "top5"),
    ("goal", "goals"), ("assist", "assists"), ("point", "points"),
    ("save", "saves"), ("shot", "shots"), ("strikeout", "strikeouts"),
    ("home run", "homeruns"), ("hit", "hits"), ("rbi", "rbi"),
    ("inflation", "inflation_rate"), ("cpi", "inflation_rate"),
    ("gdp", "gdp_growth"), ("unemployment", "unemployment"),
    ("temperature", "temperature"), ("high temp", "temperature"),
    ("price", "price"), ("ipo", "ipo"), ("nominee", "nominee"),
    ("start", "starts"), ("mention", "unmatchable"), ("says", "unmatchable"),
)
# Poly single-token qualifiers seen platform-wide between the date and outcome code.
_POLY_QUALIFIER_METRIC = {
    "w": "winner", "g": "goals", "a": "assists", "m": "winner",
    "tg": "total", "fastlap": "fastlap", "cy": "winner", "dc": "winner",
    "cc": "winner", "sb": "stolen_bases", "hr": "homeruns", "ks": "strikeouts",
}
_GENERIC_TOKENS = frozenset({
    "the", "and", "for", "will", "who", "in", "at", "of", "vs", "v", "yes", "no",
    "main", "race", "event", "upcoming", "scheduled", "esports", "gaming", "team",
})
_SUB_ORG = frozenset({"academy", "jr", "junior", "youth", "reserve", "reserves",
                      "u17", "u18", "u19", "u20", "u21", "u23", "ii", "prospects"})
_NAME_STOP = frozenset({"de", "van", "der", "da", "la", "el", "jr", "the"})

_KALSHI_DATE = re.compile(r"(\d{2}[A-Z]{3}\d{2})")
_ISO_DATE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
_KALSHI_THR = re.compile(r"^[TB](\d+(?:\.\d+)?)$")
_KALSHI_INT_THR = re.compile(r"^(\d+(?:\.\d+)?)$")
_POLY_THR = re.compile(r"^(gte|gt|lte|lt)(\d+)(?:pt(\d+))?(?:(lt|lte)(\d+)(?:pt(\d+))?)?(?:pct|f|c)?$")
_POLY_PT = re.compile(r"^(\d+)pt(\d+)$")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class MarketKey:
    """Normalized, deterministic identity of one market's settlement proposition."""

    venue: str
    market_id: str
    category: str | None = None          # normalized grouping evidence (tags/league)
    event_date: date | None = None       # None = id carries no full date
    event_year: int | None = None        # season/year evidence when no full date
    event_tokens: frozenset = frozenset()  # identity of the event (teams, race, city)
    outcome_code: str | None = None      # the id's outcome segment, lowercased
    outcome_names: frozenset = frozenset()  # full-name tokens from the title
    metric: str = "unknown"
    thr_lo: float | None = None
    thr_hi: float | None = None
    scope: frozenset = frozenset()       # period/handicap markers (id_scope_tags)
    matchable: bool = True               # False = positively junk (mention/novelty)


def _tokens(text: str | None) -> frozenset:
    return frozenset(w for w in _WORD.findall((text or "").lower())
                     if w not in _GENERIC_TOKENS and len(w) > 1)


def metric_from_text(text: str | None) -> str:
    t = (text or "").lower()
    for kw, metric in _METRIC_KEYWORDS:
        if kw in t:
            return metric
    return "unknown"


# ---------------------------------------------------------------- kalshi

def parse_kalshi(ticker: str, title: str = "", series_meta: dict | None = None) -> MarketKey:
    """Tokenize a Kalshi ticker + title. ``series_meta`` (from the synced /series
    table: {title, category, tags}) supplies the data-driven metric/category; without
    it the market TITLE is the fallback metric evidence."""
    parts = (ticker or "").split("-")
    series = parts[0]
    middle = parts[1:-1]
    outcome = parts[-1] if len(parts) >= 2 else None

    meta_title = (series_meta or {}).get("title") or ""
    metric = metric_from_text(meta_title) if meta_title else "unknown"
    if metric == "unknown":
        metric = metric_from_text(title)
    category = None
    if series_meta:
        tags = series_meta.get("tags") or []
        category = (tags[0].lower() if tags else None) or (
            (series_meta.get("category") or "").lower() or None)

    event_date = None
    event_year = None
    ev_tokens: set = set()
    thr_lo = thr_hi = None
    for seg in middle:
        m = _KALSHI_DATE.search(seg)
        if m and event_date is None:
            try:
                event_date = datetime.strptime(m.group(1), "%y%b%d").date()
            except ValueError:
                pass
            residue = seg.replace(m.group(1), "")
            ev_tokens |= _tokens(residue)
            continue
        tm = _KALSHI_THR.match(seg)
        if tm:
            v = float(tm.group(1))
            if seg[0] == "T":
                thr_lo = v
            else:
                thr_hi = v
            continue
        mm = re.fullmatch(r"(\d{2})([A-Z]{3})", seg)
        if mm and event_date is None:
            # month-coded period (26JUN): year + month token evidence
            event_year = 2000 + int(mm.group(1))
            ev_tokens.add(mm.group(2).lower())
            continue
        ym = re.fullmatch(r"([A-Z]+?)(\d{2})", seg)
        if ym and event_date is None:
            # season-coded event (EER26, BRIGP26): identity token + year evidence
            ev_tokens |= _tokens(ym.group(1))
            event_year = 2000 + int(ym.group(2))
            continue
        ev_tokens |= _tokens(seg)

    # outcome may itself be a threshold (KXCPIYOY-26NOV-T3.6, KXATPGTOTAL-...-47)
    if outcome:
        tm = _KALSHI_THR.match(outcome)
        im = _KALSHI_INT_THR.match(outcome)
        if tm:
            v = float(tm.group(1))
            thr_lo, thr_hi = (v, thr_hi) if outcome[0] == "T" else (thr_lo, v)
            outcome = None
        elif im and metric not in ("winner", "unknown", "unmatchable"):
            # stat/scalar lines ride as the outcome segment: KMBAPP10-2 = 2+ goals,
            # KXATPGTOTAL-...-47 = 47+ total games, KXCPIYOY-...-T3.6 handled above
            thr_lo = float(im.group(1))
            outcome = None

    names = frozenset()
    if title:
        m = (re.search(r" - (.+?)$", title)
             or re.search(r"^Will (.+?) (?:win|be|set|score|record|finish|start)", title))
        if m:
            names = _tokens(m.group(1))
    scope = frozenset(id_scope_tags(ticker))
    return MarketKey(
        venue="kalshi", market_id=ticker, category=category,
        event_date=event_date, event_year=event_year,
        event_tokens=frozenset(ev_tokens) | (_tokens(title) & frozenset()),
        outcome_code=(outcome or "").lower() or None, outcome_names=names,
        metric=metric, thr_lo=thr_lo, thr_hi=thr_hi, scope=scope,
        matchable=metric != "unmatchable",
    )


# ---------------------------------------------------------------- polymarket

def parse_poly(slug: str, title: str = "") -> MarketKey:
    """Tokenize a Polymarket US slug + title (family-agnostic)."""
    toks = [t for t in (slug or "").lower().split("-") if t]
    event_date = None
    thr_lo = thr_hi = None
    metric = "unknown"
    ev_tokens: set = set()
    outcome_code = None

    m = _ISO_DATE.search(slug or "")
    if m:
        try:
            event_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    i = 0
    date_idx = None
    for idx in range(len(toks) - 2):
        if (toks[idx].isdigit() and len(toks[idx]) == 4
                and toks[idx + 1].isdigit() and toks[idx + 2].isdigit()):
            date_idx = idx
            break
    pre = toks[:date_idx] if date_idx is not None else toks
    post = toks[date_idx + 3:] if date_idx is not None else []

    # family + league prefixes are grouping evidence, then event identity tokens
    if pre:
        ev_tokens |= {t for t in pre[1:] if not t.isdigit()}  # drop the family prefix
        if len(pre) >= 2:
            pass
    for seg in list(post):
        tm = _POLY_THR.match(seg)
        if tm:
            lo = float(f"{tm.group(2)}.{tm.group(3) or 0}")
            if tm.group(1) in ("gte", "gt"):
                thr_lo = lo
            else:
                thr_hi = lo
            if tm.group(5):
                thr_hi = float(f"{tm.group(5)}.{tm.group(6) or 0}")
            post.remove(seg)
            continue
        pm = _POLY_PT.match(seg)
        if pm:
            thr_lo = float(f"{pm.group(1)}.{pm.group(2)}")
            post.remove(seg)
            continue
        if seg in _POLY_QUALIFIER_METRIC:
            metric = _POLY_QUALIFIER_METRIC[seg]
            post.remove(seg)
            continue
    if post:
        outcome_code = post[-1]
        ev_tokens |= set(post[:-1])
    if metric == "unknown":
        metric = metric_from_text(title)
    if metric == "unknown" and re.search(r" vs\.? ", (title or ""), re.IGNORECASE):
        metric = "winner"                     # h2h phrasing = winner market

    names = frozenset()
    event_names: frozenset = frozenset()
    if title:
        m2 = re.search(r" - (.+?)$", title)
        if m2 and m2.group(1).strip().lower() not in ("yes", "no"):
            names = _tokens(m2.group(1))
            event_names = _tokens(title[:m2.start()])
        else:
            event_names = _tokens(title)
    scope = frozenset(id_scope_tags(slug))
    return MarketKey(
        venue="polymarket_us", market_id=slug, category=(toks[1] if len(toks) > 1 else None),
        event_date=event_date, event_year=event_date.year if event_date else None,
        event_tokens=frozenset(ev_tokens) | event_names, outcome_code=outcome_code,
        outcome_names=names, metric=metric, thr_lo=thr_lo, thr_hi=thr_hi, scope=scope,
    )


# ---------------------------------------------------------------- alignment

def person_codes(name_tokens) -> set:
    """Deterministic code candidates for a person/entity name: poly-style first3+last3
    ('chase elliott' -> chaell), 6-char prefixes, and raw tokens."""
    toks = [t for t in name_tokens if t not in _NAME_STOP]
    out: set = set()
    for t in toks:
        out.add(t[:6])
        out.add(t)
    ordered = sorted(toks)  # order-insensitive fallback pairs
    if len(toks) >= 2:
        for a in toks:
            for b in toks:
                if a != b:
                    out.add(a[:3] + b[:3])
    return out


def outcome_align(a: MarketKey, b: MarketKey) -> bool:
    """Do the two YES outcomes name the same entity? Evidence order: full names
    (tokens either-side-contained, sub-org guarded) -> generated codes vs the other
    side's outcome code -> raw code containment (COM in fracom; suffix/prefix)."""
    na, nb = a.outcome_names, b.outcome_names
    if na and nb:
        if (na <= nb or nb <= na) and not ((na ^ nb) & _SUB_ORG):
            return True
    for names, other in ((na, b), (nb, a)):
        code = other.outcome_code
        if names and code:
            gens = person_codes(names)
            if code in gens:
                return True
            # prefixed venue codes: fwckylmba endswith kylmba (generated kyl+mba)
            if any(code.endswith(g) or g.endswith(code) for g in gens if len(g) >= 4):
                return True
            # last-name suffix / containment: COM <- fracom, KRU <- ashkru
            if any(code.endswith(t[:len(code)]) or t.startswith(code) or code in t
                   for t in names if len(code) >= 2):
                return True
    ca, cb = a.outcome_code, b.outcome_code
    if ca and cb and len(ca) >= 2 and len(cb) >= 2:
        if ca == cb or ca in cb or cb in ca or cb.endswith(ca) or ca.endswith(cb):
            return True
    return False


def code_aligns_tokens(code: str, tokens) -> bool:
    """Can ``code`` be decomposed into 1-6 char pieces, each a prefix OR suffix of a
    distinct token? Handles both venue conventions deterministically:
    BRIGP -> bri(tish)+g(rand)+p(rix); COMDON -> (fra)com+(mat)don; EER -> eer(o)."""
    toks = [t for t in tokens if len(t) >= 2]
    if not code or not toks:
        return False

    def rec(rest: str, used: frozenset) -> bool:
        if not rest:
            return True
        for n in range(min(6, len(rest)), 0, -1):
            piece = rest[:n]
            for i, t in enumerate(toks):
                if i in used:
                    continue
                if t.startswith(piece) or t.endswith(piece):
                    if rec(rest[n:], used | {i}):
                        return True
        return False

    return len(code) >= 2 and rec(code, frozenset())


def events_align(a: MarketKey, b: MarketKey) -> bool:
    """Same underlying event: dates equal when both known (±1 day for timezone skew);
    with a one-sided date, the season year must not conflict and the event tokens must
    share identity. Event tokens align by containment either way."""
    if a.event_date and b.event_date:
        if abs((a.event_date - b.event_date).days) > 1:
            return False
    else:
        ya = a.event_year or (a.event_date.year if a.event_date else None)
        yb = b.event_year or (b.event_date.year if b.event_date else None)
        if ya and yb and ya != yb:
            return False
        if not (a.event_tokens and b.event_tokens):
            return False                     # no date AND no shared identity -> never
    ea, eb = a.event_tokens, b.event_tokens
    if not ea or not eb:
        # dates matched exactly on both sides; allow when at least outcome aligns
        return bool(a.event_date and b.event_date)
    small, big = (ea, eb) if len(ea) <= len(eb) else (eb, ea)
    hits = sum(1 for t in small if any(t in o or o in t for o in big if len(t) >= 2))
    if hits >= 1:
        return True
    # abbreviation codes: one side's compact event code segments over the other's tokens
    return any(code_aligns_tokens(t, big) for t in small if 3 <= len(t) <= 12)


def keys_match(a: MarketKey, b: MarketKey) -> bool:
    """Deterministic same-proposition check: exact metric + threshold + scope, same
    event, same YES entity. Fails closed on unknown metric or unmatchable types."""
    if not a.matchable or not b.matchable:
        return False
    if a.metric == "unknown" or b.metric == "unknown" or a.metric != b.metric:
        return False
    if a.scope != b.scope:
        return False
    if (a.thr_lo, a.thr_hi) != (b.thr_lo, b.thr_hi):
        return False
    if not events_align(a, b):
        return False
    if a.outcome_code is None and b.outcome_code is None and a.metric != "winner":
        # Event-level scalar (CPI above X, temperature, totals): neither id names an
        # outcome entity — the metric+threshold+event IS the proposition. (Player-stat
        # thresholds keep an outcome code on at least one side and fall through.)
        return True
    return outcome_align(a, b)
