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
    ("exact", "exact_score"), ("set winner", "set_winner"),
    ("correct score", "exact_score"), ("first to score", "first_score"),
    ("to score first", "first_score"), ("score first", "first_score"),
    ("method of victory", "mov"), ("margin of victory", "mov"),
    ("score or assist", "soa"), ("win margin", "mov"),
    # qualifier/award propositions are NOT the same as winning the thing itself —
    # without these, "#1 Seed" ties "Division Winner" at the mutual-best stage
    ("seed", "seed"), ("mvp", "mvp"), ("most valuable", "mvp"),
    ("county", "county"), ("popular vote", "popvote"), ("spread", "spread"),
    ("comeback", "cpoty"), ("reliever of", "reloty"), ("manager of", "moty"),
    ("executive of", "eoty"), ("hank aaron", "haaron"),
    ("silver slugger", "silverslugger"), ("all-star", "allstar"),
    ("all star", "allstar"), ("extra inning", "extras"),
    ("draft", "draft"), ("relegat", "relegation"), ("promot", "promotion"),
    ("rookie of", "rookie"), ("coach of", "coach"), ("cy young", "cyyoung"),
    ("ballon", "ballondor"), ("playoff", "playoffs"), ("make the playoffs", "playoffs"),
    ("nominee", "nominee"), ("nomination", "nominee"),
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
    ("price", "price"), ("ipo", "ipo"),
    ("start", "starts"), ("mention", "unmatchable"), ("says", "unmatchable"),
)
# Slug FAMILY prefixes carry the market type platform-wide (tsc = total score,
# asc = against-the-spread...). More reliable than titles; checked first.
_POLY_FAMILY_METRIC = {
    "tsc": "total", "asc": "spread", "aqc": "playoffs",
    "adpc": "draft", "arankc": "draft",
}
_LEADER_STATS = ("bavg", "avg", "era", "ops", "war", "doubles", "triples", "steals",
                 "saves", "runs", "wins", "hits", "hrs", "hr", "rbi", "rbis",
                 "strikeouts", "ks")
_LEADER_CANON = {"bavg": "avg", "hrs": "hr", "rbis": "rbi", "ks": "strikeouts"}


def leader_metric(text: str) -> str:
    """Season stat-LEADER markets are their own metric family (ldr_rbi != rbi), so a
    'Judge RBI leader' future can never cross-match a 'Judge 2+ RBIs tonight' prop."""
    t = (text or "").lower()
    if "leader" not in t:
        return ""
    for stat in _LEADER_STATS:
        if re.search(rf"\b{stat}\b", t):
            return "ldr_" + _LEADER_CANON.get(stat, stat)
    return "ldr_unknown"


# Poly single-token qualifiers seen platform-wide between the date and outcome code.
_POLY_QUALIFIER_METRIC = {
    "w": "winner", "g": "goals", "a": "assists", "ga": "ga", "m": "winner",
    "tg": "total", "fastlap": "fastlap", "cy": "cyyoung", "roy": "rookie",
    "mvp": "mvp", "dc": "winner",
    "cc": "winner", "sb": "stolen_bases", "hr": "homeruns", "ks": "strikeouts",
    # UFC prop families: method-of-finish vs round-of-victory are DIFFERENT
    # metrics (a "round: other" market married a "method: draw" market as
    # winner/winner)
    "mof": "mof", "mov": "mof", "rov": "round",
}
_GENERIC_TOKENS = frozenset({
    "the", "and", "for", "will", "who", "in", "at", "of", "vs", "v", "yes", "no",
    "go", "to", "by", "end", "contest", "visit",
    "main", "race", "event", "upcoming", "scheduled", "esports", "gaming", "team",
    # structural/metric words: never event IDENTITY (Dallas-high vs Midwest-high
    # must not align on "high")
    "high", "low", "temp", "temperature", "winner", "game", "match", "final",
    "championship", "champion", "champ", "tournament", "series", "cup", "league",
    "season", "week", "pro", "division",
    # sport names are CATEGORY evidence, not event identity — "Pro Football
    # Championship" must not event-align with "AFC South" via 'football'
    "football", "basketball", "baseball", "hockey", "soccer", "tennis", "golf",
    "boxing", "cricket", "volleyball",
})
# Category compatibility (NEGATIVE guard only): when BOTH sides resolve to a known
# group and the groups differ -> reject; anything unknown passes (future-proof).
# kalshi evidence = series tags; poly evidence = slug league segment.
_CATEGORY_GROUPS = {
    "esports": {"esports", "cs2", "dota2", "lol", "valorant", "r6", "cod", "sc2",
                "rocketleague", "overwatch"},
    # gendered tennis circuits are DIFFERENT events even when names collide
    # (BONWEI: men's ITF matched women's ITF via a 3-letter 'wei' collision)
    "tennis_men": {"atp", "itfme"},
    "tennis_women": {"wta", "itfwo", "itfw"},
    "soccer": {"soccer", "football-soccer", "fwc", "ucl", "epl", "laliga", "mls",
               "seriea", "bundesliga", "ligue1", "uel", "uecl", "bdor",
               "brasileiro", "uefa", "fifa"},
    "rugby": {"rugby"},
    "entertainment": {"entertainment", "tv", "movies", "music", "videogames",
                      "gaming2", "celebrity", "awards-entertainment"},
    "basketball": {"basketball", "nba", "wnba", "ncaab"},
    "football": {"football", "nfl", "ncaaf", "cfb"},
    "baseball": {"baseball", "mlb", "kbo", "npb"},
    "hockey": {"hockey", "nhl"},
    "motorsport": {"motorsport", "f1", "nascar", "indycar", "motogp"},
    "golf": {"golf", "pga", "liv"},
    "mma": {"mma", "ufc", "boxing"},
    "cricket": {"cricket", "t20", "ipl"},
    "chess": {"chess"},
    "weather": {"weather", "temp", "climate"},
    "politics": {"politics", "elections", "usgub", "usse", "usprez", "ushouse"},
    "economics": {"economics", "inflation", "fed", "gdp", "uscpi"},
    "crypto": {"crypto", "bitcoin", "ethereum", "btc", "eth"},
    "entertainment": {"entertainment", "movies", "music", "oscars", "awards"},
}
_TOKEN_GROUP = {t: g for g, toks in _CATEGORY_GROUPS.items() for t in toks}


def category_conflict(a: "MarketKey", b: "MarketKey") -> bool:
    ga = _TOKEN_GROUP.get((a.category or "").lower())
    gb = _TOKEN_GROUP.get((b.category or "").lower())
    return bool(ga and gb and ga != gb)
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
    two_party: bool = False              # title has an A-vs-B structure (game/match)


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
    # metric evidence: series title AND series ticker together (the ticker often
    # carries the qualifier the title omits: KXNFL1SEED's title just says "win the
    # conference" — 'seed' is in the ticker; keyword order handles specificity).
    # Stat-LEADER series are their own metric family (ldr_rbi), never plain stats.
    _K_SERIES_METRIC = {"KXUFCMOF": "mof", "KXUFCVICROUND": "round",
                        "KXPGACOMPETE": "compete"}
    forced = _K_SERIES_METRIC.get(series)
    metric = forced or leader_metric(f"{meta_title} {series.lower()}")
    if not metric:
        metric = metric_from_text(f"{meta_title} {series.lower()}") if meta_title else "unknown"
    if metric == "unknown":
        metric = metric_from_text(title)
    category = None
    if series_meta:
        tags = series_meta.get("tags") or []
        _GENERIC_TAGS = {"awards", "sports", "politics", "world"}
        tag0 = tags[0].lower() if tags else None
        if tag0 in _GENERIC_TAGS:
            # "Awards" tags both the Game Awards and the Ballon d'Or — the series
            # CATEGORY (Entertainment vs Sports) is the real discriminator
            tag0 = (series_meta.get("category") or "").lower() or tag0
        category = tag0 or (
            (series_meta.get("category") or "").lower() or None)
    # gendered tennis series override the generic "tennis" tag so the category
    # conflict guard can separate the circuits deterministically
    if series.startswith(("KXWTA", "KXITFW")):
        category = "wta"
    elif series.startswith(("KXATP", "KXITFMATCH")):
        category = "atp"

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
            # start time rides right after the date (26JUL02 0700 OSGSOS): strip it
            # or the team pair-code tokenizes as '0700osgsos' and its grams are junk
            residue = re.sub(r"^\d{4}", "", residue)
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
        if re.fullmatch(r"\d{2}", seg) and event_year is None:
            event_year = 2000 + int(seg)
            continue
        ev_tokens |= _tokens(seg)
    if not ev_tokens and meta_title:
        # sparse middle (futures/elections/IPO): the series title IS the event evidence
        ev_tokens |= _tokens(meta_title)

    if metric == "spread" and outcome:
        # spread outcomes fuse team+line: ATL2 = "Atlanta wins by over 1.5" = margin
        # >= 2. Normalize to integer-margin form (poly neg-1pt5 normalizes the same).
        sp = re.fullmatch(r"([A-Z]+?)(\d+)", outcome)
        if sp:
            outcome = sp.group(1)
            thr_lo = float(sp.group(2))
    if metric == "draft":
        # KXMLBDRAFTTOP-26-10-AGRA / KXMLBDRAFTPICK-26-1-AGRA: the bare int in the
        # middle is the pick/top-N qualifier -> scope, so top3/top5/top10 books and
        # exact-pick books never cross-match
        kind = "top" if "TOP" in series else "pick"
        # the qualifier FOLLOWS the year code (KXMLBDRAFTTOP-26-10-AGRA): scan from
        # the end, and never eat the segment already consumed as the year
        for seg in reversed(middle):
            if (seg.isdigit() and len(seg) <= 2 and int(seg) <= 40
                    and (event_year is None or 2000 + int(seg) != event_year)):
                ev_tokens.discard(seg)
                scope_extra_k = f"{kind}{int(seg)}"
                break
        else:
            scope_extra_k = None
    else:
        scope_extra_k = None
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
    scope = set(id_scope_tags(ticker))
    # Series-level scope: kalshi encodes sub-game scopes in the SERIES name itself
    # (KXWNBA1QWINNER, KXNBA2HWINNER, KXVALORANTMAP) — without these, quarter/map/full
    # markets collide onto one poly market and the fan-out backstop kills them all.
    qm = re.search(r"(\d)Q", series)
    if qm:
        scope.add(f"q{qm.group(1)}")
    fm = re.search(r"F(\d)(?![A-Z0-9]*GP)", series)   # KXMLBF5* (not F1GP etc.)
    if fm and "F1" not in series:
        scope.add(f"f{fm.group(1)}")
    if scope_extra_k:
        scope.add(scope_extra_k)
    hm = re.search(r"(\d)H|([FS])H(?![A-Z])", series)
    if hm and "MATCH" not in series and "GAME" not in series:
        scope.add(f"h{hm.group(1) or hm.group(2).lower()}")
    if "MAP" in series:
        # map number rides the middle/outcome: KXVALORANTMAP-...GAMETS-2-TS
        nums = [seg for seg in parts[1:-1] if re.fullmatch(r"\d", seg)]
        scope.add(f"map{nums[-1] if nums else '?'}")
    scope = frozenset(scope)
    two_party = bool(re.search(r"\bvs\.?\b|\bagainst\b", title or "", re.I))
    oc = (outcome or "").lower() or None
    if oc and _DATE_SHAPED.match(oc):
        # a date-shaped "outcome" is a mis-split residue (unknown-series ticker
        # like KXSUPERBOWLWHITEHOUSE-26DEC31): treat it as the DATE, never as an
        # outcome code — as an outcome it substring-matched poly's 'dec' and
        # married the Super Bowl White House market to a McGregor fight prop.
        if event_date is None:
            try:
                from datetime import datetime as _dt
                event_date = _dt.strptime(oc[:7], "%y%b%d").date()
                event_year = event_date.year
            except ValueError:
                pass
        oc = None
    return MarketKey(
        venue="kalshi", market_id=ticker, category=category,
        event_date=event_date, event_year=event_year,
        event_tokens=frozenset(ev_tokens) | (_tokens(title) & frozenset()),
        outcome_code=oc, outcome_names=names, two_party=two_party,
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

    event_year = None
    m = _ISO_DATE.search(slug or "")
    if m:
        try:
            event_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    else:
        ym = re.search(r"\b(20\d{2})", (slug or "").replace("-", " "))
        if ym:
            event_year = int(ym.group(1))

    scope_extra: set = set()
    family = toks[0] if toks else ""
    fam_metric = _POLY_FAMILY_METRIC.get(family, "")
    spread_sign = None
    date_idx = None
    for idx in range(len(toks) - 2):
        if (toks[idx].isdigit() and len(toks[idx]) == 4
                and toks[idx + 1].isdigit() and toks[idx + 2].isdigit()):
            date_idx = idx
            break
    if date_idx is not None:
        pre, post = toks[:date_idx], toks[date_idx + 3:]
    else:
        # undated slug (ipcc-2026ipos-databricks): the trailing-outcome convention
        # still holds — last token is the outcome, the rest is event identity
        pre, post = toks[:-1], toks[-1:]

    # family + league prefixes are grouping evidence, then event identity tokens
    if pre:
        # PRE-date tokens are event identity ONLY — never scope. "2game" here is the
        # team 2Game Esports (FURA2GAME taught us); real sub-game markers (map2/
        # game2) ride POST-date, where the loop below extracts them.
        ev_tokens |= {t for t in pre[1:] if not t.isdigit()}
        if len(pre) >= 2:
            pass
    for seg in list(post):
        tm = _POLY_THR.match(seg)
        if tm:
            lo = float(f"{tm.group(2)}.{tm.group(3) or 0}")
            if tm.group(1) in ("gte", "gt"):
                # count-stat convention: "over 46.5" == "47 or more" (kalshi's integer
                # form). Normalize x.5 gt-lines up to the minimum qualifying integer.
                thr_lo = lo + 0.5 if (tm.group(1) == "gt" and lo % 1 == 0.5) else lo
            else:
                thr_hi = lo
            if tm.group(5):
                thr_hi = float(f"{tm.group(5)}.{tm.group(6) or 0}")
            post.remove(seg)
            continue
        if seg in ("neg", "pos") and fam_metric == "spread":
            spread_sign = seg
            post.remove(seg)
            continue
        if seg in ("top3", "top5", "top10", "top20"):
            scope_extra.add(seg)
            post.remove(seg)
            continue
        om = re.fullmatch(r"(\d)(?:st|nd|rd|th)", seg)
        if om:
            scope_extra.add(f"pick{om.group(1)}")
            post.remove(seg)
            continue
        pm = _POLY_PT.match(seg)
        if pm:
            v = float(f"{pm.group(1)}.{pm.group(2)}")
            if fam_metric == "spread":
                # neg-1pt5 = covers -1.5 = wins by margin >= 2 (integer-margin form,
                # matching kalshi's ATL2 normalization)
                thr_lo = v + 0.5
            else:
                # bare x.5 totals ("O/U 10.5 - Over") = ">= 11" count form, matching
                # kalshi's outcome-digit convention (-11 = over 10.5)
                thr_lo = v + 0.5 if v % 1 == 0.5 else v
            post.remove(seg)
            continue
        if seg in _POLY_QUALIFIER_METRIC:
            metric = _POLY_QUALIFIER_METRIC[seg]
            post.remove(seg)
            continue
        fm = re.match(r"^f(\d)$", seg)
        if fm:
            # partial-game scope: f5 = first 5 innings (the F5-vs-full-game false
            # match traded live on 2026-07-07; cost $3.04 to flatten)
            scope_extra.add(f"f{fm.group(1)}")
            post.remove(seg)
            continue
        sm = re.match(r"^(?:(map|game|set)(\d)|(\d)(map|game|set)s?)$", seg)
        if sm:
            # sub-game scope in the slug (map2 / game2 / 2game): normalize map==game
            # cross-venue (dota "game 2" is kalshi's map 2) so scopes compare equal
            kind = (sm.group(1) or sm.group(4)).replace("game", "map")
            scope_extra.add(f"{kind}{sm.group(2) or sm.group(3)}")
            post.remove(seg)
            continue
    if post:
        outcome_code = post[-1]
        ev_tokens |= set(post[:-1])
    lm = leader_metric(" ".join(toks) + " " + (title or ""))
    if lm:
        metric = lm
    elif fam_metric:
        metric = fam_metric                   # family prefix beats title heuristics
    if metric == "unknown":
        metric = metric_from_text(title)
    if metric == "unknown" and re.search(r" vs\.? ", (title or ""), re.IGNORECASE):
        metric = "winner"                     # h2h phrasing = winner market
    if fam_metric == "spread":
        if spread_sign == "neg" and date_idx is not None and date_idx >= 3:
            outcome_code = toks[2]            # the minus line belongs to the FIRST team
        else:
            # pos lines are polarity-INVERTED vs any kalshi book (A +1.5 YES == B
            # wins-by-2+ NO); the pair model can't express inversion -> fail closed
            return MarketKey(venue="polymarket_us", market_id=slug, matchable=False,
                             metric="spread")

    names = frozenset()
    event_names: frozenset = frozenset()
    if title:
        m2 = re.search(r" - (.+?)$", title)
        if m2 and m2.group(1).strip().lower() not in ("yes", "no"):
            names = _tokens(m2.group(1))
            event_names = _tokens(title[:m2.start()])
        else:
            event_names = _tokens(title)
    scope = frozenset(id_scope_tags(slug)) | frozenset(scope_extra)
    return MarketKey(
        venue="polymarket_us", market_id=slug, category=(toks[1] if len(toks) > 1 else None),
        event_date=event_date,
        event_year=event_date.year if event_date else event_year,
        event_tokens=frozenset(ev_tokens) | event_names, outcome_code=outcome_code,
        two_party=bool(re.search(r"\bvs\.?\b|\bagainst\b", title or "", re.I)),
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
    toks = [t for t in tokens if len(t) >= 2][:10]     # bound the search space
    if not code or not toks or not (2 <= len(code) <= 10):
        return False

    def rec(rest: str, used: frozenset, depth: int, ones: int, multi: int, big: bool) -> bool:
        if not rest:
            # substance requirement: one >=3 piece (nak) OR two multi-char pieces
            # (pe+eg over petrocub/egnatia) — never a pile of initials
            return big or multi >= 2
        if depth >= 4:                                  # a code is <=4 abbreviation pieces
            return False
        for n in range(min(6, len(rest)), 0, -1):
            piece = rest[:n]
            if n == 1 and ones >= 2:
                continue                                # at most two initials (g+p in BRIGP)
            for i, t in enumerate(toks):
                if i in used:
                    continue
                if n == 1:
                    ok = t.startswith(piece)            # 1-char = INITIAL only
                else:
                    ok = t.startswith(piece) or t.endswith(piece)
                if ok and rec(rest[n:], used | {i}, depth + 1,
                              ones + (n == 1), multi + (n >= 2), big or n >= 3):
                    return True
        return False

    return rec(code, frozenset(), 0, 0, 0, False)


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
            # +/-1 exists ONLY for season spillover (NFL 26 season ends Jan/Feb 27);
            # otherwise strict — the tolerance married the 2027 Nobel to the 2026
            # prize and Club Brugge's UCL-27 to the 2026 Ballon d'Or.
            dated_month = (a.event_date or b.event_date)
            spill = dated_month is not None and dated_month.month <= 2
            if abs(ya - yb) > 1 or not spill:
                return False
        # FUTURES-vs-GAME guard: a year-only side (undated futures like KXUCL-27)
        # must never marry a fully-DATED two-team game (atc-ucl-inte-lin-2026-07-14)
        # just because the outcome code aligns — "Inter wins the 2027 UCL" is not
        # "Inter wins this game" (traded live 2026-07-08, -$5.40).
        # FUTURES-vs-GAME guard: an undated season/championship market must never
        # marry a DATED two-party game just because its outcome code names one of
        # the teams — "Inter wins the 2027 UCL" is not "Inter wins this game"
        # (traded live 2026-07-08, -$5.40). Single-subject dated events (drafts,
        # awards) have no vs-structure and still pair with undated futures.
        dated, undated = (a, b) if a.event_date else (b, a)
        if dated.event_date and not undated.event_date and dated.two_party \
                and not undated.two_party:
            return False
        if not (a.event_tokens and b.event_tokens):
            # no date and a side with no event identity (KXIPO-26-DATABRICKS has only
            # its outcome): defer to the outcome gate — with EXACT year agreement
            # (the +/-1 tolerance married the 2027 Nobel to the 2026 prize and Club
            # Brugge's UCL-27 to the 2026 Ballon d'Or)
            return bool(ya and yb and ya == yb)
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


def match_score(a: MarketKey, b: MarketKey) -> int:
    """Evidence strength for a gated pair (0 = fails a hard gate). Used by the join
    to pick the MUTUAL BEST counterpart instead of accepting every loose alignment —
    the true pair (exact outcome code + strong event evidence) outscores a same-day
    lookalike (shared 3-gram, weak containment), so over-production stops killing
    true pairs via the fan-out backstop."""
    if not keys_match(a, b):
        return 0
    score = 0
    # date evidence
    if a.event_date and b.event_date:
        score += 3 if a.event_date == b.event_date else 1
    # event-token evidence: count real hits
    ea, eb = a.event_tokens, b.event_tokens
    small, big = (ea, eb) if len(ea) <= len(eb) else (eb, ea)
    hits = sum(1 for t in small if any(t == o for o in big))
    part = sum(1 for t in small if len(t) >= 3 and any(t in o or o in t for o in big))
    score += min(3, 2 * hits + part)
    # outcome evidence
    ca, cb = a.outcome_code, b.outcome_code
    if ca and cb and ca == cb:
        score += 3
    else:
        gens_a = person_codes(a.outcome_names) if a.outcome_names else set()
        gens_b = person_codes(b.outcome_names) if b.outcome_names else set()
        if (cb and cb in gens_a) or (ca and ca in gens_b):
            score += 3
        elif (a.outcome_names and b.outcome_names
              and (a.outcome_names <= b.outcome_names or b.outcome_names <= a.outcome_names)):
            score += 3
        elif (cb and any(cb.endswith(g) or g.endswith(cb) for g in gens_a if len(g) >= 4))                 or (ca and any(ca.endswith(g) or g.endswith(ca) for g in gens_b if len(g) >= 4)):
            score += 2
        else:
            score += 1
    return score


_CITY_ALIASES = {"mdw": "chi", "chicago": "chi", "nyc": "ny", "nychigh": "ny",
                 "lax": "la", "phl": "phil", "philadelphia": "phil", "denver": "den",
                 "sfo": "sf", "austin": "aus", "miami": "mia"}


def _scalar_subject(k) -> set:
    """Subject evidence for scalar markets, derived from the id itself: the kalshi
    series residue (KXHIGHPHIL -> highphil, phil) and poly slug segments
    (tc-temp-mdwhigh -> mdwhigh, mdw/chi). Cheap, id-level, venue-agnostic."""
    out = set()
    mid = (k.market_id or "").lower()
    if k.venue == "kalshi":
        seg = mid.split("-")[0].replace("kx", "")
        for pref in ("high", "low"):
            if seg.startswith(pref) and len(seg) > len(pref):
                city = seg[len(pref):]
                out |= {seg, _CITY_ALIASES.get(city, city)}
        out.add(seg)
    else:
        segs = mid.split("-")
        if segs and segs[0] in ("tc", "cpic", "ec", "fc"):
            out |= set(segs[1:3])
        for segp in segs:
            if segp.endswith(("high", "low")) and len(segp) > 3:
                city = segp[:-4] if segp.endswith("high") else segp[:-3]
                out |= {segp, _CITY_ALIASES.get(city, city)}
    for t in list(out):
        for stem in ("cpi", "inf", "gdp", "temp"):
            if stem in t:
                out.add(stem)
    return out


_DATE_SHAPED = re.compile(r"^\d{2}[a-z]{3}\d{2}$|^\d{2}[a-z]{3}$", re.I)


def keys_match(a: MarketKey, b: MarketKey) -> bool:
    """Deterministic same-proposition check: exact metric + threshold + scope, same
    event, same YES entity. Fails closed on unknown metric or unmatchable types."""
    if not a.matchable or not b.matchable:
        return False
    if a.metric == "unknown" or b.metric == "unknown" or a.metric != b.metric:
        return False
    if category_conflict(a, b):
        return False
    if a.scope != b.scope:
        return False
    if (a.thr_lo, a.thr_hi) != (b.thr_lo, b.thr_hi):
        return False
    if not events_align(a, b):
        return False
    def _thr_shaped(oc):
        return oc is None or re.fullmatch(r"t?\d+(?:\.\d+)?[a-z]?", oc or "")
    if _thr_shaped(a.outcome_code) and _thr_shaped(b.outcome_code) \
            and a.metric != "winner":
        # Event-level scalar (CPI above X, temperature, totals): neither id names an
        # outcome entity — the metric+threshold+event IS the proposition. But the
        # SUBJECT must corroborate: same-day scalars with no shared identity married
        # Philadelphia's high temp to Chicago's and Brazil's inflation to US CPI.
        sa, sb = _scalar_subject(a), _scalar_subject(b)
        if sa and sb:
            # id-level subjects on both sides are authoritative — generic title
            # words ("highest", "temp") must not vouch for Philadelphia == Chicago
            if not any(x == y or (len(x) >= 3 and (x in y or y in x))
                       for x in sa for y in sb):
                return False
            return True
        ta = {_CITY_ALIASES.get(t, t) for t in a.event_tokens if t not in _GENERIC_TOKENS} | sa
        tb = {_CITY_ALIASES.get(t, t) for t in b.event_tokens if t not in _GENERIC_TOKENS} | sb
        if ta and tb and not any(
                x == y or (len(x) >= 3 and (x in y or y in x)) for x in ta for y in tb):
            return False
        return True
    return outcome_align(a, b)


# ---------------------------------------------------------------- join

def join_pairs(kalshi_keys, poly_keys, *, undated_window_days: int = 366):
    """Deterministic cross-venue join, fully BLOCKED so it scales to whole boards:
    poly keys are indexed by (metric, thr, date, token-3-gram) — both the first-3 and
    last-3 of every event token (kalshi pair-codes like NYSEA need the suffix gram to
    meet poly's ny/sea tokens; COMDON's halves meet fracom/matdon via suffixes). A
    kalshi key only ever meets poly keys sharing real token evidence; keys_match then
    does the exact verification. Pure function; caller applies fan-out/blacklist."""
    from datetime import date as _date, timedelta

    def grams(key):
        toks = list(key.event_tokens)[:12] or list(key.outcome_names)[:6]
        out = set()
        for t in toks:
            if len(t) >= 3:
                out.add(t[:3]); out.add(t[-3:])
            elif len(t) == 2:
                out.add(t)
        return out

    ix_dated: dict = {}
    ix_undated: dict = {}
    for pk in poly_keys:
        if not pk.matchable or pk.metric == "unknown":
            continue
        mt = (pk.metric, pk.thr_lo, pk.thr_hi)
        if pk.event_date is None:
            # undated poly (IPO-class: ipcc-2026ipos-databricks) joins only via the
            # undated index — year agreement + outcome strength gate in keys_match
            for g in grams(pk):
                ix_undated.setdefault((*mt, g), []).append(pk)
            continue
        for g in grams(pk):
            ix_dated.setdefault((*mt, pk.event_date, g), []).append(pk)
            ix_undated.setdefault((*mt, g), []).append(pk)

    today = _date.today()
    out = []
    seen = set()
    scored: list = []
    for kk in kalshi_keys:
        if not kk.matchable or kk.metric == "unknown":
            continue
        mt = (kk.metric, kk.thr_lo, kk.thr_hi)
        cands: dict = {}
        if kk.event_date is not None:
            for pd in (kk.event_date + timedelta(days=d) for d in (-1, 0, 1)):
                for g in grams(kk):
                    for pk in ix_dated.get((*mt, pd, g), ()):
                        cands[id(pk)] = pk
        else:
            for g in grams(kk):
                for pk in ix_undated.get((*mt, g), ()):
                    py = pk.event_date.year if pk.event_date else pk.event_year
                    if kk.event_year and py and abs(py - kk.event_year) > 1:
                        continue
                    if pk.event_date and abs((pk.event_date - today).days) > undated_window_days:
                        continue
                    cands[id(pk)] = pk
        for pk in cands.values():
            sc = match_score(kk, pk)
            if sc > 0:
                scored.append((sc, kk, pk))

    # MUTUAL BEST: each market keeps only its strongest counterpart, and only when
    # the choice is unambiguous (a strict margin over the runner-up on both sides).
    best_k: dict = {}
    best_p: dict = {}
    for sc, kk, pk in scored:
        for side, key in ((best_k, kk.market_id), (best_p, pk.market_id)):
            cur = side.get(key)
            if cur is None or sc > cur[0]:
                side[key] = (sc, kk, pk, cur[0] if cur else 0)
            elif sc > cur[3]:
                side[key] = (cur[0], cur[1], cur[2], sc)
    for sc, kk, pk, second in best_k.values():
        if sc <= second:
            continue                                 # ambiguous on the kalshi side
        bp = best_p.get(pk.market_id)
        if bp and bp[1] is kk and bp[0] > bp[3]:
            pair_id = (kk.market_id, pk.market_id)
            if pair_id not in seen:
                seen.add(pair_id)
                out.append((kk, pk))
    return out


# ---------------------------------------------------------------- name alignment

def participants(title: str) -> list:
    """Participant name-token-sets from an 'A vs B' title (empty when not a
    matchup). Case-insensitive — esports team names are often lowercase."""
    t = title or ""
    m = re.search(r"\bthe\s+(.+?)(?::|\?|$)", t)
    seg = m.group(1) if m else t
    halves = re.split(r"\s+vs\.?\s+", seg, flags=re.I)
    if len(halves) != 2:
        return []
    a, b = halves
    b = re.sub(r"\s+(match|game|series|fight|map \d+)\??\s*$", "", b, flags=re.I)
    return [_tokens(a), _tokens(b)]


def names_fully_align(ka_title: str, poly_title: str, poly_slug: str) -> bool:
    """True when EVERY kalshi participant (or the single subject) aligns at the
    NAME level with the poly evidence. This is the deterministic discriminator
    between a rules-LLM 'different players' verdict that is name-form pedantry
    ('Zampardo' vs 'Maddy Zampardo' — both participants align -> override) and a
    genuine id collision (BONWEI: only 'wei' of two participants aligns -> the
    drop stands). Sub-org tokens (academy/junior/...) block alignment — BESTIA
    Academy never aligns with BESTIA."""
    poly_ev = _tokens(poly_title) | _tokens(poly_slug.replace("-", " "))
    poly_codes = set(re.findall(r"[a-z0-9]+", (poly_slug or "").lower()))

    def side_aligns(toks: frozenset) -> bool:
        toks = frozenset(t for t in toks if t not in _NAME_STOP)
        if not toks:
            return False
        if toks & _SUB_ORG:
            # a sub-org name only aligns if the poly side carries the marker too
            if not (poly_ev & _SUB_ORG):
                return False
        core = [t for t in toks if t not in _SUB_ORG]
        hit = sum(1 for t in core
                  if t in poly_ev
                  or any(t in o or o in t for o in poly_ev if len(t) >= 4)
                  or any(c.endswith(t[:3]) or t[:3] == c[-3:] or t[:6] in c
                         for c in poly_codes if len(c) >= 5))
        # at least one core token per participant must align by NAME (not 3-char code)
        return hit >= 1 and any(t in poly_ev or any(t in o for o in poly_ev if len(t) >= 4)
                                for t in core)

    parts = participants(ka_title)
    if len(parts) == 2:
        return side_aligns(parts[0]) and side_aligns(parts[1])
    # single-subject (futures/awards): the outcome name itself must align
    m = re.search(r" - (.+?)$", ka_title or "") or \
        re.search(r"^Will (.+?) (?:win|be|set)", ka_title or "")
    if m:
        return side_aligns(_tokens(m.group(1)))
    return False
