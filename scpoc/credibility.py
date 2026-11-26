"""Credibility check: is this article genuine, suspicious or likely fake?

The LLM only extracts the article's claims and warning signs (ClaimsOutput). The evidence
for each claim and the verdict are plain Python, so they are repeatable and explainable:

  1. Graph grounding     do the named companies / countries / industries exist in the network,
                         and does a company really sit in the country and industry claimed?
  2. Internal checks     contradictions between claims and style / sourcing red flags (from the LLM)
  3. Corroboration       if RAG examples are on, how many of the closest labelled historic articles
                         were false positives

There is no live web access, so this cannot prove an event happened. "likely_fake" means the
article fails checks the system can make; a person still decides.
"""
from __future__ import annotations

from . import tools
from .schemas import ClaimCheck, ClaimsOutput, CredibilityOutput

# suspicion points
CONTRADICTED_CLAIM = 2     # a claim conflicts with the network (company in the wrong country / industry)
LLM_CONTRADICTION = 2      # two claims in the article contradict each other (each, capped)
UNKNOWN_COMPANY = 1        # a named company is not in the network (cannot be verified)
RED_FLAG = 1               # each style / sourcing warning sign (capped)
RAG_FAKE_VOTES = 1         # most of the closest labelled examples were false positives
MAX_RED_FLAG_POINTS = 2
MAX_CONTRADICTION_POINTS = 4
FAKE_AT, SUSPICIOUS_AT = 3, 1


def check_claims(claims: ClaimsOutput, g, *, examples: list[dict] | None = None) -> CredibilityOutput:
    countries = {d["country"].lower(): d["country"] for _, d in g.nodes(data=True)}
    industries = {d["industry"].lower(): d["industry"] for _, d in g.nodes(data=True)}
    checks, points, reasons, unknown = [], 0, [], set()

    for c in claims.claims:
        status, why = "supported", []
        resolved = []
        for name in c.companies:
            hit = tools.resolve_entity(name, g=g)["match"]
            if hit and hit["score"] >= 0.8:
                resolved.append(hit)
            else:
                status = "unverifiable" if status == "supported" else status
                why.append(f"'{name}' is not in the supplier network, so it cannot be verified")
                if name.lower() not in unknown:          # count each unknown company once
                    unknown.add(name.lower())
                    points += UNKNOWN_COMPANY
        for ctry in c.countries:
            if ctry.lower() not in countries:
                status = "unverifiable" if status == "supported" else status
                why.append(f"country '{ctry}' is not in the network")
        for ind in c.industries:
            if ind.lower() not in industries:
                status = "unverifiable" if status == "supported" else status
                why.append(f"industry '{ind}' is not a known industry")
        # attribute consistency: only for an event claim naming exactly one company and one place
        if c.kind == "event" and len(resolved) == 1 and len(c.countries) == 1 and c.countries[0].lower() in countries:
            actual = g.nodes[resolved[0]["company_id"]]["country"]
            if actual.lower() != c.countries[0].lower():
                status = "contradicted"
                why.append(f"{resolved[0]['name']} is based in {actual}, not {countries[c.countries[0].lower()]}")
                points += CONTRADICTED_CLAIM
        if status == "supported" and resolved:
            why.append("named companies exist in the network")
        checks.append(ClaimCheck(claim=c.text, status=status, reasons=why))

    for text in claims.contradictions:
        reasons.append(f"Contradiction: {text}")
    points += min(len(claims.contradictions) * LLM_CONTRADICTION, MAX_CONTRADICTION_POINTS)
    for text in claims.red_flags:
        reasons.append(f"Warning sign: {text}")
    points += min(len(claims.red_flags) * RED_FLAG, MAX_RED_FLAG_POINTS)

    if examples:
        fp = sum(1 for e in examples if e.get("kind") == "false_positive")
        if fp * 2 > len(examples):
            points += RAG_FAKE_VOTES
            reasons.append(f"{fp} of the {len(examples)} closest labelled historic articles were false positives")

    for ch in checks:
        if ch.status != "supported":
            reasons.append(f"{ch.status.capitalize()}: {ch.claim} ({'; '.join(ch.reasons)})")
    if not claims.claims:
        reasons.append("No checkable claims were found in the text")
        points += RED_FLAG

    verdict = "likely_fake" if points >= FAKE_AT else "suspicious" if points >= SUSPICIOUS_AT else "genuine"
    if verdict == "genuine":
        reasons.append("No contradictions or warning signs found; claims are consistent with the network")
    return CredibilityOutput(verdict=verdict, score=points, checks=checks,
                             contradictions=list(claims.contradictions),
                             red_flags=list(claims.red_flags), reasons=reasons)
