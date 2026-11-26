"""Phase 2 - typed objects shared by every step of the graph."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

DisruptionType = Literal["geopolitical", "economic_crisis", "natural_disaster",
                         "cybersecurity", "industrial_accident", "labour_action",
                         "regulatory", "other", "none"]


class MonitoringOutput(BaseModel):
    """Strict JSON returned by the Monitoring step (the LLM)."""
    reasoning: str = Field(description="Short step-by-step reasoning (chain of thought).")
    is_disruption: bool
    disruption_type: DisruptionType
    summary: str
    countries: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)


class CSCOOutput(BaseModel):
    """Strict JSON returned by the CSCO step (the LLM). Actions themselves stay deterministic."""
    executive_summary: str
    network_impact: str
    key_recommendations: list[str]


class SelfCritiqueOutput(BaseModel):
    """Output of the self-critique step."""
    suggestions: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


ClaimKind = Literal["event", "cause", "impact", "quantity", "timing", "other"]


class Claim(BaseModel):
    """One atomic factual claim made by the article."""
    text: str
    kind: ClaimKind = "other"
    companies: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list, description="Where the event happens, not buyers.")
    industries: list[str] = Field(default_factory=list)


class ClaimsOutput(BaseModel):
    """Strict JSON returned by the claim-extraction step (the LLM)."""
    claims: list[Claim] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list,
                                      description="Pairs of claims that cannot both be true.")
    red_flags: list[str] = Field(default_factory=list,
                                 description="Style or sourcing warning signs (no named source, sensational "
                                             "wording, vague attribution, missing place or date).")


class ClaimCheck(BaseModel):
    """Evidence for or against one claim (computed without the LLM)."""
    claim: str
    status: Literal["supported", "unverifiable", "contradicted"]
    reasons: list[str] = Field(default_factory=list)


class CredibilityOutput(BaseModel):
    """Overall credibility verdict, with the evidence behind it."""
    verdict: Literal["genuine", "suspicious", "likely_fake"]
    score: int = 0                       # suspicion points; higher = less credible
    checks: list[ClaimCheck] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class State(BaseModel):
    """Everything the pipeline knows. Each node reads this and returns a partial update."""
    # inputs
    article: str
    focal_name: str

    # Langfuse tracing (None when tracing is off); kept in the state so a run is one trace across the review pause
    trace_id: str | None = None
    session_id: str | None = None

    # retrieval (only when RAG flags are on): {"examples": [...], "entities": [...]}
    retrieved: dict[str, Any] = Field(default_factory=dict)

    # credibility check (only when build_app(credibility=True))
    credibility: CredibilityOutput | None = None

    # monitoring
    monitoring: MonitoringOutput | None = None

    # knowledge-graph query
    focal_id: str | None = None
    criteria: dict[str, list[str]] = Field(default_factory=dict)   # graph-resolved countries/industries/companies
    unresolved: list[str] = Field(default_factory=list)            # things the LLM named that aren't in the graph
    bfs: dict[str, Any] | None = None

    # risk / actions / visual
    risk: dict[str, Any] | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)
    network_html: str | None = None

    # CSCO + human review
    csco: CSCOOutput | None = None
    approval: Literal["approved", "edited", "rejected", "retry", "edit_extraction"] | None = None

    # AI retry (human rejects and asks the LLM to re-analyse with feedback)
    retry_count: int = 0
    retry_feedback: str = ""
    previous_monitoring: MonitoringOutput | None = None

    # alternative sourcing
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    # candidates shown to the reviewer BEFORE approval (top suppliers by risk, any level)
    alternatives_preview: list[dict[str, Any]] = Field(default_factory=list)

    # AI self-critique suggestions for the reviewer
    ai_suggestions: list[str] = Field(default_factory=list)
    ai_confidence: str = ""

    # outcome and bookkeeping
    status: Literal["running", "no_exposure", "rejected", "completed", "flagged_fake"] = "running"
    message: str = ""
    trace: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)  # per-step timing/tokens
