"""LLM access for the two LLM steps.

    QwenLLM   - Qwen/Qwen3-8B on the course inference API (OpenAI-compatible, HTTPS with a CA file)
    AzureLLM  - Azure OpenAI with strict JSON-schema output
    MockLLM   - offline, rule-based stand-in so the whole graph runs and tests without a key

get_llm() picks, in order: Qwen (if QWEN_API_KEY is set), Azure (if all AZURE_* are set),
else the mock. Settings come from the environment or a .env file in the project folder.

    QWEN_API_BASE, QWEN_API_KEY, QWEN_CA_CERT, [QWEN_MODEL]
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION
"""
from __future__ import annotations

import functools
import json
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from . import observability as obs
from .graph import ROOT
from .schemas import Claim, ClaimsOutput, CSCOOutput, MonitoringOutput, SelfCritiqueOutput

T = TypeVar("T", bound=BaseModel)

DISRUPTION_WORDS = {
    "geopolitical": ["war", "invasion", "conflict", "sanction", "embargo", "blockade", "military"],
    "natural_disaster": ["earthquake", "flood", "hurricane", "typhoon", "drought", "wildfire", "storm"],
    "industrial_accident": ["fire", "explosion", "blast", "collapse", "chemical leak"],
    "labour_action": ["strike", "walkout", "industrial action"],
    "cybersecurity": ["cyberattack", "ransomware", "cyber attack", "data breach", "hack"],
    "regulatory": ["export ban", "import ban", "tariff", "recall", "ban on"],
    "economic_crisis": ["bankruptcy", "insolvency", "collapse of", "default", "currency crisis"],
}
COUNTRY_ALIASES = {"russian": "Russia", "ukrainian": "Ukraine", "vietnamese": "Vietnam",
                   "chinese": "China", "turkish": "Turkey", "indian": "India"}


def _load_dotenv(path=ROOT / ".env") -> None:
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _traced(fn):
    """Wrap a complete_json so each call is a Langfuse generation (no-op when tracing is off)."""
    @functools.wraps(fn)
    def wrapper(self, step, system, user, schema, payload=None):
        if not obs.enabled():
            return fn(self, step, system, user, schema, payload)
        before = self.snapshot()
        with obs.generation(step, getattr(self, "model", self.name), system, user,
                            prompt=obs.prompt_for_step(step)) as gen:
            out = fn(self, step, system, user, schema, payload)
            if gen is not None:
                after = self.snapshot()
                pt, ct = after["prompt_tokens"] - before["prompt_tokens"], after["completion_tokens"] - before["completion_tokens"]
                obs.note(gen, output=obs._hide(out.model_dump()),
                         usage_details={"input": pt, "output": ct, "total": pt + ct},
                         metadata={"backend": self.name, "api_calls": after["calls"] - before["calls"]})
            return out
    wrapper._traced = True
    return wrapper


class BaseLLM:
    name = "LLM"

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        fn = cls.__dict__.get("complete_json")
        if fn is not None and not getattr(fn, "_traced", False):
            cls.complete_json = _traced(fn)         # every backend (and test double) is traced the same way

    def __init__(self):
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def complete_json(self, step: str, system: str, user: str, schema: type[T],
                      payload: dict | None = None) -> T:
        raise NotImplementedError

    def snapshot(self) -> dict:
        return dict(self.usage)


class AzureLLM(BaseLLM):
    name = "Azure OpenAI"

    def __init__(self):
        super().__init__()
        from openai import AzureOpenAI           # imported lazily so the mock needs no SDK
        self.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self.client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                                  api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                  api_version=os.environ["AZURE_OPENAI_API_VERSION"])

    def complete_json(self, step, system, user, schema, payload=None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_err = None
        for _ in range(2):                       # one retry if the output fails validation
            resp = self.client.beta.chat.completions.parse(
                model=self.deployment, messages=messages, response_format=schema, temperature=0)
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += resp.usage.prompt_tokens
            self.usage["completion_tokens"] += resp.usage.completion_tokens
            try:
                parsed = resp.choices[0].message.parsed
                if parsed is not None:
                    return parsed
                last_err = resp.choices[0].message.refusal or "empty response"
            except ValidationError as e:
                last_err = e
            messages.append({"role": "user", "content": f"Your last reply was invalid ({last_err}). "
                                                        "Reply again with valid JSON only."})
        raise RuntimeError(f"{step}: LLM failed to return valid JSON: {last_err}")


class QwenLLM(BaseLLM):
    """Qwen3-8B behind an OpenAI-compatible /chat/completions endpoint.

    Course API rules handled here: one active request per key (a lock serialises calls),
    max_tokens <= 1024, retry on 429/503 after 2, 4, 8 seconds, HTTPS verified with the
    group's CA certificate. The model has no strict-schema mode on this server, so the
    JSON schema goes in the prompt and the reply is validated with Pydantic (one repair retry).
    """
    name = "Qwen3-8B"
    _lock = threading.Lock()                     # shared by all instances: 1 request at a time per key
    RETRY_DELAYS = (2, 4, 8)

    def __init__(self, base_url: str, api_key: str, ca_cert: str, model: str = "Qwen/Qwen3-8B",
                 max_tokens: int = 1024, timeout: int = 180):
        super().__init__()
        cert = Path(ca_cert).expanduser()
        if not cert.is_absolute():
            cert = ROOT / cert
        if not cert.exists():
            raise FileNotFoundError(f"CA certificate not found: {cert} (set QWEN_CA_CERT)")
        self.url = base_url.rstrip("/") + "/chat/completions"     # base already ends in /v1
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.max_tokens = min(max_tokens, 1024)
        self.ctx = ssl.create_default_context(cafile=str(cert))   # verification stays on

    def _chat(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                           "max_tokens": self.max_tokens,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(self.url, body, {"Authorization": f"Bearer {self.api_key}",
                                                      "Content-Type": "application/json"})
        for attempt in range(len(self.RETRY_DELAYS) + 1):
            try:
                with self._lock, urllib.request.urlopen(req, timeout=self.timeout, context=self.ctx) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < len(self.RETRY_DELAYS):
                    obs.event("qwen_retry", status=e.code, attempt=attempt + 1, wait_s=self.RETRY_DELAYS[attempt])
                    time.sleep(self.RETRY_DELAYS[attempt])
                    continue
                detail = e.read().decode("utf-8", "replace")[:300]
                hint = {401: " (check QWEN_API_KEY and the URL for your group)",
                        404: " (use the base URL ending in /v1; /chat/completions is added for you)"}.get(e.code, "")
                raise RuntimeError(f"Qwen API error {e.code}{hint}: {detail}") from None
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += data.get("usage", {}).get("prompt_tokens", 0)
        self.usage["completion_tokens"] += data.get("usage", {}).get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"] or ""

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("no JSON object in the reply")
        return json.loads(text[start:end + 1])

    def complete_json(self, step, system, user, schema, payload=None):
        spec = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        messages = [{"role": "system", "content": f"{system}\n\nReply with ONLY a JSON object that matches "
                                                  f"this JSON schema, no other text:\n{spec}"},
                    {"role": "user", "content": user}]
        err = None
        for _ in range(2):                       # one repair retry
            reply = self._chat(messages)
            try:
                return schema.model_validate(self._extract_json(reply))
            except (ValueError, ValidationError) as e:    # json errors are ValueErrors
                err = e
                obs.event("json_repair_retry", error=str(e)[:200])
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": f"That reply was invalid ({str(err)[:300]}). "
                                                         "Reply again with ONLY the corrected JSON object."}]
        raise RuntimeError(f"{step}: Qwen did not return valid JSON: {str(err)[:300]}")


class MockLLM(BaseLLM):
    name = "offline mock"

    """Keyword-based stand-in. It reads the same inputs as the real LLM but is crude,
    so use it for plumbing and tests, not for judging extraction quality."""

    def __init__(self, countries: set[str], industries: set[str], companies: set[str]):
        super().__init__()
        self.countries, self.industries, self.companies = countries, industries, companies

    def complete_json(self, step, system, user, schema, payload=None):
        self.usage["calls"] += 1
        if schema is MonitoringOutput:
            return self._monitor(payload["article"])
        if schema is CSCOOutput:
            return self._csco(payload)
        if schema is SelfCritiqueOutput:
            return self._self_critique(user)
        if schema is ClaimsOutput:
            return self._claims(payload["article"])
        raise ValueError(f"MockLLM cannot handle {schema}")

    RED_FLAG_WORDS = {
        "sensational wording": ["shocking", "you won't believe", "exposed", "secret", "bombshell", "!!"],
        "vague or unnamed sourcing": ["sources say", "insiders", "reportedly", "rumour", "rumor",
                                      "unconfirmed", "anonymous", "it is said", "people are saying"],
        "calls to panic or share": ["share this", "panic", "stock up now", "wake up"],
    }

    def _claims(self, article: str) -> ClaimsOutput:
        """Crude: one claim per sentence; names found by vocabulary matching."""
        text = article.lower()
        claims = []
        for s in [x.strip() for x in re.split(r"(?<=[.!?])\s+", article.strip()) if x.strip()]:
            low = s.lower()
            companies = [c for c in sorted(self.companies) if c.lower() in low]
            # company-style names ("Zorblax Dynamics 777") that are not in the network
            companies += [m for m in re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)* \d{3}\b", s) if m not in companies]
            # a mock cannot tell event places from buyer places: for claims naming a company, only take a
            # country written right after the company ("Acme Foods 001 in Brazil")
            if companies:
                countries = [ct for ct in sorted(self.countries) for c in companies
                             if re.search(rf"{re.escape(c.lower())}(?:\s+[\w\-]+){{0,3}}?\s+(?:in|near)\s+"
                                          rf"{re.escape(ct.lower())}\b", low)]
            else:
                countries = [c for c in sorted(self.countries) if re.search(rf"\b{re.escape(c.lower())}\b", low)]
            kind = "event" if any(re.search(rf"\b{re.escape(w)}", low) for ws in DISRUPTION_WORDS.values()
                                  for w in ws) else "other"
            claims.append(Claim(text=s, kind=kind, companies=companies, countries=countries))
        flags = [name for name, words in self.RED_FLAG_WORDS.items() if any(w in text for w in words)]
        return ClaimsOutput(claims=claims, red_flags=flags)

    def _self_critique(self, user_prompt: str) -> SelfCritiqueOutput:
        suggestions: list[str] = []
        text = user_prompt.lower()
        if any(w in text for w in ("buyer", "importer", "destination")):
            suggestions.append("Check if listed countries are only buyers/importers and not disrupted producers.")
        if re.search(r"companies.*\b1\b", text) or "single company" in text:
            suggestions.append("Consider whether a broad industry label applies when only a single company is named.")
        if not suggestions:
            suggestions.append("Extraction appears reasonable. Double-check country and company specificity.")
        return SelfCritiqueOutput(suggestions=suggestions, confidence="medium")

    def _monitor(self, article: str) -> MonitoringOutput:
        text = article.lower()
        dtype = next((t for t, words in DISRUPTION_WORDS.items()
                      if any(re.search(rf"\b{re.escape(w)}", text) for w in words)), None)
        if dtype is None:
            return MonitoringOutput(reasoning="No disruption keywords found.", is_disruption=False,
                                    disruption_type="none", summary="No supply chain disruption reported.")
        companies = [c for c in sorted(self.companies) if c.lower() in text]
        rest = text
        for c in companies:                       # names like "Sakri Packaging 020" must not count as an industry
            rest = rest.replace(c.lower(), " ")
        countries = [c for c in sorted(self.countries) if re.search(rf"\b{re.escape(c.lower())}\b", rest)]
        countries += [v for k, v in COUNTRY_ALIASES.items()
                      if re.search(rf"\b{k}\b", rest) and v in self.countries and v not in countries]
        industries = [i for i in sorted(self.industries) if re.search(rf"\b{re.escape(i.lower())}\b", rest)]
        return MonitoringOutput(
            reasoning=f"Found {dtype} keywords; matched {len(countries)} countries, "
                      f"{len(industries)} industries, {len(companies)} companies.",
            is_disruption=True, disruption_type=dtype, summary=article.strip().split("\n")[0][:200],
            countries=countries, industries=industries, companies=companies)

    def _csco(self, p: dict) -> CSCOOutput:
        acts, bfs = p["actions"], p["bfs"]
        counts = {t: len(bfs[f"tier_{t}"]) for t in range(1, 5)}
        n = {lvl: sum(a["risk_level"] == lvl for a in acts) for lvl in ("HIGH", "MEDIUM", "LOW")}
        top = acts[0] if acts else None
        return CSCOOutput(
            executive_summary=(f"{p['disruption']['summary']} {bfs['disrupted_count']} suppliers in "
                               f"{bfs['focal']['name']}'s network are affected. "
                               f"Top-10 Tier-1 risks: {n['HIGH']} high, {n['MEDIUM']} medium, {n['LOW']} low."),
            network_impact=(f"Disrupted suppliers by tier: {counts}."
                            + (f" Highest risk: {top['name']} ({top['risk_score']})." if top else "")),
            key_recommendations=[f"{a['name']}: {a['action']}" for a in acts[:5]])


def get_llm(graph=None) -> BaseLLM:
    _load_dotenv()
    if os.environ.get("QWEN_API_KEY"):
        return QwenLLM(base_url=os.environ.get("QWEN_API_BASE", "https://103.42.50.41:1033/v1"),
                       api_key=os.environ["QWEN_API_KEY"],
                       ca_cert=os.environ.get("QWEN_CA_CERT", "vm05-ca.crt"),
                       model=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B"))
    needed = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT",
              "AZURE_OPENAI_API_VERSION"]
    if all(os.environ.get(k) for k in needed):
        return AzureLLM()
    from .tools import get_graph
    g = graph or get_graph()
    return MockLLM({d["country"] for _, d in g.nodes(data=True)},
                   {d["industry"] for _, d in g.nodes(data=True)},
                   {d["name"] for _, d in g.nodes(data=True)})
