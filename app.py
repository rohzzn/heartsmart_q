import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, urlparse

from flask import Flask, request, render_template_string
import requests

import parser as record_parser


try:
    # openai>=1.0.0
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


JsonObj = Dict[str, Any]


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """
    Minimal .env loader (no external deps).
    - Supports KEY=value or KEY="value" or KEY='value'
    - Ignores blank lines and comments
    - For auth/session keys, .env intentionally overrides shell env to avoid stale cookies
    """
    force_override = {
        "HEARTSMART_COOKIE_HEADER",
        "HEARTSMART_URL",
        "HEARTSMART_API_BASE",
        "HEARTSMART_REFERER",
    }
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if not k:
                    continue
                if (k in os.environ) and (k not in force_override):
                    continue
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                os.environ[k] = v
    except FileNotFoundError:
        return


_load_dotenv()

APP_TITLE = "HeartSmart Copilot"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
HEARTSMART_API_BASE = os.environ.get(
    "HEARTSMART_API_BASE",
    "https://devheartsmart.pcgcid.org/api/v2/freeze-2025-05-06",
).rstrip("/")
HEARTSMART_REFERER = os.environ.get(
    "HEARTSMART_REFERER",
    "https://devheartsmart.pcgcid.org/freeze-2025-05-06/results",
)
PREVIEW_ENDPOINT_PATH = "/query_tools/preview/"
try:
    HEARTSMART_PREVIEW_PAGE_SIZE = max(1, int(os.environ.get("HEARTSMART_PREVIEW_PAGE_SIZE", "38306")))
except ValueError:
    HEARTSMART_PREVIEW_PAGE_SIZE = 38306
try:
    HEARTSMART_API_TIMEOUT_SEC = max(30, int(os.environ.get("HEARTSMART_API_TIMEOUT_SEC", "180")))
except ValueError:
    HEARTSMART_API_TIMEOUT_SEC = 180


app = Flask(__name__)

_DATA_CACHE: Optional[JsonObj] = None
_FIELDS_CACHE: Optional[List[str]] = None
_LOAD_INFO: Optional[Tuple[float, int]] = None  # (seconds, row_count)
_LOAD_LOCK = threading.Lock()
_BACKGROUND_LOAD_THREAD: Optional[threading.Thread] = None
_BACKGROUND_LOAD_STARTED_AT: Optional[float] = None
_BACKGROUND_LOAD_ERROR: Optional[str] = None
_RUNTIME_API_BASE = HEARTSMART_API_BASE
_RUNTIME_PREVIEW_PATH = PREVIEW_ENDPOINT_PATH
_RUNTIME_PREVIEW_PAGE_SIZE = HEARTSMART_PREVIEW_PAGE_SIZE
_RUNTIME_REFERER = HEARTSMART_REFERER
_RUNTIME_COOKIE_HEADER = (os.getenv("HEARTSMART_COOKIE_HEADER", "") or "").strip()


def _runtime_data_source_label() -> str:
    return (
        f"{_RUNTIME_API_BASE}{_RUNTIME_PREVIEW_PATH}"
        f"?page=1&records_per_page={_RUNTIME_PREVIEW_PAGE_SIZE}"
    )


def _runtime_preview_url_for_form() -> str:
    return _runtime_data_source_label()


def _clear_runtime_cache_unlocked() -> None:
    global _DATA_CACHE, _FIELDS_CACHE, _LOAD_INFO
    global _BACKGROUND_LOAD_THREAD, _BACKGROUND_LOAD_STARTED_AT, _BACKGROUND_LOAD_ERROR
    _DATA_CACHE = None
    _FIELDS_CACHE = None
    _LOAD_INFO = None
    _BACKGROUND_LOAD_THREAD = None
    _BACKGROUND_LOAD_STARTED_AT = None
    _BACKGROUND_LOAD_ERROR = None


def _parse_preview_url_config(preview_url: str) -> Tuple[str, str, int]:
    raw = (preview_url or "").strip()
    if not raw:
        raise ValueError("Preview URL is required.")

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Preview URL must include protocol and host.")

    marker = "/query_tools/preview"
    path = parsed.path or ""
    idx = path.find(marker)
    if idx < 0:
        raise ValueError("Preview URL must include '/query_tools/preview'.")

    api_path = path[:idx].rstrip("/")
    if not api_path:
        raise ValueError("Preview URL is missing the API base path before '/query_tools/preview'.")

    preview_path = path[idx:]
    if not preview_path.endswith("/"):
        preview_path += "/"

    api_base = f"{parsed.scheme}://{parsed.netloc}{api_path}".rstrip("/")
    params = parse_qs(parsed.query or "")
    per_page = _RUNTIME_PREVIEW_PAGE_SIZE
    raw_per_page = (params.get("records_per_page") or [None])[0]
    if raw_per_page is not None and str(raw_per_page).strip():
        try:
            per_page = max(1, int(str(raw_per_page).strip()))
        except ValueError as exc:
            raise ValueError("records_per_page in Preview URL must be a positive integer.") from exc

    return api_base, preview_path, per_page


def _apply_runtime_connection_settings(
    preview_url: str,
    cookie_header: str,
    referer: str,
) -> None:
    global _RUNTIME_API_BASE, _RUNTIME_PREVIEW_PATH, _RUNTIME_PREVIEW_PAGE_SIZE
    global _RUNTIME_REFERER, _RUNTIME_COOKIE_HEADER

    api_base, preview_path, per_page = _parse_preview_url_config(preview_url)
    next_cookie = (cookie_header or "").strip() or _RUNTIME_COOKIE_HEADER
    next_referer = (referer or "").strip() or _RUNTIME_REFERER

    with _LOAD_LOCK:
        _RUNTIME_API_BASE = api_base
        _RUNTIME_PREVIEW_PATH = preview_path
        _RUNTIME_PREVIEW_PAGE_SIZE = per_page
        _RUNTIME_REFERER = next_referer
        _RUNTIME_COOKIE_HEADER = next_cookie
        _clear_runtime_cache_unlocked()

    os.environ["HEARTSMART_API_BASE"] = _RUNTIME_API_BASE
    os.environ["HEARTSMART_REFERER"] = _RUNTIME_REFERER
    os.environ["HEARTSMART_PREVIEW_PAGE_SIZE"] = str(_RUNTIME_PREVIEW_PAGE_SIZE)
    os.environ["HEARTSMART_COOKIE_HEADER"] = _RUNTIME_COOKIE_HEADER


INDEX_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{{ title }}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
      :root {
        --bg: #0f1115;
        --bg-soft: #171a21;
        --panel: #1f2430;
        --panel-2: #2a3140;
        --text: #ebeff8;
        --muted: #aeb6c8;
        --line: #374056;
        --brand: #19c39c;
        --brand-ink: #063d33;
        --danger: #f48d93;
        --ok: #7fe2bd;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Space Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
        background: radial-gradient(1100px 500px at -10% -30%, #2f3348 0%, transparent 55%), var(--bg);
      }
      .app {
        max-width: 1200px;
        margin: 0 auto;
        min-height: 100vh;
        padding: 14px 16px 24px;
        display: grid;
        grid-template-rows: auto auto 1fr auto;
        gap: 12px;
      }
      .top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid var(--line);
        background: rgba(32, 38, 50, 0.75);
        border-radius: 14px;
        padding: 10px 12px;
        backdrop-filter: blur(8px);
      }
      .title {
        margin: 0;
        font-size: clamp(18px, 2.5vw, 24px);
        letter-spacing: -0.01em;
      }
      .meta {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }
      .chip {
        border: 1px solid #3b4458;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
        color: var(--muted);
        background: #1b202d;
      }
      .thread {
        border: 1px solid var(--line);
        background: var(--bg-soft);
        border-radius: 16px;
        padding: 14px;
        display: grid;
        gap: 14px;
        align-content: start;
      }
      .msg {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px;
        background: var(--panel);
      }
      .msg.user {
        margin-left: auto;
        max-width: min(900px, 100%);
        border-color: #2f6f5f;
        background: #12382f;
      }
      .msg.assistant {
        max-width: 100%;
      }
      .msg h3 {
        margin: 0 0 8px;
        font-size: 16px;
      }
      .text {
        margin: 0;
        color: var(--text);
        line-height: 1.45;
        white-space: pre-wrap;
      }
      .meta-grid {
        margin-top: 10px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 8px;
      }
      .meta-card {
        border: 1px solid #3a445a;
        border-radius: 10px;
        background: var(--panel-2);
        padding: 8px 10px;
      }
      .meta-card h4 {
        margin: 0 0 4px;
        font-size: 11px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .meta-card p {
        margin: 0;
        line-height: 1.4;
        font-size: 13px;
        word-break: break-word;
      }
      .ok { color: var(--ok); }
      .danger { color: var(--danger); }
      .block-title {
        margin: 12px 0 6px;
        font-size: 13px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.03em;
      }
      ul {
        margin: 0;
        padding-left: 18px;
        display: grid;
        gap: 4px;
      }
      li {
        color: var(--text);
        line-height: 1.4;
        word-break: break-word;
      }
      details {
        border: 1px solid #3a445a;
        background: #1a1f2b;
        border-radius: 12px;
      }
      summary {
        cursor: pointer;
        padding: 10px 12px;
        font-weight: 700;
      }
      pre, code {
        font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      }
      pre {
        margin: 0;
        padding: 10px 12px;
        border-top: 1px solid #394359;
        background: #151b26;
        overflow: auto;
        color: #d5def2;
      }
      .table-wrap {
        border-top: 1px solid #394359;
        overflow: auto;
        max-height: 70vh;
      }
      table {
        width: max-content;
        min-width: 100%;
        border-collapse: collapse;
        table-layout: auto;
      }
      th, td {
        padding: 8px 9px;
        border-bottom: 1px solid #313a4f;
        font-size: 12px;
        vertical-align: top;
        white-space: nowrap;
        word-break: normal;
        overflow-wrap: normal;
      }
      th {
        text-align: left;
        position: sticky;
        top: 0;
        background: #202838;
        color: #c8d3ee;
        z-index: 1;
      }
      tbody tr:nth-child(odd) td {
        background: #1b2231;
      }
      .composer {
        border: 1px solid var(--line);
        background: rgba(27, 32, 45, 0.95);
        border-radius: 14px;
        padding: 12px;
      }
      .composer-grid {
        width: 100%;
        display: grid;
        grid-template-columns: 1fr 140px 120px;
        gap: 10px;
        align-items: end;
      }
      label {
        display: block;
        margin-bottom: 5px;
        font-size: 11px;
        text-transform: uppercase;
        color: var(--muted);
        letter-spacing: 0.03em;
        font-weight: 700;
      }
      textarea, input[type="number"], input[type="text"] {
        width: 100%;
        border: 1px solid #414b62;
        border-radius: 10px;
        background: #121722;
        color: var(--text);
        font: inherit;
        font-size: 14px;
        padding: 9px 10px;
        outline: none;
      }
      textarea {
        min-height: 90px;
        resize: vertical;
      }
      textarea:focus, input[type="number"]:focus, input[type="text"]:focus {
        border-color: #62d5b6;
        box-shadow: 0 0 0 3px rgba(98, 213, 182, 0.18);
      }
      button {
        width: 100%;
        border: none;
        border-radius: 10px;
        font: inherit;
        font-weight: 700;
        color: var(--brand-ink);
        background: linear-gradient(135deg, #5cf3c7, #24cfaa);
        padding: 11px 10px;
        cursor: pointer;
      }
      .composer-note {
        margin: 6px 0 0;
        font-size: 12px;
        color: var(--muted);
      }
      .dev-tools {
        border: 1px solid var(--line);
        background: rgba(27, 32, 45, 0.8);
        border-radius: 14px;
      }
      .dev-tools details {
        border: none;
        background: transparent;
      }
      .dev-content {
        padding: 0 12px 12px;
      }
      .dev-content h4 {
        margin: 10px 0 6px;
        font-size: 12px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.03em;
      }
      @media (max-width: 930px) {
        .composer-grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main class="app">
      <header class="top">
        <h1 class="title">{{ title }}</h1>
      </header>

      <section class="meta">
        <div class="chip">Data Source: <code>{{ data_source }}</code></div>
        {% if load_status %}
          <div class="chip">{{ load_status }}</div>
        {% endif %}
        {% if load_info %}
          <div class="chip">Loaded: {{ "%.2f"|format(load_info[0]) }}s</div>
          <div class="chip">Rows: {{ load_info[1] }}</div>
        {% endif %}
        <div class="chip">Max rows: {% if limit == 0 %}ALL{% else %}{{ limit }}{% endif %}</div>
      </section>

      <section class="thread">
        {% if q %}
          <article class="msg user">
            <p class="text">{{ q }}</p>
          </article>
        {% endif %}

        {% if error %}
          <article class="msg assistant">
            <h3 class="danger">Error</h3>
            <p class="text">{{ error }}</p>
          </article>
        {% endif %}

        {% if settings_message %}
          <article class="msg assistant">
            <p class="text ok">{{ settings_message }}</p>
          </article>
        {% endif %}

        {% if assistant_summary %}
          <article class="msg assistant">
            <p class="text">{{ assistant_summary }}</p>

            {% if requested_collections or applied_collections or unavailable_collections or server_summary or notes %}
              <div class="meta-grid">
                {% if requested_collections %}
                  <div class="meta-card">
                    <h4>Requested</h4>
                    <p>{{ requested_collections }}</p>
                  </div>
                {% endif %}
                {% if applied_collections %}
                  <div class="meta-card">
                    <h4>Applied</h4>
                    <p class="ok">{{ applied_collections }}</p>
                  </div>
                {% endif %}
                {% if unavailable_collections %}
                  <div class="meta-card">
                    <h4>Not Applied</h4>
                    <p class="danger">{{ unavailable_collections }}</p>
                  </div>
                {% endif %}
                {% if server_summary %}
                  <div class="meta-card">
                    <h4>Server Summary</h4>
                    <p>{{ server_summary }}</p>
                  </div>
                {% endif %}
                {% if notes %}
                  <div class="meta-card">
                    <h4>Notes</h4>
                    <p>{{ notes }}</p>
                  </div>
                {% endif %}
              </div>
            {% endif %}

            {% if table_rows is not none %}
              <details>
                <summary>Show Full Table Response ({{ table_rows|length }} rows, {{ table_columns|length }} columns)</summary>
                {% if table_rows|length == 0 %}
                  <p class="text" style="padding: 10px 12px;">No rows matched.</p>
                {% else %}
                  <div class="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th style="width: 54px;">#</th>
                          {% for c in table_columns %}
                            <th>{{ c }}</th>
                          {% endfor %}
                        </tr>
                      </thead>
                      <tbody>
                        {% for r in table_rows %}
                          <tr>
                            <td>{{ loop.index }}</td>
                            {% for cell in r %}
                              <td>{{ cell }}</td>
                            {% endfor %}
                          </tr>
                        {% endfor %}
                      </tbody>
                    </table>
                  </div>
                {% endif %}
              </details>
            {% endif %}
          </article>
        {% endif %}

        {% if not q %}
          <article class="msg assistant">
            <h3>How To Ask</h3>
            <p class="text">Example: cpt and cmri_data subjects with Maternal Age greater than 20 and Cohort Source is Legacy.</p>
          </article>
        {% endif %}
      </section>

      <section class="composer">
        <form method="post" action="settings">
          <div class="composer-grid">
            <div>
              <label>HeartSmart Preview URL</label>
              <input
                type="text"
                name="preview_url"
                value="{{ preview_url or '' }}"
                placeholder="https://.../query_tools/preview/?page=1&records_per_page=38306"
              />
              <p class="composer-note">Paste the exact preview API URL.</p>
            </div>
            <div>
              <label>Cookie Header</label>
              <textarea name="cookie_header" placeholder="sessionid=...; cookie2=..."></textarea>
              <p class="composer-note">Leave blank to keep current cookie.</p>
            </div>
            <div>
              <label>Referer (optional)</label>
              <input type="text" name="referer" value="{{ referer or '' }}" placeholder="https://.../results" />
              <button type="submit">Update API Connection</button>
            </div>
          </div>
        </form>

        <form method="post" action="query">
          <div class="composer-grid">
            <div>
              <label>Message</label>
              <textarea name="q" placeholder="Describe the cohort and filters you want...">{{ q or "" }}</textarea>
            </div>
            <div>
              <label>Max Results</label>
              <input type="number" name="limit" min="0" max="200000" value="{{ limit if limit is not none else 0 }}" />
              <p class="composer-note">Use 0 for all rows.</p>
            </div>
            <div>
              <label>Run</label>
              <button type="submit">Send</button>
            </div>
          </div>
        </form>
      </section>

      <section class="dev-tools">
        <details>
          <summary>Developer Tools</summary>
          <div class="dev-content">
            {% if spec %}
              <h4>Generated Filter Spec</h4>
              <pre>{{ spec }}</pre>
            {% endif %}
            {% if query_to_run %}
              <h4>Query Debug</h4>
              <pre>{{ query_to_run }}</pre>
            {% endif %}
            <h4>Available Fields (sample)</h4>
            <pre>{{ fields_preview }}</pre>
          </div>
        </details>
      </section>
    </main>
    {% if load_status and not q %}
      <script>
        setTimeout(function () {
          window.location.reload();
        }, 4000);
      </script>
    {% endif %}
  </body>
</html>
"""


ALLOWED_OPS: Set[str] = {
    "exists",
    "isnull",
    "eq",
    "ne",
    "in",
    "nin",
    "contains",
    "startswith",
    "endswith",
    "regex",
    "gt",
    "gte",
    "lt",
    "lte",
}

SITE_COLLECTIONS: List[Dict[str, str]] = [
    {"permanent_id": "cmri_data", "name": "cmri_data", "plural": "cmri_datas"},
    {"permanent_id": "copy_number_result", "name": "Copy Number Result", "plural": "Copy Number Results"},
    {"permanent_id": "emrdata_cpt", "name": "CPT", "plural": "CPTs"},
    {"permanent_id": "emrdata_computed_phenotype", "name": "emrdata_computed_phenotype", "plural": "emrdata_computed_phenotypes"},
    {"permanent_id": "emrdata_echo", "name": "emrdata_echo", "plural": "emrdata_echos"},
    {"permanent_id": "emrdata_ipccc", "name": "emrdata_ipccc", "plural": "emrdata_ipcccs"},
    {"permanent_id": "emrdata_socialdeprivation", "name": "emrdata_socialdeprivation", "plural": "emrdata_socialdeprivations"},
    {"permanent_id": "emrdata_vitals", "name": "emrdata_vitals", "plural": "emrdata_vitalss"},
    {"permanent_id": "emrdata_encounters", "name": "Encounters", "plural": "Encounterss"},
    {"permanent_id": "fish_result", "name": "Fish Result", "plural": "Fish Results"},
    {"permanent_id": "fyler_diagnoses", "name": "Fyler Diagnoses", "plural": "Fyler Diagnosess"},
    {"permanent_id": "genomics_analysis", "name": "Genomics Analysis", "plural": "Genomics Analysiss"},
    {"permanent_id": "genomics_data", "name": "Genomics Data", "plural": "Genomics Datas"},
    {"permanent_id": "genomics_metadata", "name": "Genomics Metadata", "plural": "Genomics Metadatas"},
    {"permanent_id": "emrdata_hpo", "name": "HPO", "plural": "HPOs"},
    {"permanent_id": "icd_10_cm", "name": "ICD-10-CM", "plural": "ICD-10-CMs"},
    {"permanent_id": "icd_10_pcs", "name": "ICD-10-PCS", "plural": "ICD-10-PCSs"},
    {"permanent_id": "icd_9_cm", "name": "ICD-9-CM", "plural": "ICD-9-CMs"},
    {"permanent_id": "icd_9_pcs", "name": "ICD-9-PCS", "plural": "ICD-9-PCSs"},
    {"permanent_id": "karyotype", "name": "Karyotype", "plural": "Karyotypes"},
    {"permanent_id": "emrdata_labs", "name": "Labs", "plural": "Labss"},
    {"permanent_id": "microarray_result", "name": "Microarray Result", "plural": "Microarray Results"},
    {"permanent_id": "mutation_result", "name": "Mutation Result", "plural": "Mutation Results"},
    {"permanent_id": "other_genetic_test_result", "name": "Other Genetic Test Result", "plural": "Other Genetic Test Results"},
    {"permanent_id": "outcomes_one", "name": "Outcomes One", "plural": "Outcomes Ones"},
    {"permanent_id": "outcomes_over_one", "name": "Outcomes Over One", "plural": "Outcomes Over Ones"},
    {"permanent_id": "outcomes_survey", "name": "Outcomes Survey", "plural": "Outcomes Surveys"},
    {"permanent_id": "emrdata_phecode", "name": "Phecode", "plural": "Phecodes"},
    {"permanent_id": "pregnancy_birth_history", "name": "Pregnancy Birth History", "plural": "Pregnancy Birth Historys"},
    {"permanent_id": "emrdata_rxnorm", "name": "RxNorm", "plural": "RxNorms"},
    {"permanent_id": "sample", "name": "Sample", "plural": "Samples"},
    {"permanent_id": "subject", "name": "Subject", "plural": "Subjects"},
]

SITE_COLLECTION_IDS: Set[str] = {c["permanent_id"] for c in SITE_COLLECTIONS}
SITE_COLLECTION_NAME_BY_ID: Dict[str, str] = {c["permanent_id"]: c["name"] for c in SITE_COLLECTIONS}

# Heuristic mapping from a site collection to matching local flat-row fields.
COLLECTION_FIELD_HINTS: Dict[str, List[str]] = {
    "sample": ["sample", "dna sample"],
    "emrdata_hpo": ["hpo"],
    "genomics_data": ["genotyping", "wes", "wgs", "whole exome", "whole genome", "rna-seq", "resequencing", "lrwgs", "topmed", "trio"],
    "genomics_analysis": ["genotyping", "wes", "wgs", "whole exome", "whole genome", "rna-seq", "resequencing", "lrwgs", "topmed", "trio"],
    "mutation_result": ["genotyping", "mutation", "resequencing"],
    "other_genetic_test_result": ["genotyping", "mips", "microarray", "karyotype", "fish", "resequencing"],
    "emrdata_labs": ["lab"],
    "emrdata_rxnorm": ["rxnorm", "medication"],
    "emrdata_phecode": ["phecode"],
    "emrdata_vitals": ["vital", "bmi", "heart rate", "blood pressure"],
    "emrdata_encounters": ["encounter", "visit"],
    "outcomes_one": ["outcome", "death", "hearing"],
    "outcomes_over_one": ["outcome", "death", "hearing"],
    "outcomes_survey": ["survey", "questionnaire"],
    "pregnancy_birth_history": ["maternal", "paternal", "birth", "pregnancy"],
    "emrdata_cpt": ["cpt"],
}


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _build_collection_alias_map() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for c in SITE_COLLECTIONS:
        cid = c["permanent_id"]
        names = {
            c["permanent_id"],
            c["name"],
            c["plural"],
            c["permanent_id"].replace("_", " "),
        }
        # "subject" as a raw word is too broad for NL parsing, so keep it explicit.
        if cid == "subject":
            names |= {"subject collection", "subject records", "root subject"}
            names.discard("subject")
        for n in names:
            norm = _normalize_for_match(n)
            if norm:
                aliases[norm] = cid
    return aliases


COLLECTION_ALIAS_TO_ID = _build_collection_alias_map()


def extract_collection_filters_from_text(nl_query: str) -> List[str]:
    """
    Finds collection mentions in a free-text query using known IDs/names.
    """
    q = _normalize_for_match(nl_query)
    if not q:
        return []

    found: Set[str] = set()
    for alias, cid in COLLECTION_ALIAS_TO_ID.items():
        if re.search(rf"(^|\s){re.escape(alias)}(\s|$)", q):
            found.add(cid)
    return sorted(found)


def _resolve_collection_id(value: str) -> Optional[str]:
    raw = (value or "").strip()
    if not raw:
        return None

    if raw in SITE_COLLECTION_IDS:
        return raw

    candidates: List[str] = [raw]
    without_parens = re.sub(r"\([^)]*\)", " ", raw).strip()
    if without_parens and without_parens not in candidates:
        candidates.append(without_parens)
    for inner in re.findall(r"\(([^)]+)\)", raw):
        inner = inner.strip()
        if inner and inner not in candidates:
            candidates.append(inner)

    for cand in candidates:
        norm = _normalize_for_match(cand)
        if not norm:
            continue
        if norm in COLLECTION_ALIAS_TO_ID:
            return COLLECTION_ALIAS_TO_ID[norm]
        maybe_id = norm.replace(" ", "_")
        if maybe_id in SITE_COLLECTION_IDS:
            return maybe_id
    return None


def normalize_collection_list(raw_collections: Any) -> List[str]:
    items: List[str] = []
    if isinstance(raw_collections, str):
        items = [raw_collections]
    elif isinstance(raw_collections, list):
        items = [v for v in raw_collections if isinstance(v, str)]
    else:
        return []

    out: Set[str] = set()
    for item in items:
        parts = re.split(r",|;|\band\b|\bor\b", item, flags=re.IGNORECASE)
        for part in parts:
            cid = _resolve_collection_id(part)
            if cid:
                out.add(cid)
    return sorted(out)


def _is_meaningful_value(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict, tuple, set)):
        return len(v) > 0
    # bool/int/float are considered meaningful if present.
    return True


def infer_collection_field_map(fields: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    lowered = [(f, f.lower()) for f in fields]
    for cid, hints in COLLECTION_FIELD_HINTS.items():
        matches: List[str] = []
        for f, lf in lowered:
            if any(h in lf for h in hints):
                matches.append(f)
        if matches:
            out[cid] = sorted(set(matches))
    return out


def apply_collection_filters(
    rows: List[JsonObj],
    requested_collections: List[str],
    fields: List[str],
) -> Tuple[List[JsonObj], List[str], List[str]]:
    """
    Applies collection filters on top of the parser spec match.
    Returns: (filtered_rows, applied_collections, unavailable_collections)
    """
    requested = [c for c in requested_collections if c in SITE_COLLECTION_IDS]
    if not requested:
        return rows, [], []

    field_map = infer_collection_field_map(fields)

    # "subject" is the root collection in the site response, so it does not reduce rows.
    if "subject" in requested:
        unavailable_for_display = [c for c in requested if c != "subject" and c not in field_map]
        return rows, ["subject"], unavailable_for_display

    available = [c for c in requested if c in field_map]
    unavailable = [c for c in requested if c not in field_map]

    if not available:
        return rows, [], unavailable

    def row_matches_collection(row: JsonObj, cid: str) -> bool:
        keys = field_map.get(cid, [])
        return any(_is_meaningful_value(row.get(k)) for k in keys)

    filtered = [r for r in rows if any(row_matches_collection(r, cid) for cid in available)]
    return filtered, available, unavailable


def _parse_cookie_header(cookie_header_value: str) -> Dict[str, str]:
    cookies: Dict[str, str] = {}
    for part in (cookie_header_value or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            cookies[k] = v
    return cookies


def _heartsmart_session() -> requests.Session:
    cookie_header = (_RUNTIME_COOKIE_HEADER or "").strip()
    cookies = _parse_cookie_header(cookie_header)
    if not cookies:
        raise RuntimeError("HEARTSMART_COOKIE_HEADER is missing; cannot call live HeartSmart API.")

    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Referer": _RUNTIME_REFERER,
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
            ),
        }
    )
    s.cookies.update(cookies)
    return s


def _api_get(session: requests.Session, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{_RUNTIME_API_BASE}{path}"
    r = session.get(url, params=params, timeout=HEARTSMART_API_TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()


def _api_post(session: requests.Session, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{_RUNTIME_API_BASE}{path}"
    r = session.post(url, json=payload, timeout=HEARTSMART_API_TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()


def _is_auth_error(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        resp = exc.response
        if resp is not None and resp.status_code in {401, 403}:
            return True
    text = str(exc).lower()
    return ("unauthorized" in text) or ("forbidden" in text) or ("401" in text) or ("403" in text)


def _preview_payload_to_rows(payload: Dict[str, Any]) -> List[JsonObj]:
    fields = payload.get("extended_table_def", {}).get("fields", [])
    rows = payload.get("data", [])
    keys: List[str] = []
    for f in fields:
        if isinstance(f, dict):
            key = f.get("concept_name") or f.get("label") or f.get("entry_id")
            keys.append(key)
    out: List[JsonObj] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        obj = {keys[i]: row[i] for i in range(min(len(keys), len(row)))}
        out.append(obj)
    return out


def _fetch_preview_all_rows(
    session: requests.Session,
    per_page: Optional[int] = None,
) -> Tuple[List[JsonObj], Dict[str, Any]]:
    fetch_page_size = per_page if isinstance(per_page, int) and per_page > 0 else _RUNTIME_PREVIEW_PAGE_SIZE
    first = _api_get(session, _RUNTIME_PREVIEW_PATH, params={"page": 1, "records_per_page": fetch_page_size})
    all_rows = _preview_payload_to_rows(first)

    paginator = first.get("paginator", {}) if isinstance(first.get("paginator"), dict) else {}
    last_page = paginator.get("last_page", 1)
    if not isinstance(last_page, int) or last_page < 1:
        last_page = 1

    for page in range(2, last_page + 1):
        nxt = _api_get(session, _RUNTIME_PREVIEW_PATH, params={"page": page, "records_per_page": fetch_page_size})
        all_rows.extend(_preview_payload_to_rows(nxt))

    meta = {
        "record_count": first.get("record_count"),
        "subject_count": first.get("subject_count"),
        "warnings": first.get("warnings", []),
        "errors": first.get("errors", []),
        "paginator": first.get("paginator"),
    }
    return all_rows, meta


def run_remote_collection_query(
    requested_collections: List[str],
) -> Tuple[List[JsonObj], Dict[str, Any], List[str], List[str]]:
    """
    Applies real site collection filters via /cohort_def and returns preview rows.
    Returns: (rows, meta, applied_collections, unavailable_collections)
    """
    requested = [c for c in requested_collections if c in SITE_COLLECTION_IDS]
    if not requested:
        return [], {}, [], []

    session = _heartsmart_session()

    # Reset cohort criteria each query so filters do not accumulate unexpectedly.
    clear_resp = _api_post(session, "/cohort_def/", {"transformation": {"type": "clear_all"}})
    clear_errors = clear_resp.get("errors") if isinstance(clear_resp, dict) else None
    if clear_errors:
        raise RuntimeError(f"Failed to clear cohort filters: {clear_errors}")

    applied: List[str] = []
    unavailable: List[str] = []
    try:
        for cid in requested:
            try:
                resp = _api_post(
                    session,
                    "/cohort_def/",
                    {"transformation": {"type": "add_criteria_set", "collection_id": cid}},
                )
                errs = resp.get("errors") if isinstance(resp, dict) else None
                if errs:
                    unavailable.append(cid)
                else:
                    applied.append(cid)
            except Exception:
                unavailable.append(cid)

        rows, preview_meta = _fetch_preview_all_rows(session)
        count_meta = _api_get(session, "/query_tools/count/")

        meta = {
            "record_count": preview_meta.get("record_count"),
            "subject_count": preview_meta.get("subject_count"),
            "count": count_meta.get("count"),
            "warnings": preview_meta.get("warnings", []),
            "errors": preview_meta.get("errors", []),
        }
        return rows, meta, applied, unavailable
    finally:
        # Avoid leaking scoped cohort filters into subsequent "full dataset" preview calls.
        try:
            _api_post(session, "/cohort_def/", {"transformation": {"type": "clear_all"}})
        except Exception:
            pass


def _fields_in_spec(spec: Dict[str, Any]) -> List[str]:
    out: List[str] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if "and" in node and isinstance(node["and"], list):
            for child in node["and"]:
                walk(child)
            return
        if "or" in node and isinstance(node["or"], list):
            for child in node["or"]:
                walk(child)
            return
        if "not" in node:
            walk(node["not"])
            return
        field = node.get("field")
        if isinstance(field, str) and field not in out:
            out.append(field)

    walk(spec)
    return out


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_results_table(
    rows: List[JsonObj],
    preferred_columns: List[str],
    max_columns: Optional[int] = None,
) -> Tuple[List[str], List[List[str]], bool]:
    key_order: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k in row.keys():
            if k not in key_order:
                key_order.append(k)

    front: List[str] = []
    for c in preferred_columns:
        if c in key_order and c not in front:
            front.append(c)

    # Good default context columns if present in the dataset.
    defaults = [
        "Blinded ID",
        "Cohort Source",
        "Maternal Age",
        "Paternal Age",
        "Gender",
        "Enrollment Site",
    ]
    for c in defaults:
        if c in key_order and c not in front:
            front.append(c)

    remaining = [k for k in key_order if k not in front]
    all_columns = front + remaining
    if max_columns is None:
        selected = all_columns
    else:
        selected = all_columns[:max_columns]
    truncated = len(all_columns) > len(selected)

    table_rows: List[List[str]] = []
    for row in rows:
        table_rows.append([_cell_text(row.get(c)) for c in selected])

    return selected, table_rows, truncated


_OP_LABELS: Dict[str, str] = {
    "eq": "=",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "contains": "contains",
    "startswith": "starts with",
    "endswith": "ends with",
    "in": "in",
    "nin": "not in",
    "regex": "matches",
    "exists": "exists",
    "isnull": "is null",
}


def spec_to_human_text(spec: Any) -> str:
    if not isinstance(spec, dict):
        return ""
    if "and" in spec and isinstance(spec["and"], list):
        items = [spec_to_human_text(s) for s in spec["and"]]
        items = [x for x in items if x]
        return " AND ".join(items) if items else "No field constraints"
    if "or" in spec and isinstance(spec["or"], list):
        items = [spec_to_human_text(s) for s in spec["or"]]
        items = [x for x in items if x]
        return "(" + " OR ".join(items) + ")" if items else ""
    if "not" in spec:
        inner = spec_to_human_text(spec.get("not"))
        return f"NOT ({inner})" if inner else ""

    field = spec.get("field")
    op = spec.get("op")
    if not isinstance(field, str) or not isinstance(op, str):
        return ""

    op_label = _OP_LABELS.get(op, op)
    if op in {"exists", "isnull"}:
        return f"{field} {op_label}"

    value = spec.get("value")
    if isinstance(value, str):
        rhs = f'"{value}"'
    elif isinstance(value, (list, dict)):
        rhs = json.dumps(value, ensure_ascii=False)
    else:
        rhs = str(value)
    return f"{field} {op_label} {rhs}"


def _preferred_id_field_from_keys(keys: Set[str]) -> Optional[str]:
    for k in [
        "Blinded ID",
        "Subject ID",
        "subject_id",
        "Subject",
        "ID",
        "id",
        "Sample ID (All)",
    ]:
        if k in keys:
            return k
    return None


def _preferred_id_field(rows: List[JsonObj]) -> Optional[str]:
    if not rows:
        return None
    keys: Set[str] = set()
    for r in rows[:10]:
        if isinstance(r, dict):
            keys.update(r.keys())
    return _preferred_id_field_from_keys(keys)


def _preferred_id_field_from_fields(fields: List[str]) -> Optional[str]:
    return _preferred_id_field_from_keys(set(fields))


def extract_subject_id_token(nl_query: str) -> Optional[str]:
    # HeartSmart IDs are usually shaped like "1-00079".
    m = re.search(r"\b\d{1,4}-\d{3,}\b", nl_query or "")
    return m.group(0) if m else None


def add_subject_id_constraint(spec: Dict[str, Any], id_field: str, subject_id: str) -> Dict[str, Any]:
    wanted = {"field": id_field, "op": "eq", "value": subject_id}

    def has_wanted(node: Any) -> bool:
        if not isinstance(node, dict):
            return False
        if node.get("field") == id_field and node.get("op") == "eq":
            return _cell_text(node.get("value")).strip() == subject_id
        if "and" in node and isinstance(node["and"], list):
            return any(has_wanted(ch) for ch in node["and"])
        if "or" in node and isinstance(node["or"], list):
            return any(has_wanted(ch) for ch in node["or"])
        if "not" in node:
            return has_wanted(node["not"])
        return False

    if has_wanted(spec):
        return spec
    if isinstance(spec, dict) and "and" in spec and isinstance(spec["and"], list):
        return {"and": [*spec["and"], wanted]}
    return {"and": [spec, wanted]}


def _count_unique_people(rows: List[JsonObj]) -> Optional[int]:
    id_field = _preferred_id_field(rows)
    if not id_field:
        return None
    ids: Set[str] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        pid = _cell_text(r.get(id_field)).strip()
        if pid:
            ids.add(pid)
    return len(ids)


def _is_count_query(nl_query: str) -> bool:
    q = (nl_query or "").lower()
    return bool(re.search(r"\b(how many|count|number of|total)\b", q))


def _meaningful_row_pairs(row: JsonObj, max_items: int = 8) -> List[str]:
    preferred = [
        "Cohort Source",
        "Gender",
        "Maternal Age",
        "Paternal Age",
        "Enrollment Site",
        "Working Group",
        "Consent Group",
        "Relationship",
    ]
    out: List[str] = []
    used: Set[str] = set()

    for key in preferred:
        v = row.get(key)
        if _is_meaningful_value(v):
            out.append(f"{key}: {_cell_text(v)}")
            used.add(key)
        if len(out) >= max_items:
            return out

    for k, v in row.items():
        if k in used:
            continue
        if _is_meaningful_value(v):
            out.append(f"{k}: {_cell_text(v)}")
        if len(out) >= max_items:
            break
    return out


def build_query_to_run_text(
    nl_query: str,
    requested_collections: List[str],
    applied_collections: List[str],
    spec: Dict[str, Any],
) -> str:
    collection_part = ", ".join(applied_collections or requested_collections) or "none"
    human_filter = spec_to_human_text(spec) or "No field constraints"
    return (
        f"Natural language: {nl_query}\n"
        f"Collections: {collection_part}\n"
        f"Field filters: {human_filter}\n"
        f"Spec JSON: {json.dumps(spec, ensure_ascii=False)}"
    )


def build_assistant_summary(
    nl_query: str,
    rows: List[JsonObj],
    matched_count: int,
    requested_collections: List[str],
    applied_collections: List[str],
    unavailable_collections: List[str],
) -> str:
    if applied_collections:
        collection_text = ", ".join(applied_collections)
    elif requested_collections:
        collection_text = "the local dataset (requested collections could not be applied)"
    else:
        collection_text = "the current dataset"
    unavailable_text = ", ".join(unavailable_collections)
    is_count = _is_count_query(nl_query)

    if matched_count == 0 and not is_count:
        msg = f"I could not find matches for that request in {collection_text}."
        if unavailable_collections:
            msg += f" Some requested collections were not applied: {unavailable_text}."
        return msg

    unique_people = _count_unique_people(rows)
    people_count = unique_people if unique_people is not None else matched_count

    subject_id = extract_subject_id_token(nl_query)
    id_field = _preferred_id_field(rows)
    if subject_id and id_field:
        row = next(
            (
                r
                for r in rows
                if isinstance(r, dict) and _cell_text(r.get(id_field)).strip() == subject_id
            ),
            None,
        )
        if row:
            row_count_for_subject = sum(
                1
                for r in rows
                if isinstance(r, dict) and _cell_text(r.get(id_field)).strip() == subject_id
            )
            detail_pairs = _meaningful_row_pairs(row, max_items=8)
            details = ", ".join(detail_pairs) if detail_pairs else "No additional populated fields were found."
            msg = f"I found subject {subject_id} in {collection_text}."
            if row_count_for_subject > 1:
                msg += f" There are {row_count_for_subject} matching rows for this subject."
            msg += f" {details}"
            if unavailable_collections:
                msg += f" Some requested collections were not applied: {unavailable_text}."
            return msg

    if is_count:
        noun = "person" if people_count == 1 else "people"
        verb = "is" if people_count == 1 else "are"
        msg = f"There {verb} {people_count} {noun} matching your query in {collection_text}."
        if unique_people is not None and matched_count != people_count:
            msg += f" That corresponds to {matched_count} matching rows."
        if unavailable_collections:
            msg += f" Some requested collections were not applied: {unavailable_text}."
        return msg

    msg = f"I found {people_count} matching people in {collection_text}."
    if unique_people is not None and matched_count != people_count:
        msg += f" This corresponds to {matched_count} matching rows."
    if unavailable_collections:
        msg += f" Some requested collections were not applied: {unavailable_text}."
    return msg


def _cache_is_ready() -> bool:
    return _DATA_CACHE is not None and _FIELDS_CACHE is not None and _LOAD_INFO is not None


def _fields_preview_text(fields: List[str]) -> str:
    if not fields:
        return "(No API fields loaded yet.)"
    return "\n".join(fields[:200]) + ("" if len(fields) <= 200 else f"\n... (+{len(fields)-200} more)")


def _background_load_worker() -> None:
    global _BACKGROUND_LOAD_ERROR
    try:
        load_data_once()
        with _LOAD_LOCK:
            _BACKGROUND_LOAD_ERROR = None
    except Exception as exc:
        with _LOAD_LOCK:
            _BACKGROUND_LOAD_ERROR = str(exc)


def _ensure_background_data_load() -> None:
    global _BACKGROUND_LOAD_THREAD, _BACKGROUND_LOAD_STARTED_AT, _BACKGROUND_LOAD_ERROR
    with _LOAD_LOCK:
        if _cache_is_ready():
            return
        if _BACKGROUND_LOAD_THREAD is not None and _BACKGROUND_LOAD_THREAD.is_alive():
            return
        _BACKGROUND_LOAD_ERROR = None
        _BACKGROUND_LOAD_STARTED_AT = time.time()
        _BACKGROUND_LOAD_THREAD = threading.Thread(
            target=_background_load_worker,
            name="heartsmart-data-load",
            daemon=True,
        )
        _BACKGROUND_LOAD_THREAD.start()


def _load_state_snapshot() -> Tuple[bool, Optional[str], Optional[str], List[str], Optional[Tuple[float, int]]]:
    with _LOAD_LOCK:
        if _cache_is_ready():
            return True, None, None, list(_FIELDS_CACHE or []), _LOAD_INFO
        if _BACKGROUND_LOAD_THREAD is not None and _BACKGROUND_LOAD_THREAD.is_alive():
            elapsed = 0
            if _BACKGROUND_LOAD_STARTED_AT is not None:
                elapsed = max(0, int(time.time() - _BACKGROUND_LOAD_STARTED_AT))
            return False, f"Loading in background... {elapsed}s", None, [], None
        if _BACKGROUND_LOAD_ERROR:
            return False, None, _BACKGROUND_LOAD_ERROR, [], None
        return False, "Preparing background load...", None, [], None


def load_data_once() -> Tuple[JsonObj, List[str], Tuple[float, int]]:
    global _DATA_CACHE, _FIELDS_CACHE, _LOAD_INFO, _BACKGROUND_LOAD_ERROR
    if _cache_is_ready():
        return _DATA_CACHE, _FIELDS_CACHE, _LOAD_INFO  # type: ignore[return-value]

    with _LOAD_LOCK:
        if _cache_is_ready():
            return _DATA_CACHE, _FIELDS_CACHE, _LOAD_INFO  # type: ignore[return-value]

        t0 = time.time()
        session = _heartsmart_session()
        # Ensure base preview load is unscoped by leftover cohort criteria.
        clear_resp = _api_post(session, "/cohort_def/", {"transformation": {"type": "clear_all"}})
        clear_errors = clear_resp.get("errors") if isinstance(clear_resp, dict) else None
        if clear_errors:
            raise RuntimeError(f"Failed to clear cohort filters before preview load: {clear_errors}")
        rows, meta = _fetch_preview_all_rows(session, per_page=_RUNTIME_PREVIEW_PAGE_SIZE)
        data: JsonObj = {
            "rows_as_objects": rows,
            "source": _runtime_data_source_label(),
            "meta": meta,
        }
        fields: List[str] = []
        for r in rows:
            if isinstance(r, dict):
                fields = sorted(list(r.keys()))
                break
        t1 = time.time()
        info = (t1 - t0, len(rows))

        _DATA_CACHE = data
        _FIELDS_CACHE = fields
        _LOAD_INFO = info
        _BACKGROUND_LOAD_ERROR = None
        return data, fields, info


def _spec_depth(spec: Any, depth: int = 0) -> int:
    if not isinstance(spec, dict):
        return depth
    if "and" in spec and isinstance(spec["and"], list):
        return max([depth] + [_spec_depth(s, depth + 1) for s in spec["and"]])
    if "or" in spec and isinstance(spec["or"], list):
        return max([depth] + [_spec_depth(s, depth + 1) for s in spec["or"]])
    if "not" in spec:
        return _spec_depth(spec["not"], depth + 1)
    return depth + 1


def validate_spec(spec: Any, allowed_fields: Set[str]) -> None:
    """
    Defensive validation. We do NOT execute arbitrary code; we only accept
    a small JSON structure compatible with parser.matches().
    """
    if not isinstance(spec, dict):
        raise ValueError("Spec must be a JSON object")
    if _spec_depth(spec) > 12:
        raise ValueError("Spec too deep/complex")

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            raise ValueError("Invalid node (must be object)")

        keys = set(node.keys())
        if keys & {"and", "or", "not"}:
            if "and" in node:
                if not isinstance(node["and"], list) or len(node["and"]) > 50:
                    raise ValueError("'and' must be a list (max 50)")
                for child in node["and"]:
                    walk(child)
            if "or" in node:
                if not isinstance(node["or"], list) or len(node["or"]) > 50:
                    raise ValueError("'or' must be a list (max 50)")
                for child in node["or"]:
                    walk(child)
            if "not" in node:
                walk(node["not"])
            # allow only logical keys at this node
            extra = keys - {"and", "or", "not"}
            if extra:
                raise ValueError(f"Unexpected keys in logical node: {sorted(extra)}")
            return

        # leaf condition
        if "field" not in node:
            raise ValueError("Leaf condition missing 'field'")
        if "op" not in node:
            raise ValueError("Leaf condition missing 'op'")
        field = node["field"]
        op = node["op"]
        if not isinstance(field, str) or field not in allowed_fields:
            raise ValueError(f"Unknown field: {field!r}")
        if not isinstance(op, str) or op not in ALLOWED_OPS:
            raise ValueError(f"Unsupported op: {op!r}")
        if op in {"exists", "isnull"}:
            extra = keys - {"field", "op"}
            if extra:
                raise ValueError(f"Unexpected keys for op {op}: {sorted(extra)}")
            return
        # value-required ops
        if "value" not in node:
            raise ValueError(f"Leaf condition op {op!r} requires 'value'")
        extra = keys - {"field", "op", "value"}
        if extra:
            raise ValueError(f"Unexpected keys in leaf node: {sorted(extra)}")

    walk(spec)


def build_llm_prompt(nl_query: str, fields: List[str]) -> str:
    # Include parser logic (as requested), plus the available fields and schema.
    # Keep it tight so we don't waste tokens on the huge dataset.
    parser_code = (
        "Parser logic summary (verbatim structure):\n"
        "- Spec supports:\n"
        '  {"and": [<spec|cond>, ...]}\n'
        '  {"or":  [<spec|cond>, ...]}\n'
        '  {"not": <spec|cond>}\n'
        '- Leaf condition is:\n'
        '  {"field": "<field name>", "op": "<op>", "value": <any>}\n'
        '  For ops "exists" and "isnull", omit "value".\n'
        "- Allowed ops: exists, isnull, eq, ne, in, nin, contains, startswith, endswith, regex, gt, gte, lt, lte\n"
        "- Numeric compares (gt/gte/lt/lte) coerce strings to numbers when possible.\n"
    )

    field_list = ", ".join(fields[:300])
    if len(fields) > 300:
        field_list += f", ... (+{len(fields)-300} more)"

    collections_list = "\n".join(
        f"- {c['permanent_id']} ({c['name']})"
        for c in SITE_COLLECTIONS
    )

    return f"""
You convert an English query into a STRICT JSON filter spec for the parser described below.

RULES:
- Output MUST be valid JSON only (no markdown, no commentary).
- Output MUST be an object with keys: "spec" and optional "notes" and optional "collections".
- "spec" MUST follow the parser schema exactly.
- Use ONLY the provided field names exactly as written (case/spacing).
- Prefer "and" to combine multiple constraints.
- If the user asks for "age" without clarifying, pick the most explicit matching field name; add a brief note in "notes".
- If the user asks for site-level collection filters (example: Labs, RxNorm, emrdata_hpo), do not turn those into field conditions.
- Put those collection filters in "collections" as a list of collection permanent_id values.
- If no field constraint is requested, use {{"and": []}} for "spec".
- Handle natural-language comparators:
  - "older than"/"greater than"/">" -> gt
  - "at least"/">=" -> gte
  - "less than"/"<" -> lt
  - "at most"/"<=" -> lte
- Handle natural-language set logic:
  - "either/or" -> or
  - "not"/"exclude"/"without" -> not or ne/nin where appropriate
- Handle natural-language string logic:
  - "contains"/"includes" -> contains
  - "starts with" -> startswith
  - "ends with" -> endswith

{parser_code}

Available fields (use exact strings):
{field_list}

Available site collections (use permanent_id in "collections"):
{collections_list}

User query:
{nl_query}
""".strip()


def call_openai_for_spec(nl_query: str, fields: List[str]) -> Tuple[Dict[str, Any], Optional[str], List[str]]:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install -r requirements.txt")
    client = OpenAI()

    prompt = build_llm_prompt(nl_query, fields)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a precise translator from English to JSON filters."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        # best-effort extract first JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model did not return JSON")
        obj = json.loads(content[start : end + 1])

    if not isinstance(obj, dict) or "spec" not in obj:
        raise ValueError("Model response must be a JSON object with a 'spec' key")
    notes = obj.get("notes")
    spec = obj["spec"]
    if not isinstance(spec, dict):
        raise ValueError("'spec' must be a JSON object")
    raw_collections = obj.get("collections")
    collections = normalize_collection_list(raw_collections)
    return spec, notes if isinstance(notes, str) else None, sorted(set(collections))


@app.post("/settings")
def update_settings():
    preview_url = (request.form.get("preview_url") or "").strip()
    cookie_header = (request.form.get("cookie_header") or "").strip()
    referer = (request.form.get("referer") or "").strip()

    settings_message: Optional[str] = None
    error: Optional[str] = None
    try:
        _apply_runtime_connection_settings(preview_url=preview_url, cookie_header=cookie_header, referer=referer)
        _ensure_background_data_load()
        settings_message = "API connection updated. Loading data in background."
    except Exception as exc:
        error = str(exc)

    _, load_status, load_error, fields, info = _load_state_snapshot()
    fields_preview = _fields_preview_text(fields)
    return render_template_string(
        INDEX_TEMPLATE,
        title=APP_TITLE,
        data_source=_runtime_data_source_label(),
        preview_url=_runtime_preview_url_for_form(),
        referer=_RUNTIME_REFERER,
        settings_message=settings_message,
        load_status=load_status,
        load_info=info,
        q="",
        limit=0,
        spec=None,
        notes=None,
        table_columns=None,
        table_rows=None,
        columns_truncated=False,
        matched_count=None,
        error=error or load_error,
        requested_collections=None,
        applied_collections=None,
        unavailable_collections=None,
        server_summary=None,
        assistant_summary=None,
        person_details=None,
        query_to_run=None,
        fields_preview=fields_preview,
    )


@app.get("/")
def index():
    _ensure_background_data_load()
    _, load_status, load_error, fields, info = _load_state_snapshot()
    error = load_error
    fields_preview = _fields_preview_text(fields)
    return render_template_string(
        INDEX_TEMPLATE,
        title=APP_TITLE,
        data_source=_runtime_data_source_label(),
        preview_url=_runtime_preview_url_for_form(),
        referer=_RUNTIME_REFERER,
        settings_message=None,
        load_status=load_status,
        load_info=info,
        q="",
        limit=0,
        spec=None,
        notes=None,
        table_columns=None,
        table_rows=None,
        columns_truncated=False,
        matched_count=None,
        error=error,
        requested_collections=None,
        applied_collections=None,
        unavailable_collections=None,
        server_summary=None,
        assistant_summary=None,
        person_details=None,
        query_to_run=None,
        fields_preview=fields_preview,
    )


@app.post("/query")
def query():
    nl_query = (request.form.get("q") or "").strip()
    limit_raw = (request.form.get("limit") or "0").strip()
    try:
        limit = int(limit_raw)
    except ValueError:
        limit = 0
    limit = max(0, min(200000, limit))

    _ensure_background_data_load()
    ready, load_status, load_error, _, _ = _load_state_snapshot()
    if not ready:
        msg = (
            load_error
            if load_error
            else "Data is still loading from HeartSmart API in the background. Please try again in a few seconds."
        )
        return render_template_string(
            INDEX_TEMPLATE,
            title=APP_TITLE,
            data_source=_runtime_data_source_label(),
            preview_url=_runtime_preview_url_for_form(),
            referer=_RUNTIME_REFERER,
            settings_message=None,
            load_status=load_status,
            load_info=None,
            q=nl_query,
            limit=limit,
            spec=None,
            notes=None,
            table_columns=None,
            table_rows=None,
            columns_truncated=False,
            matched_count=None,
            error=msg,
            requested_collections=None,
            applied_collections=None,
            unavailable_collections=None,
            server_summary=None,
            assistant_summary=None,
            person_details=None,
            query_to_run=None,
            fields_preview="(No API fields loaded yet.)",
        )

    try:
        data, fields, info = load_data_once()
    except Exception as e:
        return render_template_string(
            INDEX_TEMPLATE,
            title=APP_TITLE,
            data_source=_runtime_data_source_label(),
            preview_url=_runtime_preview_url_for_form(),
            referer=_RUNTIME_REFERER,
            settings_message=None,
            load_status=None,
            load_info=None,
            q=nl_query,
            limit=limit,
            spec=None,
            notes=None,
            table_columns=None,
            table_rows=None,
            columns_truncated=False,
            matched_count=None,
            error=str(e),
            requested_collections=None,
            applied_collections=None,
            unavailable_collections=None,
            server_summary=None,
            assistant_summary=None,
            person_details=None,
            query_to_run=None,
            fields_preview="(No API fields loaded yet.)",
        )

    allowed_fields = set(fields)
    fields_preview = _fields_preview_text(fields)

    if not nl_query:
        return render_template_string(
            INDEX_TEMPLATE,
            title=APP_TITLE,
            data_source=_runtime_data_source_label(),
            preview_url=_runtime_preview_url_for_form(),
            referer=_RUNTIME_REFERER,
            settings_message=None,
            load_status=None,
            load_info=info,
            q=nl_query,
            limit=limit,
            spec=None,
            notes=None,
            table_columns=None,
            table_rows=None,
            columns_truncated=False,
            matched_count=None,
            error="Please enter a query.",
            requested_collections=None,
            applied_collections=None,
            unavailable_collections=None,
            server_summary=None,
            assistant_summary=None,
            person_details=None,
            query_to_run=None,
            fields_preview=fields_preview,
        )

    try:
        spec, notes, llm_collections = call_openai_for_spec(nl_query, fields)
        subject_id_token = extract_subject_id_token(nl_query)
        id_field_for_query = _preferred_id_field_from_fields(fields)
        if subject_id_token and id_field_for_query:
            spec = add_subject_id_constraint(spec, id_field_for_query, subject_id_token)
        validate_spec(spec, allowed_fields)

        extracted_collections = extract_collection_filters_from_text(nl_query)
        requested_collections = sorted(set(llm_collections + extracted_collections))
        server_summary: Optional[str] = None

        if requested_collections:
            try:
                remote_rows, remote_meta, applied_collections, unavailable_collections = run_remote_collection_query(
                    requested_collections
                )
                matched = record_parser.filter_rows({"rows_as_objects": remote_rows}, spec)
                server_summary = (
                    f"count={remote_meta.get('count')}, "
                    f"subject_count={remote_meta.get('subject_count')}, "
                    f"record_count={remote_meta.get('record_count')}"
                )
            except Exception as remote_err:
                if _is_auth_error(remote_err):
                    raise RuntimeError(
                        "HeartSmart session is unauthorized (401/403), so collection filters cannot be applied. "
                        "Update Cookie Header using 'Update API Connection' on this page and run the query again."
                    )
                # Fallback to local heuristic behavior if remote API call fails.
                matched = record_parser.filter_rows(data, spec)
                matched, applied_collections, unavailable_collections = apply_collection_filters(
                    matched,
                    requested_collections,
                    fields,
                )
                extra = (
                    "Remote cohort API unavailable; used local heuristic fallback. "
                    "Collection scoping may be incomplete. "
                    f"Error: {remote_err}"
                )
                notes = f"{notes} | {extra}" if notes else extra
                server_summary = "Remote cohort API call failed; fallback mode used."
        else:
            matched = record_parser.filter_rows(data, spec)
            applied_collections = []
            unavailable_collections = []

        matched_count = len(matched)
        shown = matched if limit == 0 else matched[:limit]
        preferred_columns = _fields_in_spec(spec)
        table_columns, table_rows, columns_truncated = build_results_table(shown, preferred_columns, max_columns=None)
        assistant_summary = build_assistant_summary(
            nl_query=nl_query,
            rows=matched,
            matched_count=matched_count,
            requested_collections=requested_collections,
            applied_collections=applied_collections,
            unavailable_collections=unavailable_collections,
        )
        query_to_run = build_query_to_run_text(
            nl_query=nl_query,
            requested_collections=requested_collections,
            applied_collections=applied_collections,
            spec=spec,
        )
        return render_template_string(
            INDEX_TEMPLATE,
            title=APP_TITLE,
            data_source=_runtime_data_source_label(),
            preview_url=_runtime_preview_url_for_form(),
            referer=_RUNTIME_REFERER,
            settings_message=None,
            load_status=None,
            load_info=info,
            q=nl_query,
            limit=limit,
            spec=json.dumps(spec, ensure_ascii=False, indent=2),
            notes=notes,
            table_columns=table_columns,
            table_rows=table_rows,
            columns_truncated=columns_truncated,
            matched_count=matched_count,
            error=None,
            requested_collections=", ".join(
                f"{c} ({SITE_COLLECTION_NAME_BY_ID.get(c, c)})" for c in requested_collections
            ) or None,
            applied_collections=", ".join(
                f"{c} ({SITE_COLLECTION_NAME_BY_ID.get(c, c)})" for c in applied_collections
            ) or None,
            unavailable_collections=", ".join(
                f"{c} ({SITE_COLLECTION_NAME_BY_ID.get(c, c)})" for c in unavailable_collections
            ) or None,
            server_summary=server_summary,
            assistant_summary=assistant_summary,
            person_details=None,
            query_to_run=query_to_run,
            fields_preview=fields_preview,
        )
    except Exception as e:
        return render_template_string(
            INDEX_TEMPLATE,
            title=APP_TITLE,
            data_source=_runtime_data_source_label(),
            preview_url=_runtime_preview_url_for_form(),
            referer=_RUNTIME_REFERER,
            settings_message=None,
            load_status=None,
            load_info=info,
            q=nl_query,
            limit=limit,
            spec=None,
            notes=None,
            table_columns=None,
            table_rows=None,
            columns_truncated=False,
            matched_count=None,
            error=str(e),
            requested_collections=None,
            applied_collections=None,
            unavailable_collections=None,
            server_summary=None,
            assistant_summary=None,
            person_details=None,
            query_to_run=None,
            fields_preview=fields_preview,
        )


if __name__ == "__main__":
    # For local dev. In production, run via gunicorn/uwsgi.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5050")), debug=True)
