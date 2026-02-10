import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from flask import Flask, request, render_template_string

import parser as record_parser


try:
    # openai>=1.0.0
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


JsonObj = Dict[str, Any]

APP_TITLE = "Concept Pairs Query (LLM → parser spec)"
DATA_PATH = os.environ.get("CONCEPT_PAIRS_PATH", "preview_concept_pairs.json")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


app = Flask(__name__)

_DATA_CACHE: Optional[JsonObj] = None
_FIELDS_CACHE: Optional[List[str]] = None
_LOAD_INFO: Optional[Tuple[float, int]] = None  # (seconds, row_count)


INDEX_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{{ title }}</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
      .wrap { max-width: 1100px; margin: 0 auto; }
      textarea { width: 100%; min-height: 90px; padding: 10px; font-size: 14px; }
      input[type="number"] { width: 120px; padding: 6px; }
      button { padding: 10px 14px; cursor: pointer; }
      code, pre { background: #f6f7f8; padding: 2px 4px; border-radius: 6px; }
      pre { padding: 12px; overflow: auto; }
      .row { display: flex; gap: 14px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
      .muted { color: #666; font-size: 13px; }
      .err { color: #b00020; }
      .ok { color: #0b6b0b; }
      table { border-collapse: collapse; width: 100%; margin-top: 12px; }
      td, th { border: 1px solid #ddd; padding: 8px; vertical-align: top; font-size: 13px; }
      th { background: #fafafa; text-align: left; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h2>{{ title }}</h2>
      <div class="muted">
        Data file: <code>{{ data_path }}</code>
        {% if load_info %}
          — loaded in {{ "%.2f"|format(load_info[0]) }}s ({{ load_info[1] }} rows)
        {% endif %}
      </div>

      <form method="post" action="/query">
        <div class="row">
          <div style="flex: 1 1 650px;">
            <label class="muted"><b>Ask in plain English</b> (e.g. "fetch all records with Maternal Age greater than 20 and Cohort Source is Legacy")</label>
            <textarea name="q" placeholder="Your query...">{{ q or "" }}</textarea>
          </div>
          <div>
            <label class="muted"><b>Max results</b></label><br/>
            <input type="number" name="limit" min="1" max="5000" value="{{ limit or 200 }}" />
          </div>
          <div>
            <label class="muted"><b>&nbsp;</b></label><br/>
            <button type="submit">Run</button>
          </div>
        </div>
      </form>

      {% if error %}
        <p class="err"><b>Error:</b> {{ error }}</p>
      {% endif %}

      {% if spec %}
        <h3>LLM filter spec (executed by parser)</h3>
        <pre>{{ spec }}</pre>
      {% endif %}

      {% if notes %}
        <div class="muted"><b>LLM notes:</b> {{ notes }}</div>
      {% endif %}

      {% if results is not none %}
        <h3>Results</h3>
        <div class="muted">
          Matched: <b class="ok">{{ matched_count }}</b>
          — showing first <b>{{ results|length }}</b>
        </div>
        {% if results|length == 0 %}
          <p class="muted">No rows matched.</p>
        {% else %}
          <table>
            <thead>
              <tr>
                <th style="width: 55px;">#</th>
                <th>Row (JSON)</th>
              </tr>
            </thead>
            <tbody>
              {% for r in results %}
                <tr>
                  <td>{{ loop.index }}</td>
                  <td><pre style="margin:0;">{{ r }}</pre></td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
        {% endif %}
      {% endif %}

      <h3>Available fields (sample)</h3>
      <div class="muted">
        The model is constrained to these keys. If you refer to "age", specify which one (e.g. <code>Maternal Age</code>, <code>Paternal Age</code>, <code>Age At Form Completion</code>).
      </div>
      <pre>{{ fields_preview }}</pre>
    </div>
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


def load_data_once() -> Tuple[JsonObj, List[str], Tuple[float, int]]:
    global _DATA_CACHE, _FIELDS_CACHE, _LOAD_INFO
    if _DATA_CACHE is not None and _FIELDS_CACHE is not None and _LOAD_INFO is not None:
        return _DATA_CACHE, _FIELDS_CACHE, _LOAD_INFO

    t0 = time.time()
    data = record_parser.load_json(DATA_PATH)
    rows = data.get("rows_as_objects", [])
    if not isinstance(rows, list):
        raise ValueError("Expected preview file to contain 'rows_as_objects' list")
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

    return f"""
You convert an English query into a STRICT JSON filter spec for the parser described below.

RULES:
- Output MUST be valid JSON only (no markdown, no commentary).
- Output MUST be an object with keys: "spec" and optional "notes".
- "spec" MUST follow the parser schema exactly.
- Use ONLY the provided field names exactly as written (case/spacing).
- Prefer "and" to combine multiple constraints.
- If the user asks for "age" without clarifying, pick the most explicit matching field name; add a brief note in "notes".

{parser_code}

Available fields (use exact strings):
{field_list}

User query:
{nl_query}
""".strip()


def call_openai_for_spec(nl_query: str, fields: List[str]) -> Tuple[Dict[str, Any], Optional[str]]:
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
    return spec, notes if isinstance(notes, str) else None


@app.get("/")
def index():
    data, fields, info = load_data_once()
    fields_preview = "\n".join(fields[:200]) + ("" if len(fields) <= 200 else f"\n... (+{len(fields)-200} more)")
    return render_template_string(
        INDEX_TEMPLATE,
        title=APP_TITLE,
        data_path=DATA_PATH,
        load_info=info,
        q="",
        limit=200,
        spec=None,
        notes=None,
        results=None,
        matched_count=None,
        error=None,
        fields_preview=fields_preview,
    )


@app.post("/query")
def query():
    data, fields, info = load_data_once()
    allowed_fields = set(fields)

    nl_query = (request.form.get("q") or "").strip()
    limit_raw = (request.form.get("limit") or "200").strip()
    try:
        limit = int(limit_raw)
    except ValueError:
        limit = 200
    limit = max(1, min(5000, limit))

    fields_preview = "\n".join(fields[:200]) + ("" if len(fields) <= 200 else f"\n... (+{len(fields)-200} more)")

    if not nl_query:
        return render_template_string(
            INDEX_TEMPLATE,
            title=APP_TITLE,
            data_path=DATA_PATH,
            load_info=info,
            q=nl_query,
            limit=limit,
            spec=None,
            notes=None,
            results=None,
            matched_count=None,
            error="Please enter a query.",
            fields_preview=fields_preview,
        )

    try:
        spec, notes = call_openai_for_spec(nl_query, fields)
        validate_spec(spec, allowed_fields)
        matched = record_parser.filter_rows(data, spec)
        matched_count = len(matched)
        shown = matched[:limit]
        shown_pretty = [json.dumps(r, ensure_ascii=False, indent=2) for r in shown]
        return render_template_string(
            INDEX_TEMPLATE,
            title=APP_TITLE,
            data_path=DATA_PATH,
            load_info=info,
            q=nl_query,
            limit=limit,
            spec=json.dumps(spec, ensure_ascii=False, indent=2),
            notes=notes,
            results=shown_pretty,
            matched_count=matched_count,
            error=None,
            fields_preview=fields_preview,
        )
    except Exception as e:
        return render_template_string(
            INDEX_TEMPLATE,
            title=APP_TITLE,
            data_path=DATA_PATH,
            load_info=info,
            q=nl_query,
            limit=limit,
            spec=None,
            notes=None,
            results=None,
            matched_count=None,
            error=str(e),
            fields_preview=fields_preview,
        )


if __name__ == "__main__":
    # For local dev. In production, run via gunicorn/uwsgi.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5050")), debug=True)

