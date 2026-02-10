import json
import os
import re
import sys
import requests
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_URL = "https://devheartsmart.pcgcid.org/api/v2/freeze-2025-05-06/query_tools/preview/?page=1&records_per_page=38306"

# If you *really* want static cookies, set HEARTSMART_COOKIE_HEADER in your env,
# or provide a raw header dump file to this script (see usage below).
DEFAULT_COOKIES: Dict[str, str] = {}

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Referer": "https://devheartsmart.pcgcid.org/freeze-2025-05-06/results",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
}

_COOKIE_PAIR_RE = re.compile(r"^\s*([^=;,\s]+)\s*=\s*(.*?)\s*$")

def _load_dotenv(dotenv_path: str = ".env") -> None:
    """
    Minimal .env loader (no external deps).
    - Supports: KEY=value or KEY="value" or KEY='value'
    - Ignores blank lines and lines starting with '#'
    - Does not overwrite existing environment variables
    """
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if not k or k in os.environ:
                    continue
                if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
                    v = v[1:-1]
                os.environ[k] = v
    except FileNotFoundError:
        return


def _parse_cookie_header(cookie_header_value: str) -> Dict[str, str]:
    """
    Parses an HTTP Cookie header string into a dict.

    Example:
      "a=1; b=two; c=three" -> {"a": "1", "b": "two", "c": "three"}
    """
    out: Dict[str, str] = {}
    for part in cookie_header_value.split(";"):
        part = part.strip()
        if not part:
            continue
        m = _COOKIE_PAIR_RE.match(part)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        out[k] = v
    return out


def _parse_set_cookie(set_cookie_value: str) -> Optional[Tuple[str, str]]:
    """
    Parses the first 'name=value' pair from a Set-Cookie header value.
    """
    first = (set_cookie_value or "").split(";", 1)[0].strip()
    if not first or "=" not in first:
        return None
    k, v = first.split("=", 1)
    k = k.strip()
    v = v.strip()
    if not k:
        return None
    return k, v


def parse_header_dump(text: str) -> Dict[str, Any]:
    """
    Parses a Chrome/DevTools-style raw header dump like:

      set-cookie
      AWSALB=...; Path=/
      :scheme
      https
      :authority
      example.com
      :path
      /api/...
      cookie
      a=1; b=2

    Returns:
      {
        "request_headers": { ... },  # lowercased keys
        "request_cookies": { ... },  # parsed from Cookie header
        "set_cookies": { ... },      # parsed from Set-Cookie headers
        "url": "https://example.com/api/..."
      }
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    pairs: List[Tuple[str, str]] = []

    i = 0
    while i < len(lines):
        k = lines[i]
        v = lines[i + 1] if i + 1 < len(lines) else ""
        pairs.append((k, v))
        i += 2

    request_headers: Dict[str, str] = {}
    set_cookies: Dict[str, str] = {}

    for k, v in pairs:
        lk = k.lower()
        if lk == "set-cookie":
            parsed = _parse_set_cookie(v)
            if parsed:
                ck, cv = parsed
                set_cookies[ck] = cv
            continue
        request_headers[lk] = v

    request_cookies: Dict[str, str] = {}
    if "cookie" in request_headers:
        request_cookies = _parse_cookie_header(request_headers["cookie"])

    url = ""
    scheme = request_headers.get(":scheme", "").strip()
    authority = request_headers.get(":authority", "").strip()
    path = request_headers.get(":path", "").strip()
    if scheme and authority and path:
        url = f"{scheme}://{authority}{path}"

    return {
        "request_headers": request_headers,
        "request_cookies": request_cookies,
        "set_cookies": set_cookies,
        "url": url,
    }


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def response_to_concept_pairs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts:
      payload["extended_table_def"]["fields"] (column metadata)
      payload["data"] (rows as lists)
    into:
      rows_as_objects: List[Dict[concept_name, value]]
    """
    fields = payload.get("extended_table_def", {}).get("fields", [])
    rows = payload.get("data", [])

    if not fields or not rows:
        return {
            "meta": {
                "record_count": payload.get("record_count"),
                "subject_count": payload.get("subject_count"),
                "paginator": payload.get("paginator"),
            },
            "fields": fields,
            "rows_as_objects": [],
        }

    keys: List[str] = []
    for f in fields:
        key = f.get("concept_name") or f.get("label") or f.get("entry_id")
        keys.append(key)

    rows_as_objects: List[Dict[str, Any]] = []
    for row in rows:
        obj = {keys[i]: row[i] for i in range(min(len(keys), len(row)))}
        rows_as_objects.append(obj)

    return {
        "meta": {
            "record_count": payload.get("record_count"),
            "subject_count": payload.get("subject_count"),
            "paginator": payload.get("paginator"),
            "warnings": payload.get("warnings", []),
            "errors": payload.get("errors", []),
        },
        "rows_as_objects": rows_as_objects,
    }


def main() -> None:
    """
    Usage:
      python hi.py
        - Uses DEFAULT_URL and cookies from HEARTSMART_COOKIE_HEADER (if set)

      python hi.py /path/to/header_dump.txt
        - Extracts URL + cookies from the dump (recommended)

    Env:
      HEARTSMART_COOKIE_HEADER="a=1; b=2; sessionid=..."
      HEARTSMART_URL="https://..."
    """
    _load_dotenv()
    session = requests.Session()

    raw_dump_path = sys.argv[1] if len(sys.argv) > 1 else None

    url = os.getenv("HEARTSMART_URL", "").strip() or DEFAULT_URL
    cookies: Dict[str, str] = dict(DEFAULT_COOKIES)

    if raw_dump_path:
        parsed = parse_header_dump(_read_text_file(raw_dump_path))
        if parsed.get("url"):
            url = parsed["url"]
        if parsed.get("request_cookies"):
            cookies = parsed["request_cookies"]
    else:
        cookie_header = os.getenv("HEARTSMART_COOKIE_HEADER", "").strip()
        if cookie_header:
            cookies = _parse_cookie_header(cookie_header)

    # Important: avoid hard-coding auth cookies into source files.
    r = session.get(url, headers=HEADERS, cookies=cookies, timeout=60)
    r.raise_for_status()

    payload = r.json()

    transformed = response_to_concept_pairs(payload)

    out_path = "preview_concept_pairs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    print(f"Rows saved: {len(transformed.get('rows_as_objects', []))}")


if __name__ == "__main__":
    main()
