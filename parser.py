import json
import re
from typing import Any, Dict, List, Optional, Union

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_by_path(obj: Dict[str, Any], path: str) -> Any:
    """
    Dot-path lookup. Example: get_by_path(record, "DNA Sample Type")
    Also supports nested paths like "meta.paginator.current_page".
    """
    if not path:
        return None
    cur: Any = obj
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def _norm_str(x: Any) -> str:
    return str(x).strip().lower()


def _canonical_gender_token(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    token = re.sub(r"[^a-z]+", "", x.lower())
    if token in {"m", "male", "males", "man", "men"}:
        return "male"
    if token in {"f", "female", "females", "woman", "women"}:
        return "female"
    return None


def _text_equal(a: Any, b: Any) -> bool:
    ga = _canonical_gender_token(a)
    gb = _canonical_gender_token(b)
    if ga is not None and gb is not None:
        return ga == gb
    if isinstance(a, str) and isinstance(b, str):
        return _norm_str(a) == _norm_str(b)
    return a == b


def coerce_number(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        s_clean = s.replace(",", "")
        try:
            return float(s_clean)
        except ValueError:
            # Age-like strings are common, e.g. "20 years, 196 days".
            m_years = re.search(r"([-+]?\d+(?:\.\d+)?)\s*years?\b", s_clean, flags=re.IGNORECASE)
            if m_years:
                try:
                    return float(m_years.group(1))
                except ValueError:
                    pass
            m_any = re.search(r"[-+]?\d+(?:\.\d+)?", s_clean)
            if m_any:
                try:
                    return float(m_any.group(0))
                except ValueError:
                    return None
            return None
    return None


def match_condition(value: Any, cond: Dict[str, Any]) -> bool:
    """
    cond examples:
      {"field": "Cohort Source", "op": "eq", "value": "Legacy"}
      {"field": "Has ND Protocol Data", "op": "eq", "value": False}
      {"field": "Comments", "op": "isnull"}
      {"field": "Consent Group", "op": "contains", "value": "Biomedical"}
      {"field": "Targeted Resequencing Availability", "op": "eq", "value": True}
      {"field": "Maternal Age", "op": "gte", "value": 30}
      {"field": "Blinded ID", "op": "regex", "value": r"^0000-0011-09"}
    """
    op = cond.get("op", "eq")
    expected = cond.get("value", None)

    if op == "exists":
        return value is not None
    if op == "isnull":
        return value is None

    if op == "eq":
        return _text_equal(value, expected)
    if op == "ne":
        return not _text_equal(value, expected)
    if op == "in":
        if not isinstance(expected, list):
            return False
        if isinstance(value, str):
            return any(_text_equal(value, item) for item in expected)
        return value in expected
    if op == "nin":
        if not isinstance(expected, list):
            return True
        if isinstance(value, str):
            return all(not _text_equal(value, item) for item in expected)
        return value not in expected

    # string-ish operators
    if op in {"contains", "startswith", "endswith", "regex"}:
        if value is None:
            return False
        s = str(value)
        t = "" if expected is None else str(expected)
        s_fold = s.lower()
        t_fold = t.lower()

        if op == "contains":
            return t_fold in s_fold
        if op == "startswith":
            return s_fold.startswith(t_fold)
        if op == "endswith":
            return s_fold.endswith(t_fold)
        if op == "regex":
            try:
                return re.search(t, s, flags=re.IGNORECASE) is not None
            except re.error:
                return False

    # numeric comparisons (best effort)
    if op in {"gt", "gte", "lt", "lte"}:
        a = coerce_number(value)
        b = coerce_number(expected)
        if a is None or b is None:
            return False
        if op == "gt":
            return a > b
        if op == "gte":
            return a >= b
        if op == "lt":
            return a < b
        if op == "lte":
            return a <= b

    raise ValueError(f"Unsupported op: {op}")


def matches(record: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    """
    spec supports:
      {"and": [<spec|cond>, ...]}
      {"or":  [<spec|cond>, ...]}
      {"not": <spec|cond>}
      <cond> where cond has {"field","op","value"} (value optional for exists/isnull)
    """
    if "and" in spec:
        return all(matches(record, s) for s in spec["and"])
    if "or" in spec:
        return any(matches(record, s) for s in spec["or"])
    if "not" in spec:
        return not matches(record, spec["not"])

    # leaf condition
    field = spec["field"]
    value = get_by_path(record, field)
    return match_condition(value, spec)


def filter_rows(data: Dict[str, Any], spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = data.get("rows_as_objects", [])
    if not isinstance(rows, list):
        raise ValueError("Expected data['rows_as_objects'] to be a list")
    return [r for r in rows if isinstance(r, dict) and matches(r, spec)]


def main():
    # Example usage
    data = load_json("input.json")

    spec = {
        "and": [
            {"field": "Cohort Source", "op": "eq", "value": "Legacy"},
            {"field": "Targeted Resequencing Availability", "op": "eq", "value": True},
            {"field": "Consent Group", "op": "contains", "value": "Biomedical Research"},
            {"field": "Gender", "op": "isnull"},
        ]
    }

    out = filter_rows(data, spec)
    print(f"Matched: {len(out)}")

    with open("filtered.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
