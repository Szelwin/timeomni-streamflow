"""Answer extraction and validity, verbatim from TimeOmni-1/eval/utils.py
(compute_reward, branch task_type == "event_aware_forecasting"):

  1. text between <answer> and </answer>            (extract_answer)
  2. first [...] inside it                           (extract_list)
  3. ast.literal_eval -> list/tuple (scalar -> [x])
  4. len(pred) == len(ground truth)  (7 here)
  5. every element float()-convertible

Anything else is invalid. Negative values pass (as in the original); they are only
flagged so their count can be reported.
"""
import ast
import re

HORIZON = 7


def extract_answer(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1) if match else None


def extract_list(text):
    if text is None:
        return None
    match = re.search(r"\[(.*?)\]", text, re.DOTALL)
    return "[" + match.group(1) + "]" if match else None


def parse_forecast(text, horizon: int = HORIZON):
    """Returns (values | None, reason). reason == "ok" iff values is a list of `horizon` floats."""
    if text is None or text == "" or text == "null":
        return None, "empty"
    try:
        content = extract_list(extract_answer(text))
        pred = ast.literal_eval(content)
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
    except Exception:  # noqa: BLE001  (same blanket except as the original)
        return None, "no_answer_list" if extract_answer(text) is None else "unparseable_list"
    if len(pred) != horizon:
        return None, f"length_{len(pred)}"
    try:
        values = [float(x) for x in pred]
    except Exception:  # noqa: BLE001
        return None, "non_numeric"
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        return None, "non_finite"
    return values, "ok"
