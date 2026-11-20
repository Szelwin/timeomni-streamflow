import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parse import parse_forecast  # noqa: E402

REPRO = Path(__file__).resolve().parents[2] / "TimeOmni-repro"


def test_valid():
    v, r = parse_forecast("<think>x</think>\n<answer>\n[1, 2.5, 3, 4, 5, 6, 7]\n</answer>")
    assert r == "ok" and v == [1.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_failures():
    assert parse_forecast(None)[1] == "empty"
    assert parse_forecast("<answer>[1,2,3,4,5,6]</answer>")[1] == "length_6"
    assert parse_forecast("[1,2,3,4,5,6,7]")[1] == "no_answer_list"        # no tags
    assert parse_forecast("<answer>1,2,3,4,5,6,7</answer>")[1] == "unparseable_list"
    assert parse_forecast("<answer>[a,b,c,d,e,f,g]</answer>")[1] == "unparseable_list"
    assert parse_forecast("<answer>[1,2,3,4,5,6,'x']</answer>")[1] == "non_numeric"
    assert parse_forecast("<answer>[1,2,3,4,5,6,7] and [1,2]</answer>")[1] == "ok"   # first list wins
    assert parse_forecast("<answer>[-1,2,3,4,5,6,7]</answer>")[1] == "ok"        # negatives pass, as in original


def test_agrees_with_authors_scorer_on_repro_outputs():
    """Validity must match TimeOmni-1/eval/get_score.py on the 400 reproduction generations."""
    files = sorted(REPRO.glob("results/*_results.json"))
    assert files, "run the reproduction first"
    n = 0
    for f in files:
        for item in json.load(open(f, encoding="utf-8"))[1:]:
            gt = json.loads(item["response"]) if isinstance(item["response"], str) else item["response"]
            v, r = parse_forecast(item["pred_rat"], horizon=len(gt))
            assert (r == "ok") == item["is_valid"], (f.name, item["question_id"], r)
            if r == "ok":
                mae = sum(abs(a - b) for a, b in zip(v, gt)) / len(gt)
                assert abs(mae - item["reward"]) < 1e-6
            n += 1
    assert n == 400
