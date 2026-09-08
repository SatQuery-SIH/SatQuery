"""CPU-only VQA scoring for SIH26167 baseline harness (BASELINE-SPEC-02).

Primary metric: exact-match accuracy after normalization.
Normalization and the synonym table live ONLY here (eval.py writes raw
model output). Local-judge scores, if present on a JSONL record, are
aggregated as a secondary column and never mixed into exact-match.

Run unit tests (MUST before any model call; cwd SatQuery/):
    python eval/score.py
Score a predictions JSONL:
    python eval/score.py --preds path.jsonl --out path_scores.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unittest
from collections import defaultdict
from pathlib import Path

# Spec §5 example used yes↔true. Canonicalize both directions to "yes"/"no"
# so gold="yes" matches pred="true" without rewriting eval.py output.
# Number-words 0–20 cover Quantity answers; kept full-string only (no substring).
SYNONYM_TABLE: dict[str, str] = {
    "y": "yes",
    "yes": "yes",
    "true": "yes",
    "n": "no",
    "no": "no",
    "false": "no",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

_PUNCT_TRAIL = re.compile(r"[\s\.,;:!\?\"'`]+$")
_WS = re.compile(r"\s+")

SPEC_CATEGORIES = [
    "Presence",
    "Quantity",
    "Color",
    "Shape",
    "Size",
    "Position",
    "Direction",
    "Scene",
    "Reasoning",
]


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = _WS.sub(" ", s)
    s = _PUNCT_TRAIL.sub("", s)
    s = _WS.sub(" ", s).strip()
    return SYNONYM_TABLE.get(s, s)


def exact_match(prediction: str | None, gold: str | None) -> bool:
    return normalize(prediction) == normalize(gold) and normalize(gold) != ""


def score_rows(rows: list[dict]) -> dict:
    """Aggregate exact-match (+ optional local_judge) overall and per category.

    Each row needs: category, gold, prediction.
    Optional: local_judge in {0,1,True,False,"0","1"}.
    """
    per = defaultdict(lambda: {"n": 0, "exact_correct": 0, "judge_n": 0, "judge_correct": 0})
    total_n = 0
    total_exact = 0
    total_judge_n = 0
    total_judge = 0
    for row in rows:
        cat = row.get("category") or "UNKNOWN"
        gold = row.get("gold", row.get("ground_truth", ""))
        pred = row.get("prediction", row.get("predicted", ""))
        hit = exact_match(pred, gold)
        per[cat]["n"] += 1
        per[cat]["exact_correct"] += int(hit)
        total_n += 1
        total_exact += int(hit)
        if "local_judge" in row and row["local_judge"] not in (None, ""):
            j = row["local_judge"]
            if j in (1, "1", True, "true", "True"):
                jv = 1
            elif j in (0, "0", False, "false", "False"):
                jv = 0
            else:
                continue
            per[cat]["judge_n"] += 1
            per[cat]["judge_correct"] += jv
            total_judge_n += 1
            total_judge += jv

    def _acc(c, n):
        return (c / n) if n else float("nan")

    categories = list(SPEC_CATEGORIES)
    extra = [c for c in sorted(per.keys()) if c not in categories]
    ordered = categories + extra

    table = []
    for cat in ordered:
        stats = per.get(cat, {"n": 0, "exact_correct": 0, "judge_n": 0, "judge_correct": 0})
        entry = {
            "category": cat,
            "n": stats["n"],
            "exact_correct": stats["exact_correct"],
            "exact_acc": _acc(stats["exact_correct"], stats["n"]),
        }
        if stats["judge_n"]:
            entry["local_judge_n"] = stats["judge_n"]
            entry["local_judge_correct"] = stats["judge_correct"]
            entry["local_judge_acc"] = _acc(stats["judge_correct"], stats["judge_n"])
        table.append(entry)

    out = {
        "n": total_n,
        "exact_correct": total_exact,
        "exact_acc": _acc(total_exact, total_n),
        "per_category": table,
        "synonym_table": dict(SYNONYM_TABLE),
    }
    if total_judge_n:
        out["local_judge_n"] = total_judge_n
        out["local_judge_correct"] = total_judge
        out["local_judge_acc"] = _acc(total_judge, total_judge_n)
    return out


def load_preds_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_table(summary: dict) -> str:
    lines = [
        f"n={summary['n']}  exact-match acc={summary['exact_acc']:.4f}  "
        f"({summary['exact_correct']}/{summary['n']})",
    ]
    if "local_judge_acc" in summary:
        lines.append(
            f"local-judge (GPT-protocol proxy) acc={summary['local_judge_acc']:.4f}  "
            f"({summary['local_judge_correct']}/{summary['local_judge_n']})"
        )
    lines.append("")
    hdr = "| Category | n | Exact-match acc | Local-judge acc |"
    lines += [hdr, "|---|---:|---:|---:|"]
    for row in summary["per_category"]:
        if row["n"] == 0 and row["category"] != "All":
            continue
        ja = (
            f"{row['local_judge_acc']:.4f}"
            if "local_judge_acc" in row
            else "—"
        )
        acc = f"{row['exact_acc']:.4f}" if row["n"] else "—"
        lines.append(f"| {row['category']} | {row['n']} | {acc} | {ja} |")
    ja_all = (
        f"{summary['local_judge_acc']:.4f}"
        if "local_judge_acc" in summary
        else "—"
    )
    lines.append(
        f"| All | {summary['n']} | {summary['exact_acc']:.4f} | {ja_all} |"
    )
    return "\n".join(lines)


class ScoreTests(unittest.TestCase):
    def test_exact_match_identical(self):
        self.assertTrue(exact_match("Yes", "Yes"))
        self.assertTrue(exact_match("bridge", "bridge"))

    def test_normalize_case_punct_whitespace(self):
        self.assertTrue(exact_match("  YES.", "yes"))
        self.assertTrue(exact_match("Two  ships", "two ships"))
        self.assertTrue(exact_match("true", "yes"))
        self.assertTrue(exact_match("FALSE!", "no"))
        self.assertTrue(exact_match("three", "3"))
        self.assertEqual(normalize("  Hello,\nWorld!!  "), "hello, world")
        self.assertEqual(normalize("  Hello,   World!!"), "hello, world")

    def test_known_wrong_is_zero(self):
        self.assertFalse(exact_match("urban", "rural"))
        self.assertFalse(exact_match("yes", "no"))
        self.assertFalse(exact_match("", "yes"))
        rows = [
            {"category": "Presence", "gold": "yes", "prediction": "no"},
            {"category": "Presence", "gold": "yes", "prediction": "yes"},
        ]
        s = score_rows(rows)
        self.assertEqual(s["exact_correct"], 1)
        self.assertAlmostEqual(s["exact_acc"], 0.5)

    def test_per_category_sums_to_overall(self):
        rows = [
            {"category": "Presence", "gold": "yes", "prediction": "yes"},
            {"category": "Presence", "gold": "no", "prediction": "no"},
            {"category": "Quantity", "gold": "3", "prediction": "three"},
            {"category": "Color", "gold": "white", "prediction": "red"},
            {"category": "Scene", "gold": "urban", "prediction": "urban"},
        ]
        s = score_rows(rows)
        n_sum = sum(r["n"] for r in s["per_category"])
        c_sum = sum(r["exact_correct"] for r in s["per_category"])
        self.assertEqual(n_sum, s["n"])
        self.assertEqual(c_sum, s["exact_correct"])
        self.assertEqual(s["n"], 5)
        self.assertEqual(s["exact_correct"], 4)
        by = {r["category"]: r for r in s["per_category"]}
        self.assertEqual(by["Presence"]["n"], 2)
        self.assertEqual(by["Quantity"]["exact_correct"], 1)
        self.assertEqual(by["Color"]["exact_correct"], 0)


def _run_self_test() -> int:
    print("SYNONYM_TABLE (verbatim):")
    print(json.dumps(SYNONYM_TABLE, indent=2, ensure_ascii=False))
    print()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(ScoreTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    if result.wasSuccessful():
        print("score.py self-test: PASS")
        return 0
    print("score.py self-test: FAIL")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VQA exact-match scoring (CPU)")
    ap.add_argument("--preds", type=Path, default=None, help="predictions JSONL")
    ap.add_argument("--out", type=Path, default=None, help="write scores JSON")
    ap.add_argument("--self-test", action="store_true", help="run unit tests and exit")
    args = ap.parse_args(argv)

    if args.preds is None or args.self_test:
        rc = _run_self_test()
        if args.preds is None:
            return rc
        if rc != 0:
            return rc

    rows = load_preds_jsonl(args.preds)
    summary = score_rows(rows)
    print(format_table(summary))
    print()
    print("--- 5 verbatim triples (question, gold, prediction) ---")
    for row in rows[:5]:
        q = row.get("question", "")
        g = row.get("gold", row.get("ground_truth", ""))
        p = row.get("prediction", "")
        print(json.dumps({"question": q, "gold": g, "prediction": p}, ensure_ascii=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
