SatQuery eval kit — same serving stack as the demo GUI. Laptop only.

1. Put questions.jsonl + images/ in inbox/ (schema: id, mode=single|change|sar, question, image or image_a+image_b).
2. Named converters (never auto-detect): python -m judge_kit.adapters.generic --from <file> --to inbox/questions.jsonl
   python -m judge_kit.adapters.vrsbench_like --from <file> --to inbox/questions.jsonl
   python -m judge_kit.adapters.rsvqa_like --from <folder> --to inbox/questions.jsonl
   python -m judge_kit.adapters.cdvqa_like --from <file> --to inbox/questions.jsonl
3. From SatQuery/: powershell -File judge_kit/run_eval.ps1
4. Output lands in inbox/: preds.jsonl and MANIFEST.txt (optional traces/ with -WithTraces).
5. Rehearsal: powershell -File judge_kit/run_eval.ps1 -Questions judge_kit/rehearsal/questions.jsonl -Out judge_kit/rehearsal
6. Reuses a healthy frozen-base server on :8080; otherwise starts :8081 after GGUF/mmproj SHA check.
7. Resume is id-keyed on an existing preds.jsonl. Failures stay in the file. Scoring is off unless -ScoreIfGold <gold.jsonl> (exact-match only).
