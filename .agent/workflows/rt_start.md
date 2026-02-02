---
description: Start the Red Teaming Pipeline
---

1. Ask the user for the `pdf_path` (Required).
2. Ask the user for `output` (Default: "outputs/dataset.jsonl").
3. Ask the user for `dataset` (Default: "allenai/wildjailbreak").
4. Ask the user for `column` (Default: "vanilla").
5. Ask the user for `max_samples` (Default: None).
6. Ask the user for `max_concurrent` (Default: 10).
7. Ask the user for `extract_only` (Default: False).
8. Activate the venv
9. Run workflow `.agent/workflows/rt_parsing.md`
10. Run workflow `.agent/workflows/rt_extract_strategy.md`
11. Run workflow `.agent/workflows/rt_make_generator.md`
12. Run workflow `.agent/workflows/rt_generating.md`