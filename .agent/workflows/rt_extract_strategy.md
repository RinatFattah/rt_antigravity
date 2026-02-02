---
description: Extract strategy from PDF (Phase 1)
---

1. Ask the user for the `pdf_path` (Required).
2. Ask the user for `output` (Default: "outputs/dataset.jsonl").
3. Ask the user for `dataset` (Default: "allenai/wildjailbreak").
4. Ask the user for `column` (Default: "vanilla").
5. Ask the user for `max_samples` (Default: None).
6. Ask the user for `max_concurrent` (Default: 10).

7. Run command:
   ```bash
   # Construct command with optional flags
   CMD="python main.py \"$pdf_path\" --output \"$output\" --dataset \"$dataset\" --column \"$column\" --max-concurrent \"$max_concurrent\" --extract-only"

   # Only add max-samples if it is set and not "None"
   if [ -n "$max_samples" ] && [ "$max_samples" != "None" ]; then
     CMD="$CMD --max-samples $max_samples"
   fi

   # Execute
   echo "Running: $CMD"
   eval $CMD
   ```
