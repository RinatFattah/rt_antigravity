---
description: Run the Generation Phase (Phase 3)
---

1. Ask the user for `output` (Default: "outputs/dataset.jsonl").
2. Ask the user for `dataset` (Default: "allenai/wildjailbreak").
3. Ask the user for `column` (Default: "vanilla").
4. Ask the user for `max_samples` (Default: None).
5. Ask the user for `max_concurrent` (Default: 10).

6. Run command:
   ```bash
   # Build command with arguments
   CMD="python run_generation.py --output \"$output\" --dataset \"$dataset\" --column \"$column\" --max-concurrent \"$max_concurrent\""

   # Add max-samples if set
   if [ -n "$max_samples" ] && [ "$max_samples" != "None" ]; then
     CMD="$CMD --max-samples $max_samples"
   fi

   echo "Running: $CMD"
   eval $CMD
   ```
