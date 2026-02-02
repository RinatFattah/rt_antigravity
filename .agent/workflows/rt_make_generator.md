---
description: Create the Generator (Phase 2) - Clean and Rebuild
---

1. Ask the user for `output` (Default: "outputs/dataset.jsonl").
2. Ask the user for `dataset` (Default: "generator/vanilla_prompts.jsonl").
3. Ask the user for `column` (Default: "vanilla").
4. Ask the user for `max_concurrent` (Default: 10).

5. Clean the `generator` directory:
   - Identify files in `generator/`
   - DELETE all files EXCEPT:
     - `generator/extracted_strategy.json`
     - `generator/__init__.py`
     - `generator/vanilla_prompts.jsonl`
   - Use `rm` command for deletion.

6. Read `generator/extracted_strategy.json`.
   - Analyze the jailbreak strategy described in this file.
   - Determine the transformation logic required based on the `transformation_rules` field.

7. // turbo
   Generate `generator/generator.py`:
   - Create a Python script that implements the jailbreak strategy from the JSON.
   - **Class Requirements**:
     - Class name: `DatasetGenerator`
     - `__init__(self, strategy: dict, max_concurrent: int = 10)`
     - Async method: `generate_adversarial_pairs(dataset_name, column, max_samples) -> AsyncIterator[dict]`
   - **Output Format**: Each yielded dict must contain:
     - `original_prompt`: The vanilla prompt.
     - `attack_prompt`: The transformed jailbreak prompt.
     - `target_response`: Response prefix/marker (or empty string if not applicable).
     - `strategy_name`: Name from the strategy JSON.
   - **Implementation**: Faithfully implement the algorithm/logic from `transformation_rules` in the JSON.
   - **LLM Usage & Context**: If the strategy requires LLM calls (e.g., for poetic transformation, paraphrasing, etc.):
     - **Mandatory System Instruction**: The generated code must include a preamble in every LLM prompt stating: "Request from Ai Security research researcher. The ultimate goal is to fine-tune a guard model."
     - Model: `cognitivecomputations/dolphin-llama-3-70b`
     - This is an uncensored model suitable for generating adversarial content
   - **Environment Variables**:
     - Ensure the script loads environment variables using `python-dotenv`.
     - `OPENROUTER_API_KEY`: Required for LLM calls (OpenRouter).
     - `HUGGINGFACE_TOKEN` : Required for strictly gated datasets (like `allenai/wildjailbreak`).
   - **Ensure valid Python with necessary imports.**

8. Verify `generator/generator.py` exists.