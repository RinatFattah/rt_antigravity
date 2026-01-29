# Red Teaming Pipeline Universal

A modular Python project that automatically generates adversarial datasets based on research papers describing attack strategies.

## Overview

This pipeline consists of three main phases:

1. **Strategy Extraction (The Brain)**: Extracts attack strategies from research papers using Claude 3.5 Sonnet via OpenRouter
2. **Dataset Generation (The Factory)**: Transforms benign prompts into adversarial prompts using the extracted strategies
3. **Orchestration**: Coordinates the entire pipeline with async processing for efficiency

## Features

- 📄 Automatic PDF text extraction
- 🧠 AI-powered strategy extraction from research papers
- 🏭 Parallel adversarial prompt generation
- 🔄 Streaming dataset processing for memory efficiency
- ⚡ Async/await for concurrent API calls
- 🛡️ Graceful error handling

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd /home/rinat/III/rt_pipeline_universal
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add:
# - OPENROUTER_API_KEY: Your OpenRouter API key (required)
# - HUGGINGFACE_TOKEN: Your HuggingFace token (required for gated datasets like allenai/wildjailbreak)
```

**Getting API Keys:**
- OpenRouter: https://openrouter.ai/keys
- HuggingFace: https://huggingface.co/settings/tokens (needed for `allenai/wildjailbreak`)

## Usage

### Basic Usage

```bash
python main.py path/to/research_paper.pdf
```

### Advanced Usage

```bash
python main.py path/to/research_paper.pdf \
    --output outputs/my_dataset.jsonl \
    --dataset allenai/wildjailbreak \
    --column vanilla \
    --max-samples 100 \
    --max-concurrent 10
```

### Arguments

- `pdf_path` (required): Path to the research paper PDF file
- `--output`: Output path for the generated dataset (default: `outputs/dataset.jsonl`)
- `--dataset`: HuggingFace dataset name (default: `allenai/wildjailbreak`)
- `--column`: Column name containing vanilla prompts (default: `vanilla`)
- `--max-samples`: Maximum number of samples to generate (default: all)
- `--max-concurrent`: Maximum concurrent API calls (default: 10)

## Output Format

The generated dataset is saved as a JSONL file with the following structure:

```json
{"original_prompt": "...", "attack_prompt": "...", "strategy_name": "..."}
{"original_prompt": "...", "attack_prompt": "...", "strategy_name": "..."}
```

## Project Structure

```
rt_pipeline_universal/
├── src/
│   ├── __init__.py
│   ├── paper_agent.py      # Phase 1: Strategy extraction
│   └── generator.py         # Phase 2: Dataset generation
├── main.py                  # Phase 3: Orchestration
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Requirements

- Python 3.7+
- OpenRouter API key (get one at https://openrouter.ai/keys)
- HuggingFace token (required for gated datasets like `allenai/wildjailbreak`)
  - Get token: https://huggingface.co/settings/tokens
  - Accept terms: https://huggingface.co/datasets/allenai/wildjailbreak
- Internet connection for HuggingFace datasets and API calls

## Alternative Datasets

If you don't have access to `allenai/wildjailbreak`, you can use other open datasets:

```bash
# Example with a different dataset (adjust column name as needed)
python main.py paper.pdf --dataset another/dataset --column prompt_column
```

Some alternative open datasets to consider:
- Custom JSONL file (create your own with vanilla prompts)
- Other HuggingFace datasets with prompt columns

## Error Handling

The pipeline includes comprehensive error handling for:
- Missing or unreadable PDF files
- Invalid API responses
- Network errors
- Dataset loading issues (including gated dataset authentication)
- Token limit errors (automatic retry with smaller text)

### Common Issues

**"Dataset is a gated dataset" error:**
- Solution: Add `HUGGINGFACE_TOKEN` to your `.env` file
- Accept dataset terms: https://huggingface.co/datasets/allenai/wildjailbreak

**"Insufficient credits" error:**
- Solution: Add credits at https://openrouter.ai/settings/credits
- The code will automatically retry with smaller token limits

## License

This project is provided as-is for research and educational purposes.

