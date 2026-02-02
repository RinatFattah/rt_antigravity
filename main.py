"""Phase 3: Orchestration - Main pipeline runner.

Orchestrates the complete adversarial dataset generation pipeline.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from src.paper_agent import PaperAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_pipeline(
    pdf_path: str,
    output_path: str = "outputs/dataset.jsonl",
    dataset_name: str = "allenai/wildjailbreak",
    column: str = "vanilla",
    max_samples: Optional[int] = None,
    max_concurrent: int = 10,
    extract_only: bool = False
):
    """Run the complete adversarial dataset generation pipeline.
    
    Args:
        pdf_path: Path to the research paper PDF.
        output_path: Path to save the output JSONL file.
        dataset_name: HuggingFace dataset name.
        column: Column name in the dataset containing vanilla prompts.
        max_samples: Maximum number of samples to generate. None for all.
        max_concurrent: Maximum concurrent API calls.
    """
    try:
        # Phase 1: Strategy Extraction
        logger.info("=" * 60)
        logger.info("PHASE 1: Strategy Extraction (The Brain)")
        logger.info("=" * 60)
        
        paper_agent = PaperAgent()
        strategy = paper_agent.extract_strategy_from_pdf(pdf_path)
        
        logger.info(f"Extracted strategy: {strategy['strategy_name']}")
        logger.info(f"Core principle: {strategy['core_principle']}")
        
        if extract_only:
            logger.info("Extract-only mode enabled. Skipping dataset generation.")
            return
        
        # Phase 2: Dataset Generation
        logger.info("=" * 60)
        logger.info("PHASE 2: Dataset Generation (The Factory)")
        logger.info("=" * 60)
        
        # Lazy import - only needed when not in extract-only mode
        from generator.generator import DatasetGenerator
        
        generator = DatasetGenerator(
            strategy=strategy,
            max_concurrent=max_concurrent
        )
        
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate and save adversarial pairs
        logger.info(f"Generating adversarial dataset...")
        logger.info(f"Output will be saved to: {output_path}")
        if max_samples:
            logger.info(f"Processing up to {max_samples} samples")
        
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            async for pair in generator.generate_adversarial_pairs(
                dataset_name=dataset_name,
                column=column,
                max_samples=max_samples
            ):
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                f.flush()  # Ensure data is written immediately
                count += 1
        
        logger.info("=" * 60)
        logger.info("PHASE 3: Pipeline Complete")
        logger.info("=" * 60)
        logger.info(f"Successfully generated {count} adversarial pairs")
        logger.info(f"Output saved to: {output_path}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)
        raise


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate adversarial datasets from research papers"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the research paper PDF file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dataset.jsonl",
        help="Output path for the generated dataset (default: outputs/dataset.jsonl)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/wildjailbreak",
        help="HuggingFace dataset name (default: allenai/wildjailbreak)"
    )
    parser.add_argument(
        "--column",
        type=str,
        default="vanilla",
        help="Column name containing vanilla prompts (default: vanilla)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate (default: all)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API calls (default: 10)"
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract strategy from PDF, skip dataset generation"
    )
    
    args = parser.parse_args()
    
    # Run the async pipeline
    asyncio.run(run_pipeline(
        pdf_path=args.pdf_path,
        output_path=args.output,
        dataset_name=args.dataset,
        column=args.column,
        max_samples=args.max_samples,
        max_concurrent=args.max_concurrent,
        extract_only=args.extract_only
    ))


if __name__ == "__main__":
    main()





