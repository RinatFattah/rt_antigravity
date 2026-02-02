#!/usr/bin/env python3
"""
Dataset Generation Script.
Runs the DatasetGenerator to produce adversarial prompts.
"""

import asyncio
import argparse
import json
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator.generator import DatasetGenerator


async def run_generation(
    output: str,
    dataset: str,
    column: str,
    max_samples: int | None,
    max_concurrent: int
):
    """Run the dataset generation."""
    
    # Load the strategy
    strategy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'generator',
        'extracted_strategy.json'
    )
    
    with open(strategy_path, 'r', encoding='utf-8') as f:
        strategy = json.load(f)
    
    generator = DatasetGenerator(
        strategy=strategy,
        max_concurrent=max_concurrent
    )
    
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    
    count = 0
    print(f'Generating to {output}...')
    
    with open(output, 'w', encoding='utf-8') as f:
        async for pair in generator.generate_adversarial_pairs(
            dataset_name=dataset,
            column=column,
            max_samples=max_samples
        ):
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            f.flush()
            count += 1
    
    print(f'Done. Generated {count} samples.')


def main():
    parser = argparse.ArgumentParser(description='Generate adversarial dataset')
    parser.add_argument('--output', type=str, default='outputs/dataset.jsonl',
                        help='Output file path')
    parser.add_argument('--dataset', type=str, default='local',
                        help='Dataset source (defaults to local generator/vanilla_prompts.jsonl)')
    parser.add_argument('--column', type=str, default='vanilla',
                        help='Column name with vanilla prompts (in local jsonl)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to generate')
    parser.add_argument('--max-concurrent', type=int, default=10,
                        help='Maximum concurrent operations')
    
    args = parser.parse_args()
    
    asyncio.run(run_generation(
        output=args.output,
        dataset=args.dataset,
        column=args.column,
        max_samples=args.max_samples,
        max_concurrent=args.max_concurrent
    ))


if __name__ == '__main__':
    main()
