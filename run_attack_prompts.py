import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "openai/gpt-3.5-turbo-0613"
INPUT_FILE = "outputs/dataset.jsonl"
BASE_URL = "https://openrouter.ai/api/v1"

def process_dataset():
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        return

    print(f"Initializing OpenRouter client with model: {MODEL_NAME}")
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=BASE_URL
    )

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    print(f"Reading dataset from {INPUT_FILE}...")
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total_records = len(records)
    print(f"Found {total_records} records.")

    processed_count = 0
    updated = False

    try:
        for i, record in enumerate(records):
            # Process attack prompt
            if not record.get("target_response"):
                attack_prompt = record.get("attack_prompt")
                if attack_prompt:
                    print(f"Processing attack prompt for record {i+1}/{total_records}...")
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": attack_prompt}
                            ],
                        )
                        record["target_response"] = response.choices[0].message.content
                        updated = True
                        processed_count += 1
                    except Exception as e:
                        print(f"Error processing attack record {i+1}: {e}")
            
            # Process vanilla prompt
            if not record.get("vanilla_response"):
                original_prompt = record.get("original_prompt")
                if original_prompt:
                    print(f"Processing vanilla prompt for record {i+1}/{total_records}...")
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": original_prompt}
                            ],
                        )
                        record["vanilla_response"] = response.choices[0].message.content
                        updated = True
                        processed_count += 1
                    except Exception as e:
                        print(f"Error processing vanilla record {i+1}: {e}")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving progress...")

    if updated:
        print(f"Saving updated dataset to {INPUT_FILE}...")
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print("Save complete.")
    else:
        print("No changes were made to the dataset.")

    print(f"Job finished. Processed {processed_count} new prompts.")

if __name__ == "__main__":
    process_dataset()
