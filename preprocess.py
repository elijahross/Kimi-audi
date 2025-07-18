import os
import json
from datasets import Dataset
from tqdm import tqdm

# Root directory containing dataset1/, dataset2/, dataset3/
ROOT_DIR = "./assets"
DATASETS = ["dataset2"]
OUTPUT_JSONL = "finetune_codes/demo_data/audio_understanding/data.jsonl"

# Utility: Check if audio exists and fix paths
def resolve_audio_path(base_path, relative_audio_path):
    filename = os.path.basename(relative_audio_path)
    full_path = os.path.join(base_path, "filtered_audio", filename)
    return full_path if os.path.exists(full_path) else None

# Gather all valid examples from all datasets
valid_entries = []

for dataset_name in DATASETS:
    dataset_path = os.path.join(ROOT_DIR, dataset_name, "filtered_data", "dataset.arrow")
    audio_base_path = os.path.join(ROOT_DIR, dataset_name)
    
    print(f"Processing: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"❌ Skipped: {dataset_path} not found")
        continue

    ds = Dataset.from_file(dataset_path)

    for entry in tqdm(ds, desc=f"Checking audio in {dataset_name}"):
        relative_audio_path = entry["audio_path"]
        audio_abs_path = resolve_audio_path(audio_base_path, relative_audio_path)

        if audio_abs_path:
            valid_entries.append({
                "audio_path": audio_abs_path,
                "transcription": entry["transcription"]
            })

print(f"✅ Total valid entries: {len(valid_entries)}")

# Write to Kimi-Audio format
print(f"💾 Writing to {OUTPUT_JSONL}...")
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for ex in valid_entries:
        json_obj = {
            "task_type": "understanding",
            "conversation": [
                {
                    "role": "user",
                    "message_type": "text",
                    "content": "Bitte schreibe die gesprochenen Inhalte in Text um."
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": ex["audio_path"],
                    "speaker_id": ex["speaker_id"]
                },
                {
                    "role": "assistant",
                    "message_type": "text",
                    "content": ex["transcription"]
                }
            ]
        }
        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print("✅ Done!")