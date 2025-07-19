import json
import librosa
import os
from tqdm import tqdm

# === CONFIG ===
INPUT_JSONL = "./finetune_codes/demo_data/audio_understanding/data_with_semantic_codes.jsonl"  # or data_with_sematic_codes.jsonl
OUTPUT_JSONL = "./finetune_codes/demo_data/audio_understanding/data_with_semantic_codes_filtered.jsonl"
MAX_DURATION_SECONDS = 30  # Max audio length allowed
VERBOSE = True  # Print skipped reasons

def get_audio_duration(filepath):
    try:
        duration = librosa.get_duration(filename=filepath)
        return duration
    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] Couldn't load audio file: {filepath}. Reason: {e}")
        return None

def filter_long_audio(jsonl_path, output_path):
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        kept, skipped = 0, 0
        for line in tqdm(fin, desc="Filtering entries"):
            try:
                entry = json.loads(line)
                audio_paths = [
                    msg["content"]
                    for msg in entry["conversation"]
                    if msg["message_type"] == "audio"
                ]
                too_long = False
                for path in audio_paths:
                    if not os.path.exists(path):
                        if VERBOSE:
                            print(f"[SKIP] File missing: {path}")
                        too_long = True
                        break
                    duration = get_audio_duration(path)
                    if duration is None or duration > MAX_DURATION_SECONDS:
                        if VERBOSE:
                            print(f"[SKIP] Too long ({duration:.2f} sec): {path}")
                        too_long = True
                        break
                if too_long:
                    skipped += 1
                    continue

                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept += 1
            except Exception as e:
                if VERBOSE:
                    print(f"[ERROR] Failed to process line: {e}")
                skipped += 1

        print(f"\n✅ Done. Kept: {kept}, Skipped: {skipped}")

# === Run ===
filter_long_audio(INPUT_JSONL, OUTPUT_JSONL)