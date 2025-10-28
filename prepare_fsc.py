import os
import json
import pandas as pd
from pathlib import Path

# === 配置路径 ===
DATA_DIR = Path("/home/lzx/Data/fluent_speech_commands_dataset")
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 模板映射 ===
# slot → text 模板，例如 “turn on the lights in the kitchen”
INTENT_TEMPLATES = {
    "change language no object": "Change language",
    "change language": "Change language to {object}",
    "activate no location": "Activate the {object}",
    "activate": "Activate the {object} in the {location}.",
    "deactivate no location": "Deactivate the {object}",
    "deactivate": "Deactivate the {object} in the {location}",
    "bring no location": "Bring the {object}",
    "bring": "Bring the {object} from the {location}",
    "decrease no location": "Decrease the {object}",
    "decrease": "Decrease the {object} in the {location}",
    "increase no location": "Increase the {object}",
    "increase": "Increase the {object} in the {location}",

    # "turn_on": "Turn on the {object} in the {location}.",
    # "turn_off": "Turn off the {object} in the {location}.",
    # "open": "Open the {object} in the {location}.",
    # "close": "Close the {object} in the {location}.",
    # "other": "Do the action on the {object} in the {location}.",
}

def build_text(row):
    # 构建问答形式
    trans = row['transcription']
    action = row['action']
    obj = row.get('object', 'none')
    loc = row.get('location', 'none')
    if action == "change language":
        if obj != 'none':
            text = INTENT_TEMPLATES[action].format(object=obj)
        else:
            text = INTENT_TEMPLATES[action + " no object"]
    elif action in ["activate", "deactivate", "bring", "decrease", "increase"]:
        if loc != 'none':
            text = INTENT_TEMPLATES[action].format(object=obj, location=loc)
        else:
            text = INTENT_TEMPLATES[action + " no location"].format(object=obj)
    else:
        print(f"WARNING: unknown row: {row}")
        text = "Unrecognized"
    
    # if action in ["change language", "bring"]:
    #     text = INTENT_TEMPLATES[action].format(object=obj)
    # else:
    #     if loc != 'none':
    #         text = INTENT_TEMPLATES[action].format(object=obj, location=loc)
    #     else:
    #         text = INTENT_TEMPLATES[action].format(object=obj)
    # text = INTENT_TEMPLATES.get(action, "Control the {object} in the {location}.").format(object=obj, location=loc)
    return text, trans

def process_csv(filename, split):
    df = pd.read_csv(filename)
    output_path = OUTPUT_DIR / f"{split}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text, trans = build_text(row)
            example = {
                "audio": str(DATA_DIR / row["path"]),
                "text": text,
                "trans": trans,
                "intent": row["action"]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"{split} set processed: {len(df)} samples → {output_path}")

if __name__ == "__main__":
    process_csv(DATA_DIR / "data" / "train_data.csv", "train")
    process_csv(DATA_DIR / "data" / "valid_data.csv", "valid")
    process_csv(DATA_DIR / "data" / "test_data.csv", "test")