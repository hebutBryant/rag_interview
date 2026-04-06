import json
import os
import re
from pathlib import Path

from tqdm import tqdm

METAQA_KB_DIR = "/home/hdd/depcache/dataset/MetaQA/kb.txt"
METAQA_ORIGIN_DIR = "/home/hdd/depcache/dataset/MetaQA/original"
METAQA_DIR = "/home/hdd/depcache/dataset/MetaQA/process"


def process_data(metaqa_origin_dir: str, metaqa_dir: str):
    origin_root = Path(metaqa_origin_dir)
    output_root = Path(metaqa_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for hop_dir in sorted(origin_root.iterdir()):
        if not hop_dir.is_dir():
            continue

        print(f"\nProcessing folder: {hop_dir.name}")
        hop_name = hop_dir.name.replace("-", "_")
        output_file = output_root / f"{hop_name}_qa.json"

        if os.path.exists(output_file):
            print(f"{output_file} exists, skip!")
            continue

        all_data = []
        idx = 0
        for txt_name in ["qa_train.txt", "qa_dev.txt", "qa_test.txt"]:
            txt_file = hop_dir / txt_name
            if not txt_file.exists():
                print(f"  Skipping missing file: {txt_file}")
                continue

            print(f"  Processing file: {txt_file.name}")

            with txt_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t")
                    if len(parts) != 2:
                        print(f"  Skipping malformed line: {line}")
                        continue

                    question_raw, answer_raw = parts
                    entities = re.findall(r"\[(.*?)\]", question_raw)
                    question_clean = re.sub(r"[\[\]]", "", question_raw).strip()
                    answers = [a.strip() for a in answer_raw.split("|")]

                    all_data.append(
                        {
                            "id": idx,
                            "question": question_clean,
                            "entities": entities,
                            "answers": answers,
                        }
                    )
                    idx += 1

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"\nProcessed {len(all_data)} samples, saved to: {output_file}")


def get_metaqa_info(hop: str = "1-hop"):
    metaqa_root = Path(METAQA_DIR)
    data = []

    if hop == "all":
        target_files = sorted(metaqa_root.glob("*_qa.json"))
    else:
        hop_name = hop.replace("-", "_")
        target_files = [metaqa_root / f"{hop_name}_qa.json"]

    for file_path in target_files:
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        print(f"Loading: {file_path}")
        with file_path.open("r", encoding="utf-8") as f:
            hop_data = json.load(f)
            data.extend(hop_data)

    print(f"Loaded {len(data)} samples from {len(target_files)} file(s).")
    questions = []
    answers = []
    entities = []

    for instance in data:
        questions.append(instance.get("question", ""))
        answers.append(instance.get("answers", ""))
        entities.append(instance.get("entities", []))

    data_info = {"questions": questions, "answers": answers, "entities": entities}
    return data_info


def get_triplets():
    triplets = []
    print(f"Loading from {METAQA_KB_DIR}:")
    with open(METAQA_KB_DIR, "r") as fp:
        for line in tqdm(fp, desc="Reading KB triples"):
            head, rel, tail = line.strip().split("|")
            triplets.append((head, rel, tail))
    print(f"Loaded {len(triplets)} triplets.")
    return triplets


if __name__ == "__main__":

    process_data(METAQA_ORIGIN_DIR, METAQA_DIR)

    data_1 = get_metaqa_info("1-hop")
    data_2 = get_metaqa_info("2-hop")
    data_3 = get_metaqa_info("3-hop")
    data_all = get_metaqa_info("all")
    data_sets = {"1-hop": data_1, "2-hop": data_2, "3-hop": data_3, "all": data_all}

    for label, data in data_sets.items():
        print(f"=== {label}-Questions ===:")
        print(data["questions"][:5])

        print(f"=== {label}-Answers ===:")
        print(data["answers"][:5])

        print(f"=== {label}-Entities ===:")
        print(data["entities"][:5])
        print()

    triplets = get_triplets()
    print("\n=== Triplets Sample ===")
    print(triplets[:3])
