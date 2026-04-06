import json
import os
from typing import Dict, List
from utils.base import exists
from utils.base import read_json

MULTIHOP_DATAPATH = "/home/hdd/depcache/dataset/MultiHop"
MULTIHOP_KB_DATAPATH = "/home/hdd/depcache/dataset/MultiHop/multihop_triplets.json"


def get_multihop_info(q_type=None) -> Dict:
    data_file = os.path.join(MULTIHOP_DATAPATH, "MultiHopRAG.json")
    assert exists(data_file), f"{data_file} not exist!"

    # dict_keys(['query', 'answer', 'question_type', 'evidence_list', 'id'])
    # question_type {'inference_query', 'temporal_query', 'comparison_query', 'null_query'}
    questions = []
    answers = []
    with open(data_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)

        for data in data_list:
            # data = json.loads(line)
            if q_type and data["question_type"] != q_type:
                continue
            questions.append(data["query"])
            answers.append(data["answer"])

    corpus_file = os.path.join(MULTIHOP_DATAPATH, "corpus.json")
    assert exists(corpus_file), f"{corpus_file} not exist!"
    with open(corpus_file, "r") as file:
        load_data = json.load(file)
    texts = [data["body"] for data in load_data]

    data_info = {
        "texts": texts,
        "questions": questions,
        "answers": answers,
    }

    return data_info

def get_triplets():
    triplets = read_json(MULTIHOP_KB_DATAPATH)
    triplets = list(set(tuple(t) for t in triplets))
    return triplets

if __name__ == "__main__":

    multihop_info = get_multihop_info()
    print(
        f"questions {len(multihop_info['questions'])} ",
        f"answers {len(multihop_info['answers'])} ",
        f"texts {len(multihop_info['texts'])} ",
    )

    multihop_info = get_multihop_info("inference_query")
    print(
        f"questions {len(multihop_info['questions'])} ",
        f"answers {len(multihop_info['answers'])} ",
        f"texts {len(multihop_info['texts'])} ",
    )
