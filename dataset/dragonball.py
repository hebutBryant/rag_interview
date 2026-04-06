# import argparse
import json
import os

# from llama_index.core import Document
from utils.base import exists, read_jsonl, read_json

DRAGONBALL_DATAPATH = "/home/hdd/depcache/dataset/DragonBall"
DRAGONBALL_KB_DATAPATH = "/home/hdd/depcache/dataset/DragonBall/dragonball_triplets.json"

def get_dragonball_info(language=None, query_type=None):
    data_path = os.path.join(DRAGONBALL_DATAPATH, "dragonball_queries.jsonl")
    assert exists(data_path), f"{data_path} not exist!"

    # query_type {'Factual Question', 'Multi-hop Reasoning Question', 'Summary Question','Summarization Question',
    #             'Irrelevant Unsolvable Question', 'Multi-document Information Integration Question',
    #             'Multi-document Comparison Question', 'Multi-document Time Sequence Question'}

    questions = []
    answers = []
    question_types = []
    languages = []
    domains = []

    data = read_jsonl(data_path)

    for ins in data:
        domain = ins["domain"]
        language_ins = ins["language"]
        query_type_ins = ins["query"]["query_type"]

        if language is not None and language != language_ins:
            continue

        # print(query_type, query_type_ins)
        if query_type is not None and query_type != query_type_ins:
            continue

        question = ins["query"]["content"]
        answer = ins["ground_truth"]["content"]

        assert isinstance(question, str)
        assert isinstance(answer, str)

        questions.append(question)
        answers.append(answer)
        languages.append(language_ins)
        domains.append(domain)
        question_types.append(query_type_ins)

    data_info = {
        "questions": questions,
        "answers": answers,
        "languages": languages,
        "domains": domains,
        "question_types": question_types,
    }

    return data_info

def get_triplets():
    triplets = read_json(DRAGONBALL_KB_DATAPATH)
    triplets = list(set(tuple(t) for t in triplets))
    return triplets

if __name__ == "__main__":
    # Summary Question: 415 questions
    # Summarization Question: 118 questions
    
    dragaonball_info = get_dragonball_info("en", "Factual Question")
    print(f"questions: {len(dragaonball_info['questions'])}")
    print(f"answers: {len(dragaonball_info['answers'])}")
    print(f"contexts: {len(dragaonball_info['contexts'])}")
