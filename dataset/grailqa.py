import json
import os

from utils.base import read_json

GRAILQA_ORIGIN_DIR = "/home/hdd/depcache/dataset/GrailQA"
GRAILQA_DIR = "/home/hdd/depcache/dataset/GrailQA/process_data"


def get_grailqa_data(split="train"):
    assert split in ["train", "dev", "test_public"]

    input_file = os.path.join(GRAILQA_ORIGIN_DIR, f"grailqa_v1.0_{split}.json")

    assert os.path.exists(input_file), f"输入文件不存在: {input_file}"

    origin_data = read_json(input_file)

    questions = []
    answers = []
    entities = []

    for data in origin_data:
        answer_list = []
        entity_list = []

        if split != "test_public":

            for answer in data["answer"]:
                if "entity_name" in answer:
                    answer_list.append(answer["entity_name"])
                else:
                    answer_list.append(answer["answer_argument"])

            for item in data["graph_query"]["nodes"] + data["graph_query"]["edges"]:
                entity_list.append(item["friendly_name"])

            answer_list = list(set(answer_list))
            entity_list = list(set(entity_list))

            answers.append(answer_list)
            entities.append(entity_list)

        else:
            # test_public没有答案和graph_query
            answer_list_new = []
            ent_list = []

        questions.append(data["question"])

    return {"questions": questions, "answers": answers, "entities": entities}


if __name__ == "__main__":

    for split in ["train", "dev", "test_public"]:
        data = get_grailqa_data(split)
        print(f"{split} 问题数量: {len(data['questions']):,}")
