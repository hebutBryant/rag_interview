import json
import os
import re

BASE_DATA_DIR = "/home/hdd/depcache/dataset/"

WEBQ_ORIGIN_DIR = os.path.join(BASE_DATA_DIR, "WebQuestions")


def targetsToAnswers(target):
    """Convert raw's targetValue lambda-form to a plain list of answer strings."""
    # (list (description "Jazmyn Bieber") (description "Jaxon Bieber"))
    target = re.sub(r"^\(list |\)$", "", target)
    for answers in re.findall(r'\(description (?:"([^"]+?)"|([^)]+?))\) *', target):
        for answer in answers:
            if answer:
                yield answer


def get_webq_data(data_type="train"):
    if data_type == "train":
        data_files = ["webquestions.examples.train.json"]
    elif data_type == "test":
        data_files = ["webquestions.examples.test.json"]
    elif data_type == "all":
        data_files = [
            "webquestions.examples.train.json",
            "webquestions.examples.test.json",
        ]
    else:
        raise ValueError(
            f"不支持的数据类型: {data_type}，请使用 'train', 'test' 或 'all'"
        )

    questions = []
    all_answers = []
    all_entities = []

    for file in data_files:
        path = os.path.join(WEBQ_ORIGIN_DIR, file)
        assert os.path.exists(path), f"文件不存在：{path}，请确保文件路径正确"

        print(f"正在处理：{file}")
        with open(path, encoding="utf-8") as f_in:
            data = json.load(f_in)
            prefix = file.split(".")[0]
            for _, item in enumerate(data):
                question = item["utterance"]
                answers = list(targetsToAnswers(item["targetValue"]))
                entity = item["url"].split("/")[-1].replace("_", " ")
                questions.append(question)
                all_answers.append(answers)
                all_entities.append([entity])

    return {"questions": questions, "answers": all_answers, "entities": all_entities}


if __name__ == "__main__":

    type = "train"
    data = get_webq_data(data_type=type)

    questions = data["questions"]
    answers = data["answers"]
    entities = data["entities"]
    assert len(questions) == len(answers) == len(entities)

    print(f"{type} has {len(questions)} questions")
    print("questions:", questions[:5])
    print("answers:", answers[:5])
    print("entities:", entities[:5])
    print()
