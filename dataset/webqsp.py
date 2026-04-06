import json
import os

WEBQSP_ORIGIN_DIR = "/home/hdd/depcache/dataset/WebQSP/data"


def get_webqsp_data(type="train"):
    questions = []
    answers = []
    entities = []
    target_files = []

    if type == "train":
        target_files = ["WebQSP.train.json"]
    elif type == "test":
        target_files = ["WebQSP.test.json"]
    elif type == "all":
        target_files = ["WebQSP.train.json", "WebQSP.test.json"]
    else:
        raise ValueError("Invalid type. Choose 'train', 'test' or 'all'.")

    for file in target_files:
        path = os.path.join(WEBQSP_ORIGIN_DIR, file)
        assert os.path.exists(path), f"File {path} does not exist."

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for q_obj in data.get("Questions", []):
                question = q_obj["RawQuestion"]
                answer_list = []
                entity_list = []

                for parse in q_obj["Parses"]:
                    # 1. 提取主题实体
                    if parse.get("TopicEntityName") == None:
                        entity_list.append(parse.get("TopicEntityMid", ""))
                    else:
                        entity_list.append(parse["TopicEntityName"])

                    # 2. 提取约束中的实体 - 使用相同逻辑
                    for constraint in parse.get("Constraints", []):
                        if constraint.get("EntityName") == None:
                            entity_list.append(constraint.get("Argument", ""))
                        else:
                            entity_list.append(constraint["EntityName"])

                    # 3. 提取答案
                    for answer in parse.get("Answers", []):
                        if answer.get("EntityName") == None:
                            answer_list.append(answer.get("AnswerArgument", ""))
                        else:
                            answer_list.append(answer["EntityName"])

                questions.append(question)
                answers.append(sorted(set(answer_list)))
                entities.append(sorted(set(entity_list)))

    return questions, answers, entities


if __name__ == "__main__":
    questions, answers, entities = get_webqsp_data(type="all")

    # 保存到JSON文件
    with open("webqsp_data.json", "w", encoding="utf-8") as f:
        data_list = []
        for i, (q, a, e) in enumerate(zip(questions, answers, entities)):
            data_list.append({"question": q, "answers": a, "entities": e})
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print("Examples of WebQSP:")
    for i, (q, a, e) in enumerate(zip(questions[:5], answers[:5], entities[:5])):
        print(f"question {i+1}: {q}")
        print(f"answer {i+1}: {a}")
        print(f"entity {i+1}: {e}")
        print()

    print(
        f"WebQSP has {len(questions):,} questions, "
        f"WebQSP has {len(answers):,} answers, "
        f"WebQSP has {len(entities):,} entities, "
    )
