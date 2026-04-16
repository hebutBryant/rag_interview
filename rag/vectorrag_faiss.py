import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
from zai import ZhipuAiClient
from utils.remote_llm import LLMEnv
from database.faissdb import FaissDB
from dataset.rgb import get_rgb_info
from utils.base import (
    checkanswer,
    create_dir,
    get_accuracy,
    get_base_dir,
    get_date_now,
    print_text,
    save_json,
)
from utils.prompts import QA_SYSTEM, QA_USER


"""
这个是数据集的构造

TODO:
1. 从 rgb_data 中整理 texts / metadata / ids
2. 初始化 FaissDB
3. 将 metadata 和 ids 挂载回 db
4. 返回 db
"""
# def prepare_faiss_db(...):
#     pass


def prepare_faiss_db(
    rgb_data: Dict,
    persist_dir: str,
    chunk_size: int = 512,
) -> FaissDB:
    """
    用 FaissDB 构建向量库。
    这里保留了和 prepare_chroma_db 类似的输入输出风格。
    """

    texts = []
    metadatas = []
    ids = []

    for idx, text in enumerate(rgb_data["texts"]):
        if isinstance(text, list):
            for i, chunk in enumerate(text):
                texts.append(chunk)
                metadatas.append({"original_id": idx, "chunk_id": i})
                ids.append(f"doc_{idx}_chunk_{i}")
        else:
            texts.append(text)
            metadatas.append({"original_id": idx})
            ids.append(f"doc_{idx}")

    print("*****************",len(ids))
    db = FaissDB(
        db_name="rgb_en_collection",
        texts=texts,
        embed_name="text-embedding-v4",
        overwrite=False,
        batch_size=32,
        device="cuda:1",
        db_dir=persist_dir,
    )

    # 给 db 挂上 metadata 和 ids，便于后续检索时返回和 Chroma 类似的结果结构
    db.metadatas = metadatas
    db.doc_ids = ids

    return db


# def prepare_faiss_db(...):
#     pass
# 这个是完善检索流程


def vectorrag_with_faiss(
    questions: List[str],
    answers: List[str],
    faiss_db: FaissDB,
    log_file: str,
    llm: LLMEnv,
    top_k: int = 3,
):
    all_labels = []
    data = []
    all_time = []

    def normalize_ground_truth(answer):
        """
        把你原来的 answer 格式转成 RAGAS 更适合的 ground_truth 字符串。
        例如:
        [[ "January 2 2022", "Jan 2, 2022", ... ]]
        -> "January 2 2022"
        """
        if answer is None:
            return ""

        if isinstance(answer, list):
            if len(answer) == 0:
                return ""
            first = answer[0]
            if isinstance(first, list):
                return str(first[0]) if len(first) > 0 else ""
            return str(first)

        return str(answer)



# 代码的这一部分注释，
# 对问题进行向量检索
# 整理检索结果并构建上下文
# 调用大语言模型基于上下文生成答案
# 保存每条样本的检索与生成结果，供后续评测使用
###########################################################################
    for i, (question, answer) in enumerate(
        tqdm(zip(questions, answers), total=len(questions))
    ):
        # 步骤1: 检索相关文档
        retrieve_time = -time.time()
        matched_texts, distances, matched_ids = faiss_db.search(
            question, top_k=top_k
        )
        retrieve_time += time.time()

        matched_docs = []
        for rank, (text, distance) in enumerate(
            zip(matched_texts, distances[: len(matched_texts)])
        ):
            doc_id = matched_ids[rank] if rank < len(matched_ids) else rank
            record = {
                "document": text,
                "id": faiss_db.doc_ids[doc_id] if hasattr(faiss_db, "doc_ids") else f"doc_{doc_id}",
                "distance": float(distance),
                "metadata": faiss_db.metadatas[doc_id] if hasattr(faiss_db, "metadatas") else {},
            }
            matched_docs.append(record)

        # 步骤2: 构建上下文
        context = "\n".join([doc["document"] for doc in matched_docs])

        # 步骤3: 构建提示词并生成答案
        outside_generate_time = -time.time()
        ret = llm.prompt_complete(question=question, context=context)
        outside_generate_time += time.time()
############################################################################################





        response = ret["response"]
        generate_time = ret["generate_time"]
        prompt = ret["prompt"]
        print(f"{outside_generate_time=}, {generate_time=}")

        total_time = retrieve_time + generate_time
        all_time.append(total_time)

        # 步骤4: 评估答案
        label = checkanswer(response, answer)
        all_labels.append(label)
        accuracy = get_accuracy(all_labels)

        print_text(f"question_{i}: {question}", color="blue")
        print_text(f"response_{i}: {response}", color="green")
        print_text(f"answer_{i}: {answer}", color="green")
        print_text(f"label_{i}: {label}", color="red")
        print_text(f"accuracy_{i}: {accuracy}", color="green")
        print_text(f"total_time_{i}: {total_time:.4f}", color="yellow")
        print_text(f"avg_total_time_{i}: {np.average(all_time):.4f}", color="yellow")

        item = {
            "question": question,
            "answer": response,  # RAGAS里的answer = 模型回答
            "contexts": [doc["document"] for doc in matched_docs],  # RAGAS里的contexts = list[str]
            "ground_truth": normalize_ground_truth(answer),  # 标准答案压平成字符串

            "label": label,
            "context_ids": [doc["id"] for doc in matched_docs],
            "context_distances": [doc["distance"] for doc in matched_docs],
            "prompt": prompt,
            "accuracy": accuracy,
            "retrieve_time": retrieve_time,
            "generate_time": generate_time,
            "total_time": total_time,
            "avg_total_time": np.average(all_time),

            # 可选：保留原始答案，方便你自己核对
            "ground_truth_aliases": answer,
        }

        data.append(item)
        save_json(file_path=log_file, data=data)

    final_result = {
        "final_accuracy": get_accuracy(all_labels),
        "average_time": np.average(all_time[3:]) if len(all_time) > 3 else np.average(all_time),
        "sample_num": len(questions),
        "max_tokens": llm.max_tokens,
        "top_k": top_k,
    }

    print_text(f"最终结果: {final_result}\n", color="yellow")
    data.append(final_result)
    save_json(file_path=log_file, data=data)

    return final_result["final_accuracy"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        type=str,
        default="qwen",
        choices=["zhipu", "qwen"],
        help="LLM backend: zhipu or qwen",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. If not set, backend-specific default will be used.",
    )

    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument(
        "--faiss_persist_dir",
        type=str,
        default="./faiss_rgb_data",
        help="Faiss 数据库持久化路径",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-cf39778dc1b149928037819399497d0a",
        help="API key for selected backend. If not provided, read from environment variables.",
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Optional base_url, mainly for qwen compatible API.",
    )

    args = parser.parse_args()

    # backend 默认模型
    if args.model is None:
        if args.backend == "zhipu":
            args.model = "glm-4.5-air"
        elif args.backend == "qwen":
            args.model = "qwen-plus"

    llm = LLMEnv(
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        system_prompt=QA_SYSTEM,
        user_prompt=QA_USER,
        temperature=0.0,
    )

    try:
        ret = llm.prompt_complete(question="Hello", context="")
        print("#######################", ret)
    except Exception as e:
        print(f"[FATAL] Stop pipeline due to LLM failure: {e}")
        exit(1)

    rgb_info = get_rgb_info("en")
    print(f"加载完成，共{len(rgb_info['questions'])}个问题")

    base_dir = get_base_dir()
    faiss_persist_dir = os.path.join(base_dir, "faiss_rgb_data")
    os.makedirs(faiss_persist_dir, exist_ok=True)

    faiss_db = prepare_faiss_db(
        rgb_data=rgb_info,
        persist_dir=faiss_persist_dir,
    )

    print("开始处理问题...")

    create_dir("./log")
    safe_time = get_date_now().replace(":", "-")
    log_file = (
        f"./log/vector_rag_faiss_{llm.model_name}_rgb_en_"
        f"{args.backend}_top{args.top_k}_{safe_time}.json"
    )

    questions = rgb_info["questions"][:10]
    answers = rgb_info["answers"][:10]

    final_accuracy = vectorrag_with_faiss(
        questions=questions,
        answers=answers,
        faiss_db=faiss_db,
        log_file=log_file,
        llm=llm,
        top_k=args.top_k,
    )

    print(f"最终准确率: {final_accuracy:.4f}")