import threading
import queue
import math
import argparse
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from database.entitiesdb import EntitiesDB
from database.igraph import IGraph
from dataset.metaqa import get_metaqa_info, get_triplets
from dataset.rgb import get_rgb_info
from utils.base import (
    checkanswer,
    create_dir,
    get_date_now,
    save_json,
)
from utils.remote_llm import LLMEnv
from utils.logger import Logger
from utils.prompts import QA_SYSTEM, QA_USER
from utils.pruning import Pruning
from utils.timer import Timer
from dataset.rgb import get_triplets as get_rgb_triplets


"""
挖空这个类的这几个函数
_subgraph_worker()
_pruning_worker()
_generation_worker()

或是改为标准接口让同学完成
def graph_retrieve(...):
    # 实体召回 + 子图抽取 + 路径剪枝
    pass

def graph_chat(...):
    # 基于 context 调用 llm
    pass

def benchmark_graphrag(...):
    # 保存 ragas 格式结果
    pass


"""



class GraphRAGPipeline:
    def __init__(
        self,
        graph_db: IGraph,
        dataset: str,
        llm: LLMEnv,
        ent_num: int = 3,
        hop: int = 2,
        pruning: int = 30,
        log_path: str = "log.txt",
        result_json_path: Optional[str] = None,
        ratio: float = 0.2,
        batch_size: int = 8,
        entity_embedding_model: str = "/home/hdd/model/bge-large-en-v1.5",
        pruning_embedding_model: str = "/home/hdd/model/bge-large-en-v1.5",
        embed_batch_size: int = 64,
        timer_skip: int = 3,
        enable_early_stop: bool = True,
        **kwargs: Any,
    ):
        self.llm = llm
        self.ent_num = ent_num
        self.hop = hop
        self.pruning = pruning
        self.extra_config = kwargs
        self.dataset = dataset
        self.timer = Timer(name="GraphRAGPipeline", skip=timer_skip)
        self.logger: Logger = Logger(log_path=log_path)
        self.graph_db = graph_db
        self.ratio = ratio
        self.batch_size = batch_size
        self.enable_early_stop = enable_early_stop
        self.result_json_path = result_json_path

        self.entities_db = EntitiesDB(
            db_name=f"{dataset}_entities",
            entities=graph_db.entities(),
            embed_name=entity_embedding_model,
            overwrite=False,
            batch_size=embed_batch_size,
        )

        self.prunner = Pruning(
            model=pruning_embedding_model,
            batch_size=embed_batch_size,
        )

        self.subgraph_q = queue.Queue(maxsize=8)
        self.prune_q = queue.Queue(maxsize=8)
        self.gen_q = queue.Queue(maxsize=8)

        self.stop_flags: Dict[int, bool] = {}
        self.context_cache: Dict[int, List[Any]] = {}
        self.predictions: Dict[int, str] = {}
        self.sample_records: Dict[int, Dict[str, Any]] = {}
        self._check_answer = None
        self.all_questions: Sequence[str] = []
        self.all_answers: Optional[Sequence[str]] = None

        self._start_workers()

    def _normalize_ground_truth(self, answer: Any) -> str:
        """
        把原始标准答案转成 ragas 更适合的 ground_truth 字符串。
        例如:
            [["January 2 2022", "Jan 2, 2022"]] -> "January 2 2022"
        """
        if answer is None:
            return ""

        if isinstance(answer, list):
            if len(answer) == 0:
                return ""
            first = answer[0]
            if isinstance(first, list):
                return str(first[0]).strip() if len(first) > 0 else ""
            return str(first).strip()

        return str(answer).strip()

    def _normalize_contexts(self, context: Any) -> List[str]:
        """
        把 GraphRAG 的 context 转成 ragas 需要的 list[str]
        """
        if context is None:
            return []

        normalized: List[str] = []
        for item in context:
            if isinstance(item, list):
                normalized.append("\n".join([str(x) for x in item]))
            else:
                normalized.append(str(item))
        return normalized

    def _start_workers(self):
        threading.Thread(target=self._subgraph_worker, daemon=True).start()
        threading.Thread(target=self._pruning_worker, daemon=True).start()
        threading.Thread(target=self._generation_worker, daemon=True).start()

    def _subgraph_worker(self):
        while True:
            task = self.subgraph_q.get()

            entities_list = task["entities_list"]

            with self.timer.timing("subgraph retrieval"):
                question_triplets = []
                for entities in entities_list:
                    reasoning_paths = self.graph_db.subgraph_extraction_to_paths_dfs(
                        entities, self.hop
                    )
                    reasoning_paths = self.graph_db.convert_triplet_lists_to_paths(
                        reasoning_paths
                    )
                    question_triplets.append(reasoning_paths)

            task["triplets"] = question_triplets
            self.prune_q.put(task)
            self.subgraph_q.task_done()

    def _pruning_worker(self):
        while True:
            task = self.prune_q.get()

            qids = task["qids"]
            queries = [self.all_questions[qid] for qid in qids]
            question_triplets = task["triplets"]

            with self.timer.timing("path pruning"):
                pruned_results = self.prunner.semantic_pruning_triplets_batch(
                    questions=queries,
                    question_triplets=question_triplets,
                    topk=self.pruning,
                )

            contexts = []
            for qidx in range(len(queries)):
                question_results = pruned_results[qidx]  # List[List[(triplet, score)]]
                context = []
                for entity_results in question_results:
                    pruned_paths = [triplet for triplet, _ in entity_results]
                    if pruned_paths:
                        context.append(pruned_paths)
                contexts.append(context)

            task["contexts"] = contexts
            self.gen_q.put(task)
            self.prune_q.task_done()

    def _generation_worker(self):
        while True:
            task = self.gen_q.get()

            qids = task["qids"]
            new_contexts = task["contexts"]

            valid_qids = []
            data_list = []

            for i, qid in enumerate(qids):
                if self.enable_early_stop and self.stop_flags.get(qid, False):
                    continue

                old_ctx = self.context_cache.get(qid, [])
                full_ctx = old_ctx + new_contexts[i]
                self.context_cache[qid] = full_ctx

                valid_qids.append(qid)
                data_list.append(
                    {
                        "question": self.all_questions[qid],
                        "context": full_ctx,
                    }
                )

            if not data_list:
                self.gen_q.task_done()
                continue

            results = self.llm.prompt_complete_batch(data_list)

            for qid, ret in zip(valid_qids, results):
                response = ret["response"]
                prompt = ret["prompt"]

                self.predictions[qid] = response
                gt_raw = self.all_answers[qid] if self.all_answers is not None else ""
                normalized_contexts = self._normalize_contexts(self.context_cache[qid])

                label = None
                if self._check_answer is not None and self.all_answers is not None:
                    label = self._check_answer(response, gt_raw)

                self.sample_records[qid] = {
                    "question": self.all_questions[qid],
                    "answer": response,  # ragas: model answer
                    "contexts": normalized_contexts,  # ragas: list[str]
                    "ground_truth": self._normalize_ground_truth(gt_raw),
                    "label": label,
                    "prompt": prompt,
                    "context_size": len(normalized_contexts),
                    "ground_truth_aliases": gt_raw,
                }

                self.logger.log(
                    f"Question_{qid}: {self.all_questions[qid]}",
                    color="yellow",
                )
                self.logger.log(f"Response_{qid}: {response}", color="magenta")
                self.logger.log(f"Answer_{qid}: {gt_raw}", color="green")

                if self._check_answer is not None and self.all_answers is not None:
                    score = all(self._check_answer(response, gt_raw))
                    self.logger.log(f"score_{qid}: {score}", color="green")

                    if (
                        self.enable_early_stop
                        and score
                        and not self.stop_flags.get(qid, False)
                    ):
                        self.stop_flags[qid] = True
                        self.logger.log(
                            f"Early stop Question_{qid} at {len(self.context_cache[qid])} contexts.",
                            color="red",
                        )

            self.gen_q.task_done()

    def run_batch(
        self,
        questions: Sequence[str],
        answers: Optional[Sequence[str]] = None,
        check_answer: Optional[
            Callable[[str, str, Optional[str]], float | int | bool]
        ] = None,
    ) -> Dict[str, Any]:
        with self.timer.timing("total time"):
            self.all_questions = questions
            self.all_answers = answers
            self._check_answer = check_answer

            total = len(questions)
            split_num = max(1, math.ceil(self.ent_num * self.ratio))
            qid = 0

            while qid < total:
                batch_qids = list(range(qid, min(qid + self.batch_size, total)))
                batch_questions = [questions[i] for i in batch_qids]

                with self.timer.timing("query embedding"):
                    similar_entities_list, _ = self.entities_db.search(
                        batch_questions,
                        top_k=self.ent_num,
                    )

                entities_02 = [
                    similar_entities_list[i][:split_num]
                    for i in range(len(batch_qids))
                ]
                task1 = {
                    "qids": batch_qids,
                    "entities_list": entities_02,
                }
                self.subgraph_q.put(task1)

                entities_08 = [
                    similar_entities_list[i][split_num:]
                    for i in range(len(batch_qids))
                ]
                task2 = {
                    "qids": batch_qids,
                    "entities_list": entities_08,
                }
                self.subgraph_q.put(task2)

                qid += self.batch_size

            self.subgraph_q.join()
            self.prune_q.join()
            self.gen_q.join()

            preds = [self.predictions[i] for i in range(total)]

            acc = None
            if check_answer is not None and answers is not None:
                correct = 0
                for i in range(total):
                    correct += float(all(check_answer(preds[i], answers[i])))
                acc = correct / total

        records = [self.sample_records[i] for i in range(total) if i in self.sample_records]

        final_result = {
            "final_accuracy": acc,
            "sample_num": total,
            "max_tokens": self.llm.max_tokens,
            "ent_num": self.ent_num,
            "hop": self.hop,
            "pruning": self.pruning,
        }

        output_data = records + [final_result]

        if self.result_json_path is not None:
            save_json(file_path=self.result_json_path, data=output_data)

        return {
            "questions": list(questions),
            "predictions": preds,
            "accuracy": acc,
            "records": records,
        }


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        type=str,
        default="qwen",
        choices=["zhipu", "qwen"],
        help="Select the inference backend.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or local model path. If not set, backend-specific default will be used.",
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

    parser.add_argument("--num", type=int, default=10, help="Number of questions to run.")
    parser.add_argument("--dataset", type=str, default="rgb")
    parser.add_argument("--ent", type=int, default=10, help="Number of retrieved entities.")
    parser.add_argument("--hop", type=int, default=2, help="Subgraph retrieval hop.")
    parser.add_argument("--pruning", type=int, default=30, help="Top-k paths after pruning.")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    if args.model is None:
        if args.backend == "zhipu":
            args.model = "glm-4.5-air"
        elif args.backend == "qwen":
            args.model = "qwen-plus"
        else:
            raise ValueError(f"Unsupported backend: {args.backend}")

    llm = LLMEnv(
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        system_prompt=QA_SYSTEM,
        user_prompt=QA_USER,
        max_tokens=args.max_tokens,
        temperature=0,
    )

    if args.dataset == "rgb":
        rgb_info = get_rgb_info(file="en")
        questions, answers = rgb_info["questions"], rgb_info["answers"]
        triplets = get_rgb_triplets()
    elif args.dataset.startswith("metaqa"):
        match = re.search(r"(\d+-hop)", args.dataset)
        if match:
            hop = match.group(1)
        else:
            raise ValueError(
                f"Invalid metaqa dataset format: {args.dataset}, expected like 'metaqa_2-hop'"
            )

        metaqa_info = get_metaqa_info(hop=hop)
        questions, answers = metaqa_info["questions"], metaqa_info["answers"]
        triplets = get_triplets()
    else:
        raise NotImplementedError(f"dataset {args.dataset}")

    num = min(args.num, len(questions)) if args.num > 0 else len(questions)
    questions = questions[:num]
    answers = answers[:num]

    graph_db = IGraph(dataset=args.dataset, triplets=triplets)

    create_dir("./log")
    log_path = (
        f"./log/graphrag_{args.dataset}_{args.backend}_"
        f"{llm.model_name.replace('/', '_')}_{get_date_now()}.log"
    )
    result_json_path = (
        f"./log/graphrag_{args.dataset}_{args.backend}_"
        f"{llm.model_name.replace('/', '_')}_{get_date_now()}.json"
    )

    graphrag = GraphRAGPipeline(
        llm=llm,
        dataset=args.dataset,
        graph_db=graph_db,
        ent_num=args.ent,
        hop=args.hop,
        pruning=args.pruning,
        batch_size=args.batch_size,
        log_path=log_path,
        result_json_path=result_json_path,
    )

    result = graphrag.run_batch(questions, answers, checkanswer)

    print("\n===== Final Result =====")
    print(result)
    print(f"\nSaved ragas-format json to: {result_json_path}")