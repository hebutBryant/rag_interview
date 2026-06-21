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
================================================================================
【考题 2 ｜ GraphRAG 流水线】

本文件实现一个三级流水线（三个后台线程 + 三个队列）的 GraphRAG：

    run_batch  →  subgraph_q  →  _subgraph_worker   (实体→多跳子图)
                              →  prune_q  →  _pruning_worker     (语义剪枝路径)
                                          →  gen_q  →  _generation_worker (LLM生成+判分)

需要面试者补全的函数（参考实现已给出，正式面试时可挖空成 stub）：
    - _subgraph_worker()    子图抽取：调用 IGraph 的 DFS 多跳检索 + 路径字符串化
    - _pruning_worker()     路径剪枝：调用 Pruning 的语义批量剪枝
    - _generation_worker()  答案生成：累积 context → 批量调 LLM → 判分 → early-stop

这三个函数各自的详细要求写在对应函数的 docstring 里。

依赖关系（面试者需要先理解）：
    IGraph(database/igraph.py)        知识图谱，提供子图抽取 —— 见考题 2-A
    EntitiesDB(database/entitiesdb.py) 实体向量召回（已给出）
    Pruning(utils/pruning.py)          语义剪枝（已给出）
    LLMEnv(utils/remote_llm.py)        LLM 封装（已给出）

运行：
    python -m rag.graphrag_pipeline --backend qwen --dataset rgb --num 10
================================================================================
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
        entity_embedding_model: str = "text-embedding-v4",
        pruning_embedding_model: str = "text-embedding-v4",
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
        """
        【考题 2-B｜子图抽取 worker】（参考实现已给出，面试时可挖空）

        职责：流水线第 1 级。从 subgraph_q 取任务，对任务里每个问题的
              候选实体做多跳子图抽取，把得到的路径塞进 prune_q 交给下一级。

        你需要完成的逻辑：
          1. 阻塞式地从 self.subgraph_q.get() 取出一个 task
             （task 至少含 "qids" 和 "entities_list"，见 run_batch 的投递格式）
          2. 遍历 task["entities_list"]（每个元素是一个问题对应的实体列表）：
               - 调 self.graph_db.subgraph_extraction_to_paths_dfs(entities, self.hop)
                 做 DFS 多跳抽取，拿到 {实体: [[triplet,...], ...]}
               - 再调 self.graph_db.convert_triplet_lists_to_paths(...)
                 把 triplet 列表转成可读路径字符串 {实体: ["A - r -> B ...", ...]}
          3. 把结果写回 task["triplets"]，self.prune_q.put(task)
          4. 必须调用 self.subgraph_q.task_done()，否则 run_batch 里的 join() 永远不返回

        考察点：生产者-消费者队列、多跳图检索的调用方式、task_done 配平。
        """
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
        """
        【考题 2-C｜路径剪枝 worker】（参考实现已给出，面试时可挖空）

        职责：流水线第 2 级。把上一级抽出来的大量路径，用语义相似度筛掉无关项，
              只保留与问题最相关的 top-k 路径作为 context，送入 gen_q。

        你需要完成的逻辑：
          1. self.prune_q.get() 取出 task（含 "qids" 和 "triplets"）
          2. 根据 qids 还原出对应的问题文本：
               queries = [self.all_questions[qid] for qid in task["qids"]]
          3. 调 self.prunner.semantic_pruning_triplets_batch(
                   questions=queries,
                   question_triplets=task["triplets"],
                   topk=self.pruning,
             )
             返回 pruned_results[qidx] = List[List[(triplet_str, score)]]（按实体分组）
          4. 把每个问题的剪枝结果整理成 context（丢掉分数，去掉空实体），
             写回 task["contexts"]，再 self.gen_q.put(task)
          5. self.prune_q.task_done()

        考察点：语义剪枝的批处理接口、嵌套结构(问题→实体→路径)的展开、控制送入 LLM 的上下文规模。
        """
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
        """
        【考题 2-D｜答案生成 worker】（参考实现已给出，面试时可挖空）

        职责：流水线第 3 级。把剪枝后的 context 累积到每个问题上，调 LLM 生成答案，
              判分、记录 ragas 格式结果，并在命中时触发 early-stop。

        你需要完成的逻辑：
          1. self.gen_q.get() 取出 task（含 "qids" 和 "contexts"）
          2. 对每个 qid：
               - 若开了 early_stop 且 self.stop_flags[qid] 为 True，则跳过（已答对）
               - 把本批 context 追加到 self.context_cache[qid]（注意是“累积”，
                 因为 run_batch 把实体拆成两批分别投递，同一问题会多次到达这里）
               - 组装 {"question": ..., "context": full_ctx} 进 data_list
          3. self.llm.prompt_complete_batch(data_list) 批量生成
          4. 写 self.predictions[qid] / self.sample_records[qid]（ragas 字段：
             question / answer / contexts / ground_truth / label / prompt ...）
          5. 若 check_answer 命中，置 self.stop_flags[qid]=True 实现提前停止
          6. self.gen_q.task_done()

        考察点：跨批次上下文累积、提前停止、ragas 结果落盘字段设计、队列收尾。
        """
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