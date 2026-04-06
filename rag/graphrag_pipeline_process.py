import os
import math
import re
import argparse
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Sequence
from database.entitiesdb import EntitiesDB
from database.igraph import IGraph
from dataset.rgb import get_rgb_info
from dataset.rgb import get_triplets as get_rgb_triplets
from dataset.metaqa import get_metaqa_info, get_triplets
from utils.llm import LLMEnv
from utils.pruning import Pruning
from utils.prompts import QA_SYSTEM, QA_USER
from dataset.freebase import get_triplets as get_freebase_triplets
from dataset.metaqa import get_metaqa_info
from dataset.metaqa import get_triplets as get_metaqa_triplets
from dataset.rgb import get_rgb_info
from dataset.rgb import get_triplets as get_rgb_triplets
from dataset.multihop import get_multihop_info
from dataset.multihop import get_triplets as get_multihop_triplets
from dataset.dragonball import get_dragonball_info
from dataset.dragonball import get_triplets as get_dragonball_triplets
from experiment.cpu_gpu.monitor import ResourceMonitor
from utils.timer import Timer
from utils.logger import Logger
from utils.base import (
    checkanswer,
    create_dir,
    get_accuracy,
    get_date_now,
    print_text,
    read_json,
    save_json,
)
from multiprocessing import Lock
import signal
import os


class EarlyStopQuestions:
    def __init__(self):
        self.stopped = set()
        self.lock = Lock()

    def stop(self, qid):
        with self.lock:
            self.stopped.add(qid)

    def is_stopped(self, qid):
        with self.lock:
            stopped = qid in self.stopped
            return stopped
        
class GraphRAGPipelineProcess:

    def __init__(
        self,
        graph_db: IGraph,
        dataset: str,
        llm_config,
        ent_num: int = 3,
        hop: int = 2,
        pruning: int = 30,
        log_path: str = "log.txt",
        ratio=0.3,
        batch_size=8,
        entity_embedding_model="BAAI/bge-large-en-v1.5",
        pruning_embedding_model="BAAI/bge-large-en-v1.5",
        embed_batch_size=64,
        timer_skip=3,
        monitor: ResourceMonitor = None,
        enable_early_stop: bool = True,
        check_answer: Optional[
            Callable[[str, str, Optional[str]], float | int | bool]
        ] = None,
        **kwargs,
    ):

        mp.set_start_method("spawn", force=True)

        self.ent_num = ent_num
        self.hop = hop
        self.pruning = pruning
        self.dataset = dataset
        self.ratio = ratio
        self.batch_size = batch_size
        self.enable_early_stop = enable_early_stop
        self.monitor = monitor
        self.extra_config = kwargs

        self.timer = Timer(name="GraphRAGPipelineProcess", skip=timer_skip)
        self.logger = Logger(log_path=log_path)
        self.early_stop_questions = EarlyStopQuestions() if enable_early_stop else None

        self.entities_db = EntitiesDB(
            db_name=f"{dataset}_entities",
            entities=graph_db.entities(),
            embed_name=entity_embedding_model,
            overwrite=False,
            batch_size=embed_batch_size,
        )

        self.subgraph_q = mp.Queue()
        self.prune_q = mp.Queue()
        self.gen_q = mp.Queue()
        self.result_q = mp.Queue()


        self.p1 = mp.Process(
            target=self.subgraph_worker,
            args=(graph_db, 
                self.subgraph_q, 
                self.prune_q, hop, 
                self.early_stop_questions
            ),
        )

        self.p2 = mp.Process(
            target=self.pruning_worker,
            args=(
                self.prune_q,
                self.gen_q,
                pruning,
                pruning_embedding_model,
                embed_batch_size,
                self.early_stop_questions,
            ),
        )

        self.p3 = mp.Process(
            target=self.generation_worker,
            args=(
                self.gen_q,
                self.result_q,
                llm_config,
                enable_early_stop,
                check_answer,
                self.early_stop_questions,
            ),
        )

        self.p1.start()
        self.p2.start()
        self.p3.start()

    # =========================================================
    # Subgraph Worker (stateless)
    # =========================================================
    @staticmethod
    def subgraph_worker(graph_db, subgraph_q, prune_q, hop, early_stop_questions):
        timer = Timer(name="SubgraphWorker")

        while True:
            task = subgraph_q.get()
            if task is None:
                prune_q.put({
                    "_type": "timer_summary",
                    "stage": "subgraph retrieval",
                    "summary": timer.summary(),
                })
                prune_q.put(None)
                break

            entities_list = task["entities_list"]
            qids = task["qids"]

            with timer.timing("subgraph retrieval"):
                question_triplets = []

                for entities, qid in zip(entities_list, qids):
                    if early_stop_questions.is_stopped(qid):
                        question_triplets.append([])
                        continue

                    paths = graph_db.subgraph_extraction_to_paths_dfs(entities, hop)
                    paths = graph_db.convert_triplet_lists_to_paths(paths)
                    question_triplets.append(paths)

            task["triplets"] = question_triplets
            prune_q.put(task)

    # =========================================================
    # Pruning Worker (stateless)
    # =========================================================
    @staticmethod
    def pruning_worker(
        prune_q,
        gen_q,
        pruning,
        pruning_embedding_model,
        embed_batch_size,
        early_stop_questions,
    ):
        timer = Timer(name="PruningWorker")

        prunner = Pruning(
            model=pruning_embedding_model,
            batch_size=embed_batch_size,
            # device="cuda:1",
        )
        gen_q.put({
            "_type": "worker_ready",
            "stage": "pruning",
        })


        while True:
            task = prune_q.get()
            if task is None:
                gen_q.put({
                    "_type": "timer_summary",
                    "stage": "path pruning",
                    "summary": timer.summary(),
                })
                gen_q.put(None)
                break

            if task.get("_type") == "timer_summary":
                gen_q.put(task)
                continue

            questions = task["questions"]
            triplets = task["triplets"]
            valid_indices = []
            valid_questions = []
            valid_triplets = []

            for i, qid in enumerate(task["qids"]):
                if early_stop_questions.is_stopped(qid):
                    print_text(f"Skip pruning for early stopped Question_{qid}.", color="red")
                    continue
                valid_indices.append(i)
                valid_questions.append(questions[i])
                valid_triplets.append(triplets[i])
                

            with timer.timing("path pruning"):
                pruned_results = prunner.semantic_pruning_triplets_batch(
                    questions=valid_questions,
                    question_triplets=valid_triplets,
                    topk=pruning,
                )

            valid_contexts = []
            for qid in range(len(valid_questions)):
                question_results = pruned_results[qid]
                context = []
                for entity_results in question_results:
                    pruned_paths = [triplet for triplet, _ in entity_results]
                    if pruned_paths:
                        context.append(pruned_paths)
                valid_contexts.append(context)

            contexts = [None] * len(task["qids"])
            for local_idx, global_idx in enumerate(valid_indices):
                contexts[global_idx] = valid_contexts[local_idx]

            task["contexts"] = contexts
            gen_q.put(task)

    # =========================================================
    # Generation Worker (STATEFUL)
    # =========================================================
    @staticmethod
    def generation_worker(
        gen_q,
        result_q,
        llm_config,
        enable_early_stop,
        check_answer,
        early_stop_questions,
    ):

        llm = LLMEnv(
            backend=llm_config["backend"],
            model=llm_config["model"],
            system_prompt=QA_SYSTEM,
            user_prompt=QA_USER,
            max_tokens=llm_config["max_tokens"],
            temperature=0,
        )
        result_q.put({
            "_type": "worker_ready",
            "stage": "generation",
        })

        context_cache = {}

        while True:
            task = gen_q.get()
            if task is None:
                result_q.put({
                    "_type": "timer_summary",
                    "stage": "generation",
                    "summary": llm.timer.summary(),
                })
                result_q.put(None)
                break

            if task.get("_type") in ("worker_ready", "timer_summary"):
                result_q.put(task)
                continue

            qids = task["qids"]
            questions = task["questions"]
            answers = task["answers"]
            new_contexts = task["contexts"]

            valid_qids = []
            batch = []

            for i, qid in enumerate(qids):

                if enable_early_stop and early_stop_questions.is_stopped(qid):
                    continue

                old_ctx = context_cache.get(qid, [])
                full_ctx = old_ctx + new_contexts[i]
                context_cache[qid] = full_ctx

                valid_qids.append(qid)
                batch.append({
                    "question": questions[i],
                    "context": full_ctx,
                })

            if not batch:
                continue

            results = llm.prompt_complete_batch(batch)

            for qid, ret in zip(valid_qids, results):
                response = ret["response"]

                if check_answer is not None:
                    gt = answers[qids.index(qid)]
                    score = all(check_answer(response, gt))
                    if enable_early_stop and score and not early_stop_questions.is_stopped(qid):
                        early_stop_questions.stop(qid)

                result ={
                    "type": "prediction",
                    "qid": qid,
                    "question": questions[qids.index(qid)],
                    "answer": answers[qids.index(qid)],
                    "response": response,
                    "score": score if check_answer is not None else None,
                    "num_contexts": len(context_cache[qid]),
                    "raw_result": ret,
                }
                
                result_q.put(result)


    # =========================================================
    # Run batch (主进程)
    # =========================================================
    def run_batch(self, questions, answers):
        ready = set()
        required = {"pruning", "generation"}

        while ready != required:
            msg = self.result_q.get()
            if isinstance(msg, dict) and msg.get("_type") == "worker_ready":
                ready.add(msg["stage"])

        with self.timer.timing("total time"):
            if self.monitor:
                self.monitor.current_tag = "beginning"
            assert len(questions) == len(answers), "questions and answers must have the same length"

            total = len(questions)
            split_num = max(1, math.ceil(self.ent_num * self.ratio))
            qid = 0

            while qid < total:

                batch_qids = list(range(qid, min(qid + self.batch_size, total)))
                batch_questions = [questions[i] for i in batch_qids]
                batch_answers = [answers[i] for i in batch_qids]

                similar_entities_list, _ = self.entities_db.search(
                    batch_questions,
                    top_k=self.ent_num,
                )

                ent_02 = [similar_entities_list[i][:split_num] for i in range(len(batch_qids))]
                ent_08 = [similar_entities_list[i][split_num:] for i in range(len(batch_qids))]

                self.subgraph_q.put({
                    "qids": batch_qids,
                    "questions": batch_questions,
                    "answers": batch_answers,
                    "entities_list": ent_02,
                })

                self.subgraph_q.put({
                    "qids": batch_qids,
                    "questions": batch_questions,
                    "answers": batch_answers,
                    "entities_list": ent_08,
                })

                qid += self.batch_size

            # ---- stop signal
            self.subgraph_q.put(None)

            preds = {}
            stage_timers = {}
            acc = None
            correct = 0

            while True:
                item = self.result_q.get()
                if item is None:
                    break

                if isinstance(item, dict) and item.get("_type") == "timer_summary":
                    stage_timers[item["stage"]] = item["summary"]
                else:
                    qid = item["qid"]
                    question = item["question"]
                    answer = item["answer"]
                    response = item["response"]
                    score = item["score"]
                    num_contexts = item["num_contexts"]
                    raw_result = item["raw_result"]
                    preds[qid] = response
                    if score or num_contexts == self.ent_num:
                        correct += float(score)

                    self.logger.log(f"Question_{qid}: {question}", color="yellow")
                    self.logger.log(f"Response_{qid}: {response}", color="magenta")
                    self.logger.log(f"Ansewr_{qid}: {answer}", color="green")
                    self.logger.log(f"score_{qid}: {score}", color="green")
                    self.logger.log(f"NumContexts_{qid}: {num_contexts}", color="cyan")
                    for k, v in raw_result.items():
                        if k in {"response", "prompt"}:
                            continue
                        self.logger.log(f"{k}_{qid}: {v}")
                    if score and num_contexts < self.ent_num:
                        self.logger.log(
                            f"Early stop Question_{qid} at {num_contexts} contexts.",
                            color="red",
                        )
            acc = correct / total

        for stage, summary in stage_timers.items():
            self.logger.log(f"\n{summary}")

        self.logger.log("\n" + self.timer.summary())
        
        return {
            "questions": list(questions),
            "predictions": preds,
            "accuracy": acc,
        }

    @staticmethod
    def _kill_process_tree(pid):
        import psutil
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            return

        print(f"Killing process tree for PID {pid} (Found {len(children)} children)...")

        # send SIGTERM
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # wait for processes to terminate gracefully
        _, alive = psutil.wait_procs(children, timeout=1)
        
        # send SIGKILL
        for child in alive:
            try:
                print(f"Force killing stubborn child: {child.pid}")
                child.kill()
            except psutil.NoSuchProcess:
                pass

        # kill the parent itself
        try:
            parent.kill()
            parent.wait(1)
        except psutil.NoSuchProcess:
            pass

    def close(self):
        print("Closing pipeline...")

        if self.monitor:
            self.monitor.stop()
            print("Monitor closed.")

        target_processes = [self.p1, self.p2, self.p3]
        for p in target_processes:
            if not p.is_alive():
                continue
                
            print(f"Stopping worker: {p.name} (PID: {p.pid})")
            
            if p == self.p3: 
                print("Detected SGLang worker, applying tree kill...")
                self._kill_process_tree(p.pid)
            else:
                p.terminate()
                p.join(timeout=0.5)
                if p.is_alive():
                    try:
                        os.kill(p.pid, signal.SIGKILL)
                    except:
                        pass
        print("All processes closed.")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="/home/hdd/model/Qwen2.5-7B-Instruct/"
    )
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="rgb")
    parser.add_argument("--ent", type=int, default=10, help="number of entities")
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--pruning", type=int, default=30)

    parser.add_argument("--max_tokens", type=int, default=200)

    parser.add_argument(
        "--backend",
        type=str,
        default="sglang",
        help="Select the inference backend.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    print(args)

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

    llm_cfg = dict(
        backend=args.backend,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    graph_db = IGraph(dataset=args.dataset, triplets=triplets)

    pipeline = GraphRAGPipelineProcess(
        graph_db=graph_db,
        dataset=args.dataset,
        llm_config=llm_cfg,
        ent_num=args.ent,
        hop=args.hop,
        pruning=args.pruning,
        batch_size=8,
        check_answer=checkanswer
    )

    preds = pipeline.run_batch(questions, answers)

    pipeline.close()

