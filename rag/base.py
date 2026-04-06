from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

from utils.remote_llm import LLMEnv
from utils.logger import Logger
from utils.timer import Timer

class RAG(ABC):

    def __init__(
        self,
        llm: LLMEnv,
        top_k: int = 3,
        logger: Logger = None,
        timer: Timer = None,
        monitor: ResourceMonitor = None,
        **kwargs: Any,
    ) -> None:
        self.llm: LLMEnv = llm
        self.top_k = top_k
        self.logger = logger
        self.timer = timer
        self.monitor = monitor
        self.extra_config = kwargs

    # @abstractmethod
    # def preprocess(self, question: str) -> Any:
    #     raise NotImplementedError

    # @abstractmethod
    # def postprocess(
    #     self,
    #     question: str,
    #     contexts: Any,
    #     raw_answer: str,
    # ) -> str:
    #     raise NotImplementedError

    # def build_prompt(self, question: str, contexts: Any) -> str:
    #     if isinstance(contexts, (list, tuple)):
    #         ctx_text = "\n".join(map(str, contexts))
    #     else:
    #         ctx_text = str(contexts)

    #     prompt = (
    #         "You are a helpful assistant. Use the given context to answer the question.\n\n"
    #         f"Context:\n{ctx_text}\n\n"
    #         f"Question: {question}\n"
    #         "Answer:"
    #     )
    #     return prompt

    @abstractmethod
    def retrieve(self, query: str) -> Any:
        raise NotImplementedError
    
    def retrieve_batch(self, queries: List[str]) -> Any:
        raise NotImplementedError

    def generate(self, question: str) -> str:

        # query = self.preprocess(question)
        contexts = self.retrieve(question)
        # prompt = self.build_prompt(question, contexts)
        answer = self.llm.prompt_complete(question=question, context=contexts)
        # final_answer = self.postprocess(question, contexts, raw_answer)
        return answer

    def generate_batch(self, questions: list) -> List[str]:
        contexts_list = self.retrieve_batch(questions)

        with self.timer.timing("process batch prompts"):
            data_list = []
            for q, context in zip(questions, contexts_list):
                # context_str = "\n".join(
                #     [path for entity_paths in context for path in entity_paths]
                # ) if context else "No relevant knowledge found."

                data_list.append({
                    "question": q,
                    "context": context,
                })
        if self.monitor:
            self.monitor.set_tag("llm generation")
        answers = self.llm.prompt_complete_batch(data_list)
        if self.monitor:
            self.monitor.set_tag("other processing")
        return answers
    
    def run(
        self,
        questions: Sequence[str],
        answers: Optional[Sequence[str]] = None,
        check_answer: Optional[
            Callable[[str, str, Optional[str]], float | int | bool]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation on a dataset.

        Args:
            questions: Input question list.
            answers: Ground-truth answers (optional).
            check_answer: A function that evaluates (question, pred, answer)
                        and returns bool/int/float as score.
        """

        # answers and check_answer must appear together
        if (answers is None) != (check_answer is None):
            raise ValueError(
                "answers and check_answer must be both None or both not None"
            )

        # check length consistency
        if answers is not None:
            assert len(questions) == len(answers)

        preds: List[str] = []
        correct = 0.0
        total = len(questions)

        for i, q in enumerate(questions):
            pred = self.generate(q)
            preds.append(pred)

            if check_answer is not None:
                gt = answers[i]
                score = check_answer(q, pred, gt)
                correct += float(score) if not isinstance(score, bool) else float(score)

        acc = correct / total if total > 0 else 0.0

        return {
            "questions": list(questions),
            "predictions": preds,
            "accuracy": acc,
        }
