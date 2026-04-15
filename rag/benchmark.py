import json
import os
import asyncio
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset
from openai import OpenAI

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig
from langchain_core.outputs import LLMResult, Generation
from langchain_core.prompt_values import PromptValue


INPUT_JSON = "/home/lipz/rag_interview/rag/log/graphrag_rgb_qwen_qwen-plus_2026-04-06 15:05:01.json"
OUTPUT_DIR = "/home/lipz/rag_interview/rag/eval_results"


class QwenLLMEnv:
    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.verbose = verbose

        if not self.api_key:
            raise ValueError(
                "Qwen API key is missing. Please pass api_key or set DASHSCOPE_API_KEY."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def complete(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        return "" if content is None else str(content).strip()


class QwenRagasLLM(BaseRagasLLM):
    def __init__(
        self,
        qwen_env: QwenLLMEnv,
        run_config: Optional[RunConfig] = None,
        cache: Optional[Any] = None,
    ):
        super().__init__(cache=cache)
        self.qwen_env = qwen_env
        self._run_config = run_config or RunConfig()
        self.set_run_config(self._run_config)

    def _prompt_to_text(self, prompt: PromptValue) -> str:
        if hasattr(prompt, "to_string"):
            return prompt.to_string()
        return str(prompt)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
    ) -> LLMResult:
        text_prompt = self._prompt_to_text(prompt)
        generations = []

        for _ in range(n):
            text = self.qwen_env.complete(text_prompt)
            if stop:
                for s in stop:
                    if s and s in text:
                        text = text.split(s)[0]
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
    ) -> LLMResult:
        loop = asyncio.get_event_loop()
        text_prompt = self._prompt_to_text(prompt)

        async def one_call() -> str:
            text = await loop.run_in_executor(None, self.qwen_env.complete, text_prompt)
            if stop:
                for s in stop:
                    if s and s in text:
                        text = text.split(s)[0]
            return text

        texts = []
        for _ in range(n):
            texts.append(await one_call())

        generations = [[Generation(text=t)] for t in texts]
        return LLMResult(generations=generations)

    def is_finished(self, response: LLMResult) -> bool:
        return True


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_contexts(context_field: Any) -> List[str]:
    if context_field is None:
        return []
    if isinstance(context_field, list):
        return [str(x).strip() for x in context_field if str(x).strip()]
    return [str(context_field).strip()] if str(context_field).strip() else []


def convert_log_to_ragas_samples(raw_data: Any) -> List[Dict[str, Any]]:
    """
    你的文件现在已经是 RAGAS 风格：
    question / answer / contexts / ground_truth

    所以这里直接读，不再把 response/context 做旧格式映射。
    """
    if not isinstance(raw_data, list):
        raise ValueError("输入 JSON 顶层必须是 list。")

    samples = []
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        if "question" not in item:
            continue

        sample = {
            "question": str(item.get("question", "")).strip(),
            "answer": str(item.get("answer", "")).strip(),
            "contexts": normalize_contexts(item.get("contexts", [])),
            "ground_truth": str(item.get("ground_truth", "")).strip(),
            "label": item.get("label"),
            "context_ids": item.get("context_ids", []),
            "context_distances": item.get("context_distances", []),
            "prompt": item.get("prompt", ""),
            "accuracy": item.get("accuracy"),
            "retrieve_time": item.get("retrieve_time"),
            "generate_time": item.get("generate_time"),
            "total_time": item.get("total_time"),
            "avg_total_time": item.get("avg_total_time"),
        }

        if sample["question"]:
            samples.append(sample)

    return samples


def build_dataset(samples: List[Dict[str, Any]]) -> Dataset:
    rows = []
    for s in samples:
        rows.append(
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s["ground_truth"],
            }
        )
    return Dataset.from_list(rows)


######################################################################################
# def benchmark_rag(
#     input_json: str,
#     output_dir: str,
#     model: str = "qwen-plus",
#     api_key: Optional[str] = None,
#     base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
# ):
#     pass   这里封装一个benchmark函数
# 让学生实现完整流程：

# # 1. 加载 json
# # 2. 转换为 ragas 格式
# # 3. 构建 dataset
# # 4. 调用 evaluate
# # 5. 输出 summary + csv
#####################################################################################


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    qwen_env = QwenLLMEnv(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.0,
        max_tokens=1024,
        system_prompt="You are a careful evaluator for RAG systems.",
    )
    evaluator_llm = QwenRagasLLM(qwen_env=qwen_env)

    raw_data = load_json(INPUT_JSON)
    samples = convert_log_to_ragas_samples(raw_data)

    if len(samples) == 0:
        raise RuntimeError("没有可评测样本，请检查输入 JSON。")

    dataset = build_dataset(samples)

    print("===== Dataset Preview =====")
    print(dataset)
    print("First sample:")
    print(dataset[0])

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        raise_exceptions=True,
    )

    details_df = result.to_pandas()

    summary = {"n_samples": len(details_df)}
    for col in ["faithfulness", "context_precision", "context_recall"]:
        if col in details_df.columns:
            summary[col] = details_df[col].mean()

    summary_df = pd.DataFrame([summary])
    base_df = pd.DataFrame(samples)
    merged_df = pd.concat(
        [base_df.reset_index(drop=True), details_df.reset_index(drop=True)],
        axis=1,
    )

    summary_csv = os.path.join(OUTPUT_DIR, "graphrag_ragas_summary.csv")
    details_csv = os.path.join(OUTPUT_DIR, "graphrag_ragas_details.csv")

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    merged_df.to_csv(details_csv, index=False, encoding="utf-8-sig")

    print("\n===== RAGAS Summary =====")
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    print(summary_csv)
    print(details_csv)


if __name__ == "__main__":
    main()