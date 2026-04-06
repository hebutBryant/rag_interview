import argparse
import os
from tabnanny import verbose
import time
from typing import Any, Dict, List, Optional

from zai import ZhipuAiClient


import os
import time
from typing import Optional

from zai import ZhipuAiClient
from openai import OpenAI


class ZhipuLLMEnv:
    def __init__(
        self,
        model: str = "glm-5",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.model = model
        self.model_name = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

        if api_key is None:
            api_key = os.getenv("ZHIPU_API_KEY")

        if not api_key:
            raise ValueError(
                "ZHIPU_API_KEY not found. Please set environment variable ZHIPU_API_KEY "
                "or pass api_key explicitly."
            )

        self.client = ZhipuAiClient(api_key=api_key)

    def build_messages(self, **kwargs):
        messages = []

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        if self.user_prompt is not None:
            try:
                user_content = self.user_prompt.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"Missing variable {e} in template: {self.user_prompt}")
        else:
            user_content = kwargs.get("question", "")

        messages.append({"role": "user", "content": user_content})
        return messages

    def prompt_complete(self, **kwargs):
        messages = self.build_messages(**kwargs)

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content

            if content is None:
                raise RuntimeError("Zhipu API returned None content.")

            if not isinstance(content, str):
                content = str(content)

            content = content.strip()

            if content == "":
                raise RuntimeError("Zhipu API returned empty content.")

        except Exception as e:
            raise RuntimeError(f"Zhipu API call failed: {e}")

        generate_time = time.time() - start_time
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        return {
            "response": content,
            "generate_time": generate_time,
            "prompt": prompt,
        }
    

class QwenLLMEnv:
    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        一个基于阿里云百炼 DashScope OpenAI 兼容接口的通用 Qwen LLM 封装。

        参数:
            model: 模型名，例如 qwen-plus
            api_key: 阿里云百炼 API Key；不传则默认读环境变量 DASHSCOPE_API_KEY
            base_url: OpenAI 兼容接口地址
            temperature: 采样温度
            max_tokens: 最大生成长度
            system_prompt: 系统提示词模板
            user_prompt: 用户提示词模板，可用 .format(**kwargs) 填充
            verbose: 是否打印调试信息
        """
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.verbose = verbose

        if not self.api_key:
            raise ValueError(
                "Qwen API key is missing. Please pass api_key or set DASHSCOPE_API_KEY."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def build_messages(self, prompt: Optional[str] = None, **kwargs) -> List[Dict[str, str]]:
        """
        构造 messages。
        优先级：
        1. 如果传了 prompt，则直接作为 user content
        2. 否则使用 user_prompt.format(**kwargs)
        """
        messages: List[Dict[str, str]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if prompt is not None:
            user_content = prompt
        elif self.user_prompt is not None:
            user_content = self.user_prompt.format(**kwargs)
        else:
            raise ValueError("Either prompt or user_prompt must be provided.")

        messages.append({"role": "user", "content": user_content})
        return messages

    def complete(self, prompt: str, verbose: bool = False) -> Optional[str]:
        """
        单轮简单调用：直接输入 prompt，返回文本。
        """
        try:
            messages = self.build_messages(prompt=prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                return ""
            return content.strip()
        except Exception as e:
            print(f"Error in Qwen API call: {e}")
            return None

    def prompt_complete(self, **kwargs) -> Dict[str, Any]:
        """
        用模板构造 prompt 并调用，返回:
        {
            "response": ...,
            "prompt": ...,
            "generate_time": ...
        }
        """
        messages = self.build_messages(**kwargs)
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"Qwen API call failed: {e}")

        generate_time = time.time() - start_time
        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        if self.verbose:
            print("=" * 80)
            print(prompt_text)
            print("-" * 80)
            print(content)
            print("=" * 80)

        return {
            "response": content.strip(),
            "prompt": prompt_text,
            "generate_time": generate_time,
        }

    def prompt_complete_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理。注意：
        DashScope 的 OpenAI 兼容 chat 接口本身不是“单请求多样本 batch 推理”。
        所以这里是 Python 层 for-loop 逐条调用，再把结果收集起来。
        """
        results = []
        for item in data_list:
            results.append(self.prompt_complete(**item))
        return results


class LLMEnv:
    def __init__(
        self,
        backend: str = "zhipu",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        verbose: bool = False,
    ):
        """
        一个统一的 LLM 封装，支持:
        - backend="zhipu"
        - backend="qwen"

        参数:
            backend: 后端类型，支持 "zhipu" / "qwen"
            model: 模型名
            api_key: API Key；不传则按 backend 自动读取环境变量
            base_url: 仅 qwen 需要，默认使用 DashScope OpenAI 兼容接口
            system_prompt: system prompt
            user_prompt: user prompt 模板，可用 .format(**kwargs) 填充
            temperature: 温度
            max_tokens: 最大生成长度
            verbose: 是否打印调试信息
        """
        self.backend = backend.lower()
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        if self.backend == "zhipu":
            self.model = model or "glm-5"
            self.model_name = self.model

            if api_key is None:
                api_key = os.getenv("ZHIPU_API_KEY")

            if not api_key:
                raise ValueError(
                    "ZHIPU_API_KEY not found. Please set environment variable "
                    "ZHIPU_API_KEY or pass api_key explicitly."
                )

            self.api_key = api_key
            self.client = ZhipuAiClient(api_key=self.api_key)

        elif self.backend == "qwen":
            self.model = model or "qwen-plus"
            self.model_name = self.model
            self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

            if api_key is None:
                api_key = os.getenv("DASHSCOPE_API_KEY")

            if not api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY not found. Please set environment variable "
                    "DASHSCOPE_API_KEY or pass api_key explicitly."
                )

            self.api_key = api_key
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Supported backends are: 'zhipu', 'qwen'."
            )

    def build_messages(self, prompt: Optional[str] = None, **kwargs) -> List[Dict[str, str]]:
        """
        构造统一格式的 messages。

        优先级:
        1. 如果显式传入 prompt，则直接作为 user content
        2. 否则如果设置了 user_prompt，则执行 user_prompt.format(**kwargs)
        3. 否则回退到 kwargs.get("question", "")
        """
        messages: List[Dict[str, str]] = []

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        if prompt is not None:
            user_content = prompt
        elif self.user_prompt is not None:
            try:
                user_content = self.user_prompt.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"Missing variable {e} in template: {self.user_prompt}")
        else:
            user_content = kwargs.get("question", "")

        messages.append({"role": "user", "content": user_content})
        return messages

    def complete(self, prompt: str, verbose: bool = False) -> Optional[str]:
        """
        单轮简单调用：输入 prompt，返回字符串结果。
        出错时返回 None。
        """
        try:
            messages = self.build_messages(prompt=prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                return ""

            if not isinstance(content, str):
                content = str(content)

            content = content.strip()
            return content

        except Exception as e:
            print(f"Error in {self.backend} API call: {e}")
            return None

    def prompt_complete(self, **kwargs) -> Dict[str, Any]:
        """
        使用模板构造 prompt 并调用，返回:
        {
            "response": ...,
            "prompt": ...,
            "generate_time": ...
        }
        """
        messages = self.build_messages(**kwargs)
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content

            if content is None:
                raise RuntimeError(f"{self.backend} API returned None content.")

            if not isinstance(content, str):
                content = str(content)

            content = content.strip()

            if content == "":
                raise RuntimeError(f"{self.backend} API returned empty content.")

        except Exception as e:
            raise RuntimeError(f"{self.backend} API call failed: {e}")

        generate_time = time.time() - start_time
        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        if self.verbose or verbose:
            print("=" * 80)
            print(prompt_text)
            print("-" * 80)
            print(content)
            print("=" * 80)

        return {
            "response": content,
            "generate_time": generate_time,
            "prompt": prompt_text,
        }

    def prompt_complete_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理。
        这里是 Python 层逐条调用，统一返回结果列表。
        """
        results = []
        for item in data_list:
            results.append(self.prompt_complete(**item))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="qwen",
        choices=["zhipu", "qwen"],
        help="Choose llm backend: zhipu or qwen",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api_key", type=str, default="sk-cf39778dc1b149928037819399497d0a")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.model is None:
        if args.backend == "zhipu":
            args.model = "glm-5"
        elif args.backend == "qwen":
            args.model = "qwen-plus"

    llm = LLMEnv(
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        system_prompt="You are a helpful assistant.",
        user_prompt="Question: {question}\nContext: {context}\nAnswer briefly.",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    test_question = "What is the capital of France?"
    test_context = "France is a country in Europe. Its capital is Paris."

    ret = llm.prompt_complete(
        question=test_question,
        context=test_context,
    )

    print("===== Backend =====")
    print(args.backend)
    print("\n===== Model =====")
    print(args.model)
    print("\n===== Prompt =====")
    print(ret["prompt"])
    print("\n===== Response =====")
    print(ret["response"])
    print("\n===== Generate Time =====")
    print(ret["generate_time"])


if __name__ == "__main__":
    main()