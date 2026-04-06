import argparse
import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import ollama
import torch
from ollama import ChatResponse, Client
from ollama import chat as ollama_chat
from ollama import generate

# from ollama import Ollama
from openai import OpenAI
from sglang.lang.api import Engine
from sglang.srt.server_args import ServerArgs
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
    set_seed,
)

from utils.base import exists, get_base_dir, print_text, read_json, read_yaml
from utils.prompts import QA_SYSTEM, QA_USER
from utils.timer import Timer


class BaseLLMEnv(ABC):
    def __init__(
        self,
        model: str,
        system_prompt=None,
        user_prompt=None,
        verbose: str = False,
        timer_name="LLM",
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.verbose = verbose
        self.timer = Timer(name=timer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    @abstractmethod
    def complete(self, prompt):
        pass

    @abstractmethod
    def prompt_complete(self, **kwargs):
        # 1. build prompt
        # 2. generate response by call complete
        pass

    def build_prompt(self, **kwargs):
        messages = []

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        try:
            messages.append(
                {"role": "user", "content": self.user_prompt.format(**kwargs)}
            )

        except KeyError as e:
            raise ValueError(f"Missing variable {e} in template: {self.user_prompt}")

        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n\n".join([msg["content"] for msg in messages])

        return prompt


class OpenAIEnv(BaseLLMEnv):

    def __init__(
        self,
        model="gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature=0,
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, prompt, verbose=False):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=self.temperature,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in LLM API call: {e}")
            return None


class OllamaEnv(BaseLLMEnv):

    def __init__(
        self,
        model="llama3.1:8b",
        system_prompt=None,
        user_prompt=None,
        timeout=300,
        temperature=0,
        base_url="http://localhost:11434",
        max_tokens=20,
        verbose=False,
        # system=None
    ):
        super().__init__(
            model,
            system_prompt,
            user_prompt,
            verbose=verbose,
            timer_name=f"Ollama ({model})",
        )

        # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-request-with-options
        self.options = {"num_predict": max_tokens, "temperature": temperature}

        self.model = model

        self.llm = Client(
            host=base_url,
            timeout=timeout,
        )

    def prompt_complete(self, **kwargs):

        try:
            user_prompt = self.user_prompt.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} in template: {self.user_prompt}")

        output = self.llm.generate(
            self.model, user_prompt, options=self.options, system=self.system_prompt
        )

        response = output.response

        if self.verbose:
            print_text(f"{self.system_prompt}\n{user_prompt}\n", color="yellow")
            print_text(f"{response}", color="green")

        info = self.parse_response_info(output)

        return info | {
            "response": response,
            "prompt": f"{self.system_prompt}\n{user_prompt}\n",
        }

    def complete(self, prompt):
        """
        Generate a completion without using any predefined Ollama template.

        This function explicitly disables Ollama's built-in instruction template
        by setting `template="{{ .Prompt }}"`, meaning only the raw prompt text
        is sent to the model.

        When using instruction-tuned models (e.g., `*-instruct` models),
        you need to manually include special instructions (e.g., `<|start_header_id|>system<|end_header_id|>`) or role directives
        within the `prompt` to guide the model's behaviour.
        """

        template = "{{ .Prompt }}" if self.system_prompt is not None else ""

        output = self.llm.generate(
            model=self.model,
            prompt=prompt,
            options=self.options,
            template=template,
        )

        response = output.response

        if self.verbose:
            print_text(f"{prompt}\n", color="yellow")
            print_text(f"{response}", color="green")

        info = self.parse_response_info(output)

        return info | {
            "response": response,
            "prompt": prompt,
        }

    def parse_response_info(self, response):
        # total_duration: time spent generating the response
        total_time = response.total_duration / 1e9

        # load_duration: time spent in nanoseconds loading the model
        load_time = response.load_duration / 1e9

        # (prefill): prompt_eval_duration: time spent in nanoseconds evaluating the prompt
        prefill_time = response.prompt_eval_duration / 1e9

        # (generation): eval_duration: time in nanoseconds spent generating the response
        decode_time = response.eval_duration / 1e9

        # prompt_eval_count: number of tokens in the prompt
        prompt_len = (
            response.prompt_eval_count if "prompt_eval_count" in response else 0
        )

        # eval_count: number of tokens in the response
        generate_len = response.eval_count

        # if self.verbose:
        #     print_text(
        #         (
        #             f"generate_time {total_time:.3f}s, "
        #             f"load_time {load_time:.3f}s, "
        #             f"prefill_time {prefill_time:.3f}s, "
        #             f"decode_time {decode_time:.3f}s, "
        #             f"prefill_length {prompt_len}, "
        #             f"decode_length {generate_len}"
        #         ),
        #         color="red",
        #     )

        return {
            "prefill_length": prompt_len,
            "decode_length": generate_len,
            "generate_time": total_time,
            "load_time": load_time,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
        }


class HuggingfaceEnv(BaseLLMEnv):

    def __init__(
        self,
        model,
        system_prompt=None,
        user_prompt=None,
        max_tokens=20,
        temperature=0,
        verbose=False,
    ):

        super().__init__(
            model,
            system_prompt,
            user_prompt,
            verbose=verbose,
            timer_name=f"Huggingface ({model})",
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        # torch.cuda.set_device(device)
        self.llm = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model,
                dtype=torch.bfloat16,
                # device_map=f"cuda:{device}",
                attn_implementation="sdpa",
                # attn_implementation="flash_attention_2",
                device_map="auto",
            )
            # .cuda()
            .eval()
        )

        # self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        #     pretrained_model_name_or_path=model, use_fast=False
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model
            # , use_fast=False
        )

    def complete(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(data=[input_ids], dtype=torch.int64).cuda()
        input_length = input_ids.size(-1)

        self.tokenizer.pad_token_id = (
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )

        generation_config = GenerationConfig(
            do_sample=False,
            temperature=self.temperature,
            repetition_penalty=1.0,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=self.max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_strings=[
                "<|im_end|>",
                "<|eot_id|>",
                "<|end_of_text|>",
                "<|endoftext|>",
            ],
            # eos_token_id=999999,
            # stop_strings=None,
        )

        with self.timer.timing("generate"):
            outputs = self.llm.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                use_cache=True,
                # eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                tokenizer=self.tokenizer,
                attention_mask=torch.ones_like(input_ids).cuda(),
            )

            output_token = outputs[0][input_length:]
            response = self.tokenizer.decode(token_ids=output_token.tolist())

        if self.verbose:
            print_text(f"{prompt}\n", color="yellow")
            print_text(f"{response}\n", color="green")

        return {
            "response": response,
            "prefill_length": input_length,
            "decode_length": output_token.size(-1),
            # "generate_time": generate_time,
            "prompt": prompt,
        }
    def complete_batch(self, prompts: List[str]) -> List[Dict]:
        with self.timer.timing("tokenizer"):
            encoded = self.tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,          # 自动 pad 到 batch 最长
                return_tensors="pt",
            ).to(self.llm.device)

        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask

        # 计算每个 prompt 的实际长度（排除 padding）
        input_lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1).cpu().tolist()

        # 2. 设置 pad_token（与原 complete 一致）
        self.tokenizer.pad_token_id = (
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )

        # 3. GenerationConfig 与原 complete 完全一致
        generation_config = GenerationConfig(
            do_sample=False,
            temperature=self.temperature,
            repetition_penalty=1.0,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=self.max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_strings=[
                "<|im_end|>",
                "<|eot_id|>",
                "<|end_of_text|>",
                "<|endoftext|>",
            ],
        )

        # 4. 生成
        with self.timer.timing("generate"):
            outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                tokenizer=self.tokenizer,
            )

        # 5. 解码并提取每个样本的生成部分
        results = []
        for i, (prompt, input_len) in enumerate(zip(prompts, input_lengths)):
            # 输出序列中，从 input_len 开始是新生成的 token
            generated_tokens = outputs[i][input_len:]

            response = self.tokenizer.decode(
                generated_tokens.tolist(),
                skip_special_tokens=True  # 一般建议跳过特殊 token，更干净
            ).strip()

            if self.verbose:
                print_text(f"{prompt}\n", color="yellow")
                print_text(f"{response}\n", color="green")

            results.append({
                "response": response,
                "prefill_length": input_len,
                "decode_length": generated_tokens.size(0),
                "prompt": prompt,
            })

        return results
    
    def prompt_complete(self, **kwargs):
        prompt = self.build_prompt(**kwargs)
        return self.complete(prompt)
    def prompt_complete_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict]:
        """
        data_list: [{"question": "...", "context": "..."}, 
                    {"question": "...", "context": "..."},
                    ...]
        """
        prompts = []
        for i, data in enumerate(data_list):
            try:
                prompt = self.build_prompt(**data)
            except KeyError as e:
                raise ValueError(f"Missing variable {e} in template for item {i}: {data}")
            except Exception as e:
                raise ValueError(f"Error building prompt for item {i}: {e}")
            prompts.append(prompt)

        return self.complete_batch(prompts)

class SGLangEnv(BaseLLMEnv):
    def __init__(
        self,
        model,
        system_prompt=None,
        user_prompt=None,
        max_tokens=20,
        temperature=0.0,
        port=31000,
        host="127.0.0.1",
        disable_overlap_schedule=True,
        mem_fraction_static=0.9,
        log_level="info",
        gpu_id=2,
        tp_size=1,
        verbose=False,
    ):
        super().__init__(
            model,
            system_prompt,
            user_prompt,
            verbose=verbose,
            timer_name=f"SGLang ({model})",
        )

        self.max_tokens = max_tokens
        self.temperature = temperature

        server_args = ServerArgs(
            model_path=model,
            port=port,
            host=host,
            device="cuda",
            tp_size=tp_size,
            base_gpu_id=gpu_id,
            mem_fraction_static=mem_fraction_static,
            # chunked_prefill_size=chunked_prefill_size,
            # context_length=20000,  # 输入长度会超过默认值，需要设置
            log_level=log_level,
            disable_overlap_schedule=disable_overlap_schedule,
        )

        self.sampling_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "stop": ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
        }

        self.llm = Engine(server_args=server_args)

    def complete(self, prompt):
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        # return self.llm.complete(prompt, verbose=verbose, return_info=return_info)
        with self.timer.timing("generate"):
            outputs = self.llm.generate(prompt, self.sampling_params)

        response = outputs.get("text", "").strip()
        prompt_tokens = outputs["meta_info"]["prompt_tokens"]
        completion_tokens = outputs["meta_info"]["completion_tokens"]
        cached_tokens = outputs["meta_info"]["cached_tokens"]
        cache_hit_rate = cached_tokens / prompt_tokens if prompt_tokens else 0.0

        if self.verbose:
            print_text(f"{prompt}", color="yellow")
            print_text(f"{response}", color="green")

        return {
            "response": response,
            "prefill_length": prompt_tokens,
            "decode_length": completion_tokens,
            "cached_tokens": cached_tokens,
            "cache_hit_rate": cache_hit_rate,
            # "generate_time": generate_time,
            "prompt": prompt,
        }
    def complete_batch(self, prompts: List[str]) -> List[Dict]:
        with self.timer.timing("asyncio"):
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        with self.timer.timing("generate"):
            outputs_list = self.llm.generate(prompts, self.sampling_params)

        with self.timer.timing("process batch outputs"):
            # 处理结果
            results = []
            for i, outputs in enumerate(outputs_list):
                prompt = prompts[i]
                response = outputs.get("text", "").strip()

                prompt_tokens = outputs["meta_info"]["prompt_tokens"]
                completion_tokens = outputs["meta_info"]["completion_tokens"]
                cached_tokens = outputs["meta_info"]["cached_tokens"]
                cache_hit_rate = cached_tokens / prompt_tokens if prompt_tokens else 0.0

                if self.verbose:
                    print_text(f"{prompt}", color="yellow")
                    print_text(f"{response}", color="green")

                results.append({
                    "response": response,
                    "prefill_length": prompt_tokens,
                    "decode_length": completion_tokens,
                    "cached_tokens": cached_tokens,
                    "cache_hit_rate": cache_hit_rate,
                    "prompt": prompt,
                })

        return results
    def prompt_complete(self, **kwargs):
        prompt = self.build_prompt(**kwargs)
        return self.complete(prompt)
    def prompt_complete_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict]:
        """
        data_list: [{"question": "...", "context": "..."}, 
                    {"question": "...", "context": "..."},
                    ...]
        """
        with self.timer.timing("build batch prompts"):
            prompts = []
            for i, data in enumerate(data_list):
                try:
                    prompt = self.build_prompt(**data)
                except KeyError as e:
                    raise ValueError(f"Missing variable {e} in template for item {i}: {data}")
                except Exception as e:
                    raise ValueError(f"Error building prompt for item {i}: {e}")
                prompts.append(prompt)

        return self.complete_batch(prompts)

class LLMEnv:

    def __init__(
        self,
        backend: Literal[
            "openai", "deepseek", "ollama", "huggingface", "vllm", "sglang"
        ] = "ollama",
        model="llama3.1:8b",
        system_prompt=None,
        user_prompt=None,
        # openai
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # ollama
        timeout=300,
        max_tokens=20,
        # huggingface
        # device="cuda:0",
        # vllm
        memory_utilization=0.8,
        enable_prefix_caching=True,
        # logging_level="WARN",
        temperature=0,
        # top_p=0.95,
        # sglang
        # tp_size=1,
        port=31000,
        host="127.0.0.1",
        disable_overlap_schedule=True,
        # mem_fraction_static=0.8,
        log_level="debug",
        ollama_host="http://localhost:11434",
        verbose=False,
        # gpu_id=0,
    ):
        self.backend = backend
        self.max_tokens = max_tokens
        self.model = model

        if "/" in model:
            idx = -2 if "checkpoint" in model else -1
            self.model_name = model.split("/")[idx]
        else:
            self.model_name = model

        if backend == "ollama":
            self.llm = OllamaEnv(
                model=model,
                timeout=timeout,
                base_url=ollama_host,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                verbose=verbose,
            )

        elif backend == "huggingface":
            self.llm = HuggingfaceEnv(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                verbose=verbose,
            )

        elif backend == "sglang":

            # tp_size = torch.cuda.device_count()
            # if tp_size > 1 and tp_size % 2:
            #     tp_size -= 1
            tp_size = 1

            self.llm = SGLangEnv(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                port=port,
                host=host,
                disable_overlap_schedule=disable_overlap_schedule,
                mem_fraction_static=memory_utilization,
                log_level=log_level,
                gpu_id=0,
                tp_size=tp_size,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                verbose=verbose,
            )

        elif backend == "openai" or backend == "deepseek":
            self.llm = OpenAIEnv(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                # max_tokens=max_tokens,
            )

        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.timer = self.llm.timer

    def complete(self, prompt):
        return self.llm.complete(prompt=prompt)

    def prompt_complete(self, **kwargs):
        return self.llm.prompt_complete(**kwargs)

    def prompt_complete_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict]:
        return self.llm.prompt_complete_batch(data_list)
    
    def hello_world(self):
        response = self.complete("who are you?")
        print_text("Q: who are you?\n", color="red")
        print_text(f"A: {response}\n", color="green")


def test_prompote_completion():

    llama3_8b_instruct = "/home/hdd/model/Meta-Llama-3-8B-Instruct"
    question = "What were the main objectives of the Apollo 11 mission?"

    context = (
        "Document 1:\n"
        "Apollo 11 was the first manned mission to land on the Moon. "
        "It was launched by NASA on July 16, 1969, with astronauts Neil Armstrong, "
        "Buzz Aldrin, and Michael Collins aboard.\n\n"
        "Document 2:\n"
        "The mission's main objectives included performing a crewed lunar landing, "
        "collecting samples from the Moon's surface, and returning safely to Earth.\n\n"
        "Document 3:\n"
        "On July 20, 1969, Neil Armstrong became the first human to step onto the lunar surface, "
        "followed by Buzz Aldrin. They collected 47.5 pounds of lunar material before returning on July 24."
    )

    ######### test ollama
    llm = LLMEnv(
        backend="ollama",
        model="llama3.1:8b",
        max_tokens=20,
        system_prompt=QA_SYSTEM,
        user_prompt=QA_USER,
        verbose=True,
    )

    ret = llm.prompt_complete(question=question, context=context)
    print_text(
        f"response: {ret['response']}, generate_time: {ret['generate_time']}, prefill_length: {ret['prefill_length']}, decode_length: {ret['decode_length']}",
        color="blue",
    )

    ######### test sglang
    llm = LLMEnv(
        backend="sglang",
        model=llama3_8b_instruct,
        max_tokens=20,
        system_prompt=QA_SYSTEM,
        user_prompt=QA_USER,
        verbose=True,
    )

    ret = llm.prompt_complete(question=question, context=context)
    print_text(
        f"response: {ret['response']}, generate_time: {ret['generate_time']}, prefill_length: {ret['prefill_length']}, decode_length: {ret['decode_length']}",
        color="blue",
    )

    ######### test huggingface
    llm = LLMEnv(
        backend="huggingface",
        model=llama3_8b_instruct,
        max_tokens=20,
        system_prompt=QA_SYSTEM,
        user_prompt=QA_USER,
        verbose=True,
    )
    ret = llm.prompt_complete(question=question, context=context)
    print_text(
        f"response: {ret['response']}, generate_time: {ret['generate_time']}, prefill_length: {ret['prefill_length']}, decode_length: {ret['decode_length']}",
        color="blue",
    )


def test_complete():

    llama3_8b_instruct = "/home/hdd/model/Meta-Llama-3-8B-Instruct"

    question = "What is the capital of France?"

    ######### test ollama
    llm = LLMEnv(
        backend="ollama",
        model="llama3.1:8b",
        max_tokens=20,
    )

    ret = llm.complete(prompt=question)
    print_text(
        f"response: {ret['response']}, generate_time: {ret['generate_time']}, prefill_length: {ret['prefill_length']}, decode_length: {ret['decode_length']}",
        color="blue",
    )

    ######### test sglang
    llm = LLMEnv(
        backend="sglang",
        model=llama3_8b_instruct,
        max_tokens=20,
    )
    ret = llm.complete(prompt=question)
    print_text(
        f"response: {ret['response']}, generate_time: {ret['generate_time']}, prefill_length: {ret['prefill_length']}, decode_length: {ret['decode_length']}",
        color="blue",
    )

    ######### test huggingface
    llm = LLMEnv(
        backend="huggingface",
        model=llama3_8b_instruct,
        max_tokens=20,
    )
    ret = llm.complete(prompt=question)
    print_text(
        f"response: {ret['response']}, generate_time: {ret['generate_time']}, prefill_length: {ret['prefill_length']}, decode_length: {ret['decode_length']}",
        color="blue",
    )


if __name__ == "__main__":

    # how to use: CUDA_VISIBLE_DEVICES=0 python -m utils.llm

    llama3_8b_instruct = "/home/hdd/model/Meta-Llama-3-8B-Instruct"

    question = "What is the capital of France?"

    ######### test ollama
    llm = LLMEnv(
        backend="ollama",
        model="llama3.1:8b",
        max_tokens=20,
        verbose=True,
    )

    ret = llm.complete(prompt=question)
    print_text(
        f"response: {ret['response']}, generate_time: {ret['generate_time']}, prefill_length: {ret['prefill_length']}, decode_length: {ret['decode_length']}",
        color="blue",
    )

    question = "What were the main objectives of the Apollo 11 mission?"

    context = (
        "Document 1:\n"
        "Apollo 11 was the first manned mission to land on the Moon. "
        "It was launched by NASA on July 16, 1969, with astronauts Neil Armstrong, "
        "Buzz Aldrin, and Michael Collins aboard.\n\n"
        "Document 2:\n"
        "The mission's main objectives included performing a crewed lunar landing, "
        "collecting samples from the Moon's surface, and returning safely to Earth.\n\n"
        "Document 3:\n"
        "On July 20, 1969, Neil Armstrong became the first human to step onto the lunar surface, "
        "followed by Buzz Aldrin. They collected 47.5 pounds of lunar material before returning on July 24."
    )

    ######### test sglang
    llm = LLMEnv(
        backend="sglang",
        model=llama3_8b_instruct,
        max_tokens=20,
        system_prompt=QA_SYSTEM,
        user_prompt=QA_USER,
        verbose=True,
    )

    ret = llm.prompt_complete(question=question, context=context)
    print_text(
        f"response: {ret['response']}, generate_time: {ret['generate_time']}, prefill_length: {ret['prefill_length']}, decode_length: {ret['decode_length']}",
        color="blue",
    )
