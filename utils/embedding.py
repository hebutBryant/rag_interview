from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from utils.base import print_text, read_yaml
from utils.timer import Timer

EMBEDD_DIMS = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "text-embedding-ada-002": 1536,
}


class EmbeddingEnv:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = None,  # e.g., "cuda:0" / "cpu" / None(自动)
        normalize: bool = True,
        batch_size: int = -1,
        pooling: str = None,  # "cls" / "mean" / None(自动判断)
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        if pooling is not None:
            self.pooling = pooling
        else:
            if model_name.startswith("sentence-transformers/"):
                self.pooling = "mean"
            else:
                self.pooling = "cls"

        print(
            f"Loading model {self.model_name} to {self.device} "
            f"(pooling={self.pooling}) ..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name)

        self.config = AutoConfig.from_pretrained(model_name)

        max_pos = getattr(self.config, "max_position_embeddings", None)
        tok_max = getattr(self.tokenizer, "model_max_length", None)

        self.max_length = min(max_pos, tok_max)

        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            dummy_input = self.tokenizer("test", return_tensors="pt").to(self.device)
            dummy_output = self.model(**dummy_input)
            dummy_emb = self._pool(dummy_output, dummy_input["attention_mask"])
        self.dim = dummy_emb.shape[-1]

        print(
            f"EmbeddingEnv init -> model={self.model_name}, "
            f"dim={self.dim}, device={self.device}"
        )

    def __str__(self):
        return f"{self.model_name} ({self.dim}d, pooling={self.pooling})"

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # last_hidden_state: [B, L, H]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _pool(self, model_output, attention_mask):
        """根据 self.pooling 选择 CLS / mean pooling。"""
        if self.pooling == "mean":
            return self._mean_pooling(model_output, attention_mask)
        elif self.pooling == "cls":
            # 等价于你原来的 model_output[0][:, 0]
            return model_output[0][:, 0]
        else:
            # 理论上不会走到这里，做个兜底
            return model_output[0][:, 0]

    # ---------- 编码主逻辑 ----------
    def _encode(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else texts

        all_embeddings = []

        batch_size = self.batch_size if self.batch_size > 0 else len(texts)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                batch_embeddings = self._pool(
                    model_output, encoded_input["attention_mask"]
                )

            if self.normalize:
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            all_embeddings.append(batch_embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        return all_embeddings[0] if single else all_embeddings

    # ---------- 对外接口 ----------
    def get_embedding(self, text: str) -> np.ndarray:
        return self._encode(text)

    def get_embeddings(self, texts: list) -> np.ndarray:
        return self._encode(texts)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        e1 = self.get_embedding(text1)
        e2 = self.get_embedding(text2)
        sim = float(np.dot(e1, e2))
        return round(sim, 6)


if __name__ == "__main__":

    timer = Timer(name="test embed model", skip=0)

    num = 1000

    examples = [
        "Each sentence is converted" * 50,
    ] * num

    env = EmbeddingEnv(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,
    )

    # warmup
    embs = env.get_embeddings(examples)
    with timer.timing(f"minillm {num} vectors"):
        embs = env.get_embeddings(examples)

    env = EmbeddingEnv(
        model_name="BAAI/bge-large-zh-v1.5",
        batch_size=64,
    )

    # warmup
    embs = env.get_embeddings(examples)
    with timer.timing(f"bge-large {num} vectors"):
        embs = env.get_embeddings(examples)

    print(timer.summary())
