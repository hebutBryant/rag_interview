import os
import time
import numpy as np
import dashscope
from http import HTTPStatus
from typing import List, Union


class EmbeddingEnv:
    def __init__(
        self,
        model_name: str = "text-embedding-v4",
        api_key: str = None,
        normalize: bool = True,
        batch_size: int = 10,   # 官方建议 <=10
        max_retries: int = 3,
        sleep_time: float = 1.0,
        device: str = None,     # 兼容旧调用方（EntitiesDB/Pruning 会传 device）；
                                # DashScope 是云端 API，本参数无实际作用，仅作占位忽略
        **kwargs,               # 吞掉其它历史遗留 kwargs，避免 TypeError
    ):
        """
        DashScope embedding API 版本

        参数:
            model_name: embedding模型
            api_key: 阿里云百炼key
            normalize: 是否L2归一化
            batch_size: 单次请求条数（建议<=10）
        """
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = min(batch_size, 10)
        self.max_retries = max_retries
        self.sleep_time = sleep_time

        # 设置API Key
        dashscope.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not dashscope.api_key:
            raise ValueError("Please set DASHSCOPE_API_KEY")

        # 先跑一次拿dim
        test_vec = self.get_embedding("test")
        self.dim = len(test_vec)

        print(f"QwenEmbeddingEnv init -> model={self.model_name}, dim={self.dim}")

    def __str__(self):
        return f"{self.model_name} ({self.dim}d, dashscope)"

    # ---------- 核心请求 ----------
    def _request(self, texts: List[str]) -> np.ndarray:
        last_err = None

        for _ in range(self.max_retries):
            try:
                resp = dashscope.TextEmbedding.call(
                    model=self.model_name,
                    input=texts
                )

                if resp.status_code == HTTPStatus.OK:
                    embeddings = [
                        item["embedding"]
                        for item in resp.output["embeddings"]
                    ]
                    return np.array(embeddings, dtype=np.float32)
                else:
                    last_err = resp
                    time.sleep(self.sleep_time)

            except Exception as e:
                last_err = e
                time.sleep(self.sleep_time)

        raise RuntimeError(f"DashScope API failed: {last_err}")

    def _normalize(self, x):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, 1e-12, None)

    # ---------- 编码 ----------
    def _encode(self, texts: Union[str, List[str]]):
        single = isinstance(texts, str)
        texts = [texts] if single else texts

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_emb = self._request(batch)

            if self.normalize:
                batch_emb = self._normalize(batch_emb)

            all_embeddings.append(batch_emb)

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        return all_embeddings[0] if single else all_embeddings

    # ---------- 对外接口 ----------
    def get_embedding(self, text: str):
        return self._encode(text)

    def get_embeddings(self, texts: List[str]):
        return self._encode(texts)

    def calculate_similarity(self, text1: str, text2: str):
        e1 = self.get_embedding(text1)
        e2 = self.get_embedding(text2)
        return round(float(np.dot(e1, e2)), 6)


# ---------- 测试 ----------
if __name__ == "__main__":

    env = EmbeddingEnv(
        model_name="text-embedding-v4",
        normalize=True,
        batch_size=10,
    )

    text = "衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢"
    emb = env.get_embedding(text)
    print("dim:", emb.shape)

    texts = [
        "今天天气很好",
        "我喜欢机器学习",
        "这个商品质量很好"
    ]
    embs = env.get_embeddings(texts)
    print("batch shape:", embs.shape)

    sim = env.calculate_similarity("今天天气不错", "今天适合出门")
    print("similarity:", sim)