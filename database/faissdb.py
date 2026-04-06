import os
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class FaissDB:
    def __init__(
        self,
        db_name,
        texts,
        embed_name="/home/hdd/model/bge-large-en-v1.5",
        overwrite=True,
        batch_size=10,
        device="cuda:1",
        db_dir="./faiss_db",
        use_gpu_index=False,
        query_instruction="Represent this sentence for searching relevant passages: ",
    ):
        """
        Args:
            db_name: 索引库名字
            texts: 要存入 FAISS 的文本列表
            embed_name: embedding 模型名
            overwrite: 是否覆盖已存在索引
            batch_size: 编码批大小
            device: embedding 模型运行设备，例如 cuda:1
            db_dir: 索引保存目录
            use_gpu_index: 是否把 FAISS index 搬到 GPU 上检索
            query_instruction: BGE query instruction
        """
        self.db_name = db_name
        self.batch_size = batch_size
        self.device = device
        self.embed_name = embed_name
        self.use_gpu_index = use_gpu_index
        self.query_instruction = query_instruction

        os.makedirs(db_dir, exist_ok=True)

        model_short_name = embed_name.split("/")[-1]
        self.index_path = os.path.join(db_dir, f"{db_name}_{model_short_name}.index")
        self.meta_path = os.path.join(db_dir, f"{db_name}_{model_short_name}_meta.npy")

        print(f"Loading embedding model: {embed_name} on {device}")
        self.embed_model = SentenceTransformer(embed_name, device=device)
        self.dim = self.embed_model.get_sentence_embedding_dimension()

        if (
            os.path.exists(self.index_path)
            and os.path.exists(self.meta_path)
            and not overwrite
        ):
            self.load()
        else:
            print(f"Creating new FAISS index for {db_name}")
            self.index = faiss.IndexFlatIP(self.dim)

            self.texts = list(texts)
            self.id2text = {i: text for i, text in enumerate(self.texts)}

            self.generate_embedding_and_insert()
            self.save()

        if self.use_gpu_index:
            self._move_index_to_gpu()

    def _move_index_to_gpu(self):
        """把 CPU index 搬到指定 GPU 上"""
        if not self.device.startswith("cuda:"):
            print("Device is not CUDA, keep CPU index.")
            return

        gpu_id = int(self.device.split(":")[1])

        if faiss.get_num_gpus() <= gpu_id:
            print(
                f"Warning: FAISS available GPUs = {faiss.get_num_gpus()}, requested GPU = {gpu_id}. Keep CPU index."
            )
            return

        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)
            print(f"FAISS index moved to GPU {gpu_id}")
        except Exception as e:
            print(f"Move index to GPU failed: {e}")
            print("Keep CPU index.")

    def save(self):
        """保存索引和元数据"""
        print(f"Saving FAISS index to {self.index_path}")

        index_to_save = self.index
        try:
            # 如果当前是 GPU index，先转回 CPU 再保存
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        except Exception:
            pass

        faiss.write_index(index_to_save, self.index_path)
        np.save(self.meta_path, np.array(self.texts, dtype=object))

    def load(self):
        """加载索引和元数据"""
        print(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        if os.path.exists(self.meta_path):
            self.texts = np.load(self.meta_path, allow_pickle=True).tolist()
            self.id2text = {i: text for i, text in enumerate(self.texts)}
        else:
            raise FileNotFoundError(f"Meta file {self.meta_path} not found")

    def generate_embedding_and_insert(self):
        """为所有文本生成 embedding 并写入索引"""
        n_texts = len(self.texts)

        for i in tqdm(
            range(0, n_texts, self.batch_size),
            desc=f"Generating embeddings for {self.db_name}",
        ):
            start_idx = i
            end_idx = min(n_texts, i + self.batch_size)
            batch_texts = self.texts[start_idx:end_idx]
            embeddings = self.get_embedding(batch_texts, is_query=False)
            self.insert(embeddings)

    def get_embedding(self, query, is_query=True):
        """
        query 可以是 str 或 list[str]
        BGE 模型下，query 侧加 instruction，doc/text 侧不加
        """
        single_flag = isinstance(query, str)
        if single_flag:
            query = [query]

        if is_query:
            query = [self.query_instruction + q for q in query]

        embeddings = self.embed_model.encode(
            query,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if single_flag:
            return embeddings[0]
        return embeddings

    def insert(self, embeddings):
        """插入向量到索引"""
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        embeddings = embeddings.astype(np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        self.index.add(embeddings)

    def search(self, queries, top_k=5):
        """
        Args:
            queries: str or list[str]
            top_k: 返回最相似的 top_k 个文本
        Returns:
            单 query:
                matched_texts, distances, matched_ids
            多 query:
                matched_texts_list, distances, ids
        """
        single_flag = isinstance(queries, str)
        if single_flag:
            queries = [queries]

        query_embeddings = self.get_embedding(queries, is_query=True)

        if not isinstance(query_embeddings, np.ndarray):
            query_embeddings = np.array(query_embeddings, dtype=np.float32)

        query_embeddings = query_embeddings.astype(np.float32)

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[np.newaxis, :]

        distances, ids = self.index.search(query_embeddings, top_k)

        matched_texts_list = []
        matched_ids_list = []

        for q_idx in range(len(queries)):
            matched_texts = []
            matched_ids = []

            for i in range(top_k):
                text_id = int(ids[q_idx][i])
                if text_id < 0 or text_id >= len(self.texts):
                    continue
                matched_texts.append(self.texts[text_id])
                matched_ids.append(text_id)

            matched_texts_list.append(matched_texts)
            matched_ids_list.append(matched_ids)

        if single_flag:
            return matched_texts_list[0], distances[0][:len(matched_texts_list[0])], matched_ids_list[0]

        return matched_texts_list, distances, matched_ids_list


def main():
    texts = [
        "Donnie Darko is a psychological thriller movie.",
        "Ginger Rogers was a famous actress and dancer.",
        "The movie is about time travel and parallel universes.",
        "Jacques Tati is a French filmmaker.",
        "This film explores human emotions and memory.",
        "Interstellar is a science fiction film about space and time.",
        "The Godfather is a crime drama film.",
        "A beautiful dancer performed in a classic Hollywood musical.",
    ]

    db = FaissDB(
        db_name="demo_texts",
        texts=texts,
        overwrite=True,          # 第一次建库用 True；后面复用索引用 False
        batch_size=4,
        device="cuda:1",
        db_dir="./faiss_db",
        use_gpu_index=False,     # 先用 CPU index，最稳
    )

    queries = [
        "movies about time travel",
        "french director",
        "classic actress and dancer",
        "crime family movie",
    ]

    matched_texts, distances = db.search(queries, top_k=3)

    for i, query in enumerate(queries):
        print(f"\nQuery {i + 1}: {query}")
        print("Top results:")
        for j, (text, score) in enumerate(zip(matched_texts[i], distances[i])):
            print(f"  {j + 1}. {text} (similarity={score:.4f})")

    # 单条查询测试
    single_query = "science fiction movie about space"
    single_results, single_scores = db.search(single_query, top_k=3)

    print(f"\nSingle Query: {single_query}")
    print("Top results:")
    for i, (text, score) in enumerate(zip(single_results, single_scores)):
        print(f"  {i + 1}. {text} (similarity={score:.4f})")


if __name__ == "__main__":
    main()