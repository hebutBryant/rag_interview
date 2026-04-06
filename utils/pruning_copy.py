import time

import numpy as np

from utils.base import print_text
from utils.embedding import EmbeddingEnv
from utils.timer import Timer

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",
#                                    embed_batch_size=10)
embed_model = None
import faiss
from typing import Dict, List, Tuple

class Pruning:

    def __init__(
        self,
        model="BAAI/bge-small-en-v1.5",
        device="cuda:0",
        batch_size=100,
        embed_model=None,
        timer_skip=3,
    ):
        if embed_model is not None:
            self.embed_model = embed_model
        else:
            self.embed_model = EmbeddingEnv(
                model_name=model, batch_size=batch_size, device=device
            )
        self.timer = Timer(name="Pruning",skip=timer_skip)

    def get_embedding(self, text):
        if isinstance(text, list):
            embedding = self.embed_model.get_embeddings(text)
        else:
            embedding = self.embed_model.get_embedding(text)
        return embedding

    def semantic_pruning_triplets(
        self, question, triplets, rel_embeddings=None, topk=30
    ):
        with self.timer.timing("pruning_embed_query"):
            question_embed = np.array(
                self.get_embedding(question), dtype="float32"
            ).reshape(1, -1)

        print(f"len triplets: {len(triplets)}")
        with self.timer.timing("pruning_embed_triplets"):
            if rel_embeddings is None:
                rel_embeddings = self.get_embedding(triplets)
                print(f"len rel_embeddings: {len(rel_embeddings)}")
                # print(f"kg_embedding cost {time_triplet_embedding:.3f}s")
        # print(f"embed_time_cost: {embed_time_cost:.3f}s")

        with self.timer.timing("pruning_compute_similarity"):
            # rel_embeddings = np.array(rel_embeddings, dtype=np.float32)
            # rel_embeddings = rel_embeddings
            faiss.normalize_L2(rel_embeddings)
            faiss.normalize_L2(question_embed)

            dim = self.embed_model.dim
            index = faiss.IndexFlatIP(dim)
            if hasattr(faiss, "StandardGpuResources"):
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)

            index.add(rel_embeddings)
            actual_topk = min(topk, len(rel_embeddings))
            distances, indices = index.search(question_embed, actual_topk)

            results = [
                (triplets[i], float(distances[0][j])) for j, i in enumerate(indices[0])
            ]
        # print_text(f"faiss cost: {time_cost:.3f}s", color='red')

        return results
    def semantic_pruning_triplets_thread(
        self, question, triplets, rel_embeddings=None, topk=30
    ):
        with self.timer.timing("pruning_embed_query_thread"):
            question_embed = np.array(
                self.get_embedding(question), dtype="float32"
            ).reshape(1, -1)

        with self.timer.timing("pruning_embed_triplets_thread"):
            if rel_embeddings is None:
                rel_embeddings = self.get_embedding(triplets)
                # print(f"len rel_embeddings: {len(rel_embeddings)}")
                # print(f"kg_embedding cost {time_triplet_embedding:.3f}s")
        # print(f"embed_time_cost: {embed_time_cost:.3f}s")

        with self.timer.timing("pruning_compute_sim_thread"):
            rel_embeddings = np.array(rel_embeddings, dtype=np.float32)
            faiss.normalize_L2(rel_embeddings)
            faiss.normalize_L2(question_embed)

            dim = self.embed_model.dim
            index = faiss.IndexFlatIP(dim)
            if hasattr(faiss, "StandardGpuResources"):
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)

            index.add(rel_embeddings)
            actual_topk = min(topk, len(rel_embeddings))
            distances, indices = index.search(question_embed, actual_topk)

            results = [
                (triplets[i], float(distances[0][j])) for j, i in enumerate(indices[0])
            ]
        # print_text(f"faiss cost: {time_cost:.3f}s", color='red')

        return results
    def semantic_pruning_triplets_batch(
        self,
        questions: List[str],
        question_triplets: List[Dict[str, List[str]]],
        topk: int = 30,
    ):

        with self.timer.timing("pruning_unique_triplets_batch"):
            all_triplets = []
            for question in question_triplets:
                for paths in question.values():
                    all_triplets.extend(paths)
            unique_triplets = list(all_triplets)

        unique_triplets = list(set(all_triplets))
        print(f"Total unique triplets for pruning: {len(unique_triplets)}")

        # Batch embedding
        with self.timer.timing("pruning_embed_questions_batch"):
            question_embeds = np.array(
                self.get_embedding(questions), dtype="float32"
            )

        with self.timer.timing("pruning_embed_triplets_batch"):
            # triplet_embeds = np.array(
            #     self.get_embedding(unique_triplets), dtype="float32"
            # )
            triplet_embeds = self.get_embedding(unique_triplets)
        with self.timer.timing("pruning_compute_sim_batch"):
            # Normalize
            faiss.normalize_L2(question_embeds)
            faiss.normalize_L2(triplet_embeds)

            # Triplet index mapping
            triplet2idx = {t: i for i, t in enumerate(unique_triplets)}
            dim = triplet_embeds.shape[1]

            # Per-question pruning
            results = []
            for qid, subgraph in enumerate(question_triplets):
                question_embed = question_embeds[qid].reshape(1, -1)

                q_results = []
                for entity, paths in subgraph.items():
                    if not paths:
                        continue

                    indices = [triplet2idx[t] for t in paths]
                    sub_embeds = triplet_embeds[indices]

                    # subgraph FAISS
                    sub_index = faiss.IndexFlatIP(dim)
                    sub_index.add(sub_embeds)

                    k = min(topk, len(paths))
                    distances, indices_topk = sub_index.search(question_embed, k)

                    top_triplets = [(paths[i], float(distances[0][j])) for j, i in enumerate(indices_topk[0])]
                    q_results.append(top_triplets)

                results.append(q_results)
        return results


if __name__ == "__main__":
    pruner = Pruning(model="BAAI/bge-small-en-v1.5", device="cuda:0", batch_size=10)

    question = "What kind of thing is a cat?"
    triplets = [
        "dog is an animal",
        "apple is a fruit",
        "car is a vehicle",
        "cat is a pet",
        "banana grows on trees",
        "dog likes bones",
        "airplane flies in the sky",
        "fish lives in the water",
    ]

    results = pruner.semantic_pruning_triplets(question, triplets, topk=30)
    print("\nTop-5 most similar triplets:")
    for rank, (text, score) in enumerate(results, start=1):
        print(f"{rank:>2}. {text:40s}  (similarity={score:.4f})")

    questions = [
        "What kind of thing is a cat?",
        "Where do airplanes fly?"
    ]

    # 每个问题对应的子图（每个实体的 triplets）
    question_triplets = [
        {  # 问题 0
            "entity_cat": [
                "dog is an animal",
                "cat is a pet",
                "cat has fur",
                "cat likes milk",
            ],
            "entity_dog": [
                "dog is an animal",
                "dog likes bones"
            ]
        },
        {  # 问题 1
            "entity_airplane": [
                "airplane flies in the sky",
                "airplane has wings",
                "airplane carries passengers"
            ],
            "entity_bird": [
                "bird flies in the sky",
                "bird has feathers"
            ]
        }
    ]

    # 执行批量剪枝，topk=2
    results = pruner.semantic_pruning_triplets_batch(
        questions=questions,
        question_triplets=question_triplets,
        topk=2
    )

    # 打印结果
    for qid, subgraph_results in enumerate(results):
        print(f"\nQuestion {qid}: {questions[qid]}")
        for entity_idx, entity_topk in enumerate(subgraph_results):
            print(f"  Entity {entity_idx} topk triplets:")
            for rank, (triplet, score) in enumerate(entity_topk, start=1):
                print(f"    {rank:>2}. {triplet:40s} (score={score:.4f})")
