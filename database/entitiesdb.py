import os

import faiss
import numpy as np
from tqdm import tqdm

from database.igraph import IGraph
from dataset import metaqa
from utils.base import get_base_dir
from utils.embedding import EmbeddingEnv
from utils.timer import Timer

class EntitiesDB:
    def __init__(
        self,
        db_name,
        entities,
        embed_name="BAAI/bge-large-en-v1.5",
        overwrite=True,
        batch_size=10,
        device="cuda:0",
        db_dir=None,
    ):
        self.embed_model = EmbeddingEnv(
            model_name=embed_name, batch_size=batch_size, device=device
        )
        self.db_name = db_name
        self.batch_size = batch_size
        self.timer = Timer(name="EntitiesDB", skip=0)

        if db_dir is None:
            db_dir = os.path.join(get_base_dir(), "entities_db")
        os.makedirs(db_dir, exist_ok=True)

        model_short_name = embed_name.split("/")[-1]

        self.index_path = os.path.join(db_dir, f"{db_name}_{model_short_name}.index")
        self.meta_path = os.path.join(db_dir, f"{db_name}_{model_short_name}_meta.npy")

        if (
            os.path.exists(self.index_path)
            and os.path.exists(self.meta_path)
            and not overwrite
        ):
            self.load()
        else:
            print(f"Creating new FAISS index for {db_name}:")
            self.dim = self.embed_model.dim
            self.index = faiss.IndexFlatIP(self.dim)

            self.entities = sorted(list(entities))
            self.id2entity = {i: entity for i, entity in enumerate(self.entities)}

            self.generate_embedding_and_insert()
            self.save()

        if faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)

    def save(self):
        print(f"Saving FAISS index to {self.index_path}")
        faiss.write_index(self.index, self.index_path)
        np.save(self.meta_path, np.array(self.entities))

    def load(self):
        print(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.meta_path):
            self.entities = np.load(self.meta_path, allow_pickle=True).tolist()
            self.id2entity = {i: e for i, e in enumerate(self.entities)}
        else:
            raise FileNotFoundError(f"Meta file {self.meta_path} not found")

    def generate_embedding_and_insert(self):
        n_entities = len(self.entities)

        for i in tqdm(
            range(0, n_entities, self.batch_size),
            desc=f"start generate embedding and insert for {self.db_name}",
        ):
            start_idx = i
            end_idx = min(n_entities, i + self.batch_size)
            embeddings = self.get_embedding(self.entities[start_idx:end_idx])
            self.insert(embeddings)

    def get_embedding(self, query):
        if isinstance(query, list):
            embedding = self.embed_model.get_embeddings(query)
        else:
            embedding = self.embed_model.get_embedding(query)
        return embedding

    def insert(self, query_embeddings):
        if not isinstance(query_embeddings, np.ndarray):
            query_embeddings = np.array(query_embeddings, dtype=np.float32)

        faiss.normalize_L2(query_embeddings)
        self.index.add(query_embeddings)

    def search(self, queries, top_k=5):
        single_flag = isinstance(queries, str)
        if single_flag:
            queries = [queries]

        with self.timer.timing("embedding query"):
            query_embeddings = self.get_embedding(queries)

        if not isinstance(query_embeddings, np.ndarray):
            query_embeddings = np.array(query_embeddings, dtype=np.float32)

        # FAISS requires 2D arrays: (number_of_queries, vector_dimension)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[np.newaxis, :]

        faiss.normalize_L2(query_embeddings)
        
        with self.timer.timing("entity matching"):
            distances, ids = self.index.search(query_embeddings, top_k)
            # print(f"Search results IDs: {ids}")
            # print(f"Search results Distances: {distances}")

            similar_entities_list = []
            for q_idx in range(len(queries)):
                entities = []
                for i in range(top_k):
                    entity_id = ids[q_idx][i]
                    if entity_id >= len(self.entities) or entity_id < 0:
                        raise ValueError(
                            f"Invalid entity index: {entity_id}. "
                            f"Valid index range: 0-{len(self.entities)-1}. "
                            f"Query: '{queries[q_idx]}'"
                        )
                    entities.append(self.entities[entity_id])
                    
                similar_entities_list.append(entities)

        if single_flag:
            return similar_entities_list[0], distances[0]
        return similar_entities_list, distances


if __name__ == "__main__":
    dataset = "metaqa"
    triplets = metaqa.get_triplets()

    igraph_env = IGraph(dataset=dataset, triplets=triplets)
    db = EntitiesDB(
        db_name=f"{dataset}_entities",
        entities=igraph_env.entities(),
        overwrite=False,
    )
    # batch queries
    queries = ["what movies are about ginger rogers",
                "which movies can be described by moore",
                "what films can be described by occupation",
                "which films are about jacques tati",
                "what movies are about donnie darko"]
    similar_entities, distances = db.search(queries, top_k=5)

    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: '{query}'")
        print("Similar entities:")
        for j, (entity, distance) in enumerate(zip(similar_entities[i], distances[i])):
            print(f"  {j+1}. {entity} (similarity: {distance:.4f})")
    print(f"batch queries time:\n {db.timer.last_durations()}")