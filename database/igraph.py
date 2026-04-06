import argparse
import atexit
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import igraph as ig
from tqdm import tqdm

from dataset import metaqa
from utils.base import get_base_dir, read_json, save_json
from utils.timer import Timer

class IGraph:
    def __init__(
        self,
        dataset: str,
        triplets: List[Tuple[str, str, str]] = None,
        db_dir=None,
    ):
        self.dataset = dataset
        self.graph: ig.Graph = None

        if db_dir is None:
            db_dir = os.path.join(get_base_dir(), "igraph_db")
        os.makedirs(db_dir, exist_ok=True)
        self.save_path = os.path.join(db_dir, f"{self.dataset}_igraph.gml")

        atexit.register(self._close)
        loaded = self._load_graph(triplets)
        if loaded is None:
            assert triplets is not None, "Triplets must be provided for new graph."
            self._build_graph(triplets)
        self.timer = Timer(name=f"IGraph-{self.dataset}", skip=0)

    def _build_graph(self, triplets: List[Tuple[str, str, str]]):
        self.graph = ig.Graph(directed=True)
        print(f"Building knowledge graph...")

        triplets = list(set(triplets))
        self.upsert_triplets(triplets)
        print(f"Graph built: {len(self.graph.vs)} nodes, {len(self.graph.es)} edges.")

        try:
            self.graph.save(self.save_path, format="gml")
            print(f"Graph saved successfully to: {self.save_path}")
        except Exception as e:
            print(f"Failed to save graph: {e}")

    def _load_graph(self, triplets):
        if not os.path.exists(self.save_path):
            print(f"Graph file not found: {self.save_path}")
            return None
        try:
            self.graph = ig.Graph.Load(self.save_path, format="gml")
            saved_triplets = self.triplets()
            if set(saved_triplets) == set(triplets):
                print(f"Graph loaded successfully from: {self.save_path}")
                print(f"{len(self.graph.vs)} nodes, {len(self.graph.es)} edges.")
                return self.graph
            else:
                print("Graph file exists but content differs from provided triplets.")
                self.graph = None
                return None
        except Exception as e:
            print(f"Failed to load graph: {e}")
            return None

    def _close(self):
        if self.graph is None:
            return

        if "id" in self.graph.vs.attributes():
            del self.graph.vs["id"]

        current_triplets = set(self.triplets())
        saved_triplets = set()
        if os.path.exists(self.save_path):
            saved_graph = ig.Graph.Load(self.save_path, format="gml")
            for e in saved_graph.es:
                src, tgt = e.tuple
                head = saved_graph.vs[src]["name"]
                tail = saved_graph.vs[tgt]["name"]
                rel = (
                    e["name"]
                    if "name" in saved_graph.es.attribute_names()
                    else "unknown"
                )
                saved_triplets.add((head, rel, tail))

        if current_triplets != saved_triplets:
            print("Graph content changed — updating file...")
            try:
                self.graph.save(self.save_path, format="gml")
                print(f"Graph re-saved to: {self.save_path}")
            except Exception as e:
                print(f"Failed to save updated graph: {e}")
        else:
            print("Graph unchanged — no need to rewrite file.")

    def upsert_triplets(
        self, triplets: Union[Tuple[str, str, str], List[Tuple[str, str, str]]]
    ):
        if isinstance(triplets, tuple) and len(triplets) == 3:
            triplets = [triplets]
        elif not isinstance(triplets, list):
            raise TypeError("triplets must be a tuple or a list of tuples.")

        existing_nodes = set(self.entities()) if self.graph is not None else set()
        existing_edges = set(self.triplets()) if self.graph is not None else set()
        edges = []
        attributes = {"name": []}
        nodes = set()
        for head, rel, tail in triplets:
            if (head, rel, tail) in existing_edges:
                print(
                    f"Triplet already exists: {head} -{rel}-> {tail}, skip insertion."
                )
                continue
            nodes.add(head)
            nodes.add(tail)
            edges.append((head, tail))
            attributes["name"].append(rel)

        self.graph.add_vertices(list(nodes - existing_nodes))
        self.graph.add_edges(edges, attributes)
        # self.graph.vs["label"] = self.graph.vs["name"]
        print(f"Upsert {len(triplets)} triplets")

    def delete_triplets(
        self, triplets: Union[Tuple[str, str, str], List[Tuple[str, str, str]]]
    ):
        if isinstance(triplets, tuple) and len(triplets) == 3:
            triplets = [triplets]
        elif not isinstance(triplets, list):
            raise TypeError("triplets must be a tuple or a list of tuples")

        edges_to_remove = []
        for head, rel, tail in triplets:
            head_matches = self.graph.vs.select(name_eq=head)
            tail_matches = self.graph.vs.select(name_eq=tail)

            if len(head_matches) == 0 or len(tail_matches) == 0:
                print(f"Vertex not found: {head} or {tail}, skip deletion.")
                continue

            source_idx = head_matches[0].index
            target_idx = tail_matches[0].index

            edges = self.graph.es.select(
                _source=source_idx, _target=target_idx, name=rel
            )

            if edges:
                edges_to_remove.extend(edges.indices)
                print(f"Marked for deletion: {head} -{rel}-> {tail}")
            else:
                print(f"Triplet not found: {head} -{rel}-> {tail}")

        if edges_to_remove:
            self.graph.delete_edges(edges_to_remove)
            print(f"Deleted {len(edges_to_remove)} triplets")

    def entities_num(self):
        if self.graph is None:
            raise RuntimeError("Graph has not been loaded !")
        return self.graph.vcount()

    def triplets_num(self):
        if self.graph is None:
            raise RuntimeError("Graph has not been loaded !")
        return self.graph.ecount()

    def entities(self) -> List[str]:
        if self.graph is None:
            raise RuntimeError("Graph has not been loaded !")
        entities = list(set(vertex["name"] for vertex in self.graph.vs))
        return entities

    def triplets(self) -> List[Tuple[str, str, str]]:
        if self.graph is None:
            raise RuntimeError("Graph has not been loaded !")

        triplets = []
        for e in self.graph.es:
            src, tgt = e.tuple
            head = self.graph.vs[src]["name"]
            tail = self.graph.vs[tgt]["name"]
            rel = e["name"] if "name" in self.graph.es.attribute_names() else "unknown"
            triplets.append((head, rel, tail))

        return triplets

    def subgraph_extraction_to_paths_simple(
        self, entities: List[str], hop: int
    ) -> Dict[str, List[str]]:
        """
        Extracts all simple paths within specified hops starting from given entities.
        The returned paths are in the format of entity name lists, excluding edges between entities.
        For example, with hop=2, paths follow the format: [starting entity, intermediate entity, ending entity]

        """
        entity_path_dict = {}

        for entity in entities:
            if entity not in self.graph.vs["name"]:
                print(f"Entity '{entity}' not found in graph.")
                continue

            entity_idx = self.graph.vs.find(name=entity).index
            entity_path_dict[entity] = []
            all_paths = self.graph.get_all_simple_paths(
                v=entity_idx, to=None, minlen=1, maxlen=hop, mode="all"
            )

            name_paths = []
            for path in all_paths:
                name_paths.append([self.graph.vs[idx]["name"] for idx in path])
            # print(name_paths)

            entity_path_dict[entity] = name_paths

        return entity_path_dict

    def convert_node_lists_to_paths(self, entity_nodes_dict, edge_num=1):
        entity_path_dict = {}

        for entity, node_lists in entity_nodes_dict.items():
            entity_path_dict[entity] = []
            for node_list in node_lists:
                if not node_list or len(node_list) < 2:
                    continue

                if isinstance(node_list[0], str):
                    try:
                        node_indices = [
                            self.graph.vs.find(name=n).index for n in node_list
                        ]
                    except ValueError as e:
                        print(f"[Warning] Skipped invalid node: {e}")
                        continue
                else:
                    node_indices = node_list

                path_str = self.graph.vs[node_indices[0]]["name"]
                for i in range(len(node_indices) - 1):
                    u, v = node_indices[i], node_indices[i + 1]

                    edges = self.graph.es.select(_source=u, _target=v)
                    rel = ""
                    if len(edges) != 0:
                        if edge_num != -1 and len(edges) > edge_num:
                            edges = edges[:edge_num]
                        for edge in edges:
                            if rel != "":
                                rel += " | "
                            rel += (
                                edge["name"]
                                if "name" in self.graph.es.attribute_names()
                                else "unknown"
                            )
                        path_str += f" - {rel} -> {self.graph.vs[v]['name']}"
                    else:
                        edges = self.graph.es.select(_source=v, _target=u)
                        if len(edges) != 0:
                            if edge_num != -1 and len(edges) > edge_num:
                                edges = edges[:edge_num]
                            for edge in edges:
                                if rel != "":
                                    rel += " | "
                                rel += (
                                    edge["name"]
                                    if "name" in self.graph.es.attribute_names()
                                    else "unknown"
                                )
                            path_str += f" <- {rel} - {self.graph.vs[v]['name']}"
                        else:
                            u_name = (
                                self.graph.vs[u]["name"]
                                if "name" in self.graph.vs.attribute_names()
                                else str(u)
                            )
                            v_name = (
                                self.graph.vs[v]["name"]
                                if "name" in self.graph.vs.attribute_names()
                                else str(v)
                            )
                            raise ValueError(
                                f"Missing directed edge between '{u_name}' (index {u}) and "
                                f"'{v_name}' (index {v}) in both directions."
                            )
                entity_path_dict[entity].append(path_str)
        return entity_path_dict

    def subgraph_extraction_to_paths_dfs(self, entities: List[str], hop: int):
        """
        Extracts all paths within specified hops starting from given entities using DFS.
        The returned paths are in the format of triplet lists.

        """
        entity_path_dict = {}

        for entity in entities:
            with self.timer.timing(f"check entity"):
                if entity not in self.graph.vs["name"]:
                    print(f"Entity '{entity}' not found in the graph.")
                    continue
            
            with self.timer.timing(f"find entity index"):
                start_idx = self.graph.vs.find(name=entity).index
            completed_paths = []

            with self.timer.timing(f"DFS traversal"):
                def dfs(current_node, path_so_far, depth):
                    neighbors = []
                    for eid in self.graph.incident(current_node, mode="all"):
                        edge = self.graph.es[eid]
                        src, tgt = edge.tuple
                        other = tgt if src == current_node else src
                        triple = (
                            self.graph.vs[src]["name"],
                            (
                                edge["name"]
                                if "name" in self.graph.es.attribute_names()
                                else None
                            ),
                            self.graph.vs[tgt]["name"],
                        )
                        if triple not in path_so_far:
                            neighbors.append((other, triple))

                    if not neighbors or depth >= hop:
                        if path_so_far:
                            completed_paths.append(path_so_far)
                        return

                    for other_node, triple in neighbors:
                        dfs(other_node, path_so_far + [triple], depth + 1)

                dfs(start_idx, [], 0)
            entity_path_dict[entity] = completed_paths

        return entity_path_dict

    def convert_triplet_lists_to_paths(self, entity_triplets_dict):
        entity_path_dict = {}

        for entity, paths in entity_triplets_dict.items():
            str_paths = []

            for path in paths:
                if not path:
                    continue

                path_str = f"{entity}"
                current_head = entity

                for triple in path:
                    head, rel, tail = triple

                    if current_head == head:
                        path_str += f" - {rel} -> {tail}"
                        current_head = tail
                    elif current_head == tail:
                        path_str += f" <- {rel} - {head}"
                        current_head = head
                    else:
                        assert False, "Inconsistent path structure."

                str_paths.append(path_str.strip())

            entity_path_dict[entity] = str_paths

        return entity_path_dict


def test_subgraph_extraction_to_paths(depth: int = 2, num: int = 5):
    data = metaqa.get_metaqa_info(hop=f"2-hop")
    data = list(
        zip(data["questions"][:num], data["answers"][:num], data["entities"][:num])
    )
    triplets = metaqa.get_triplets()

    project_root = Path.home() / "DepCacheV2"
    output_file = project_root / f"database/log/2-hop_qa_depth{depth}_num{num}.json"

    igraph_env = IGraph(dataset="metaqa", triplets=triplets)
    results = []
    all_time = -time.time()

    for i, (question, answer, entities) in tqdm(
        enumerate(data), desc=f"Processing 2-hop QA samples"
    ):

        if not entities:
            print(f"样本 {i} 没有实体，跳过")
            continue

        question_time = -time.time()

        reasoning_paths_simple = igraph_env.subgraph_extraction_to_paths_simple(
            entities, depth
        )
        reasoning_paths_simple = igraph_env.convert_node_lists_to_paths(
            reasoning_paths_simple
        )

        reasoning_paths_dfs = igraph_env.subgraph_extraction_to_paths_dfs(
            entities, depth
        )
        reasoning_paths_dfs = igraph_env.convert_triplet_lists_to_paths(
            reasoning_paths_dfs
        )

        question_time += time.time()

        results.append(
            {
                "id": i,
                "question": question,
                "entities": entities,
                "answers": answer,
                "reasoning_paths_simple": reasoning_paths_simple,
                "reasoning_paths_dfs": reasoning_paths_dfs,
                "retrieval_time": question_time,
            }
        )
        save_json(file_path=output_file, data=results)

    all_time += time.time()
    results.append(
        {
            "time": all_time,
            "average_time": all_time / len(results),
        }
    )
    save_json(file_path=output_file, data=results)


def test_igraph_basic():
    triplets = [
        ("A", "rel", "B"),
        ("B", "rel", "C"),
        ("A", "rel", "C"),
        ("C", "rel", "D"),
    ]
    igraph_env = IGraph(dataset="test_graph", triplets=triplets)

    print(f"{igraph_env.entities_num()=}")
    print(f"{igraph_env.triplets_num()=}")

    igraph_env.upsert_triplets([("D", "rel1", "E")])
    igraph_env.upsert_triplets([("A", "rel", "B")])
    igraph_env.upsert_triplets([("A", "rel2", "D")])
    print(f"{igraph_env.entities_num()=}")
    print(f"{igraph_env.triplets_num()=}")

    igraph_env.delete_triplets([("A", "rel", "B")])
    igraph_env.delete_triplets([("D", "rel1", "E")])
    print(f"{igraph_env.entities_num()=}")
    print(f"{igraph_env.triplets_num()=}")


if __name__ == "__main__":

    test_igraph_basic()
    test_subgraph_extraction_to_paths(depth=2, num=5)
