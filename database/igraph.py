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


# To Do List
# 1. 检查起始实体是否存在于图中

# 对于每个输入实体：

# 判断该实体是否存在于图节点中
# 如果不存在，可以跳过该实体，并输出提示信息

# 例如：

# if entity not in self.graph.vs["name"]:
#     ...
# 考察点
# 图节点查找
# 异常输入处理
# 2. 找到实体对应的起始节点 index

# 对于存在的实体：

# 通过图节点名称找到其在图中的索引
# 作为 DFS 的起点

# 例如：

# start_idx = self.graph.vs.find(name=entity).index
# 考察点
# 图节点属性索引
# 名称到节点 id 的映射
# 3. 使用 DFS 遍历图，抽取路径

# 实现一个递归 DFS 函数，例如：

# def dfs(current_node, path_so_far, depth):
#     ...

# 需要完成：

# 从当前节点出发，获取相邻边
# 找到相邻节点
# 构造当前可扩展的三元组
# 递归访问下一层节点
# 考察点
# DFS 基本实现
# 递归搜索
# 图遍历逻辑
# 4. 构造路径中的三元组表示

# 对于每一条边，需要构造 triplet：

# (
#     source_entity_name,
#     relation_name,
#     target_entity_name,
# )

# 其中：

# source_entity_name 来自起点边一端
# relation_name 来自边属性 name
# target_entity_name 来自边另一端

# 例如：

# triple = (
#     self.graph.vs[src]["name"],
#     edge["name"] if "name" in self.graph.es.attribute_names() else None,
#     self.graph.vs[tgt]["name"],
# )
# 考察点
# 图中边与节点属性读取
# triplet 数据结构构建
# 5. 控制最大 hop 深度

# DFS 不应无限扩展，需要满足：

# 当 depth >= hop 时停止继续向下搜索
# 当前已有路径应加入结果集
# 考察点
# 多跳检索控制
# 搜索边界条件设计
# 6. 避免路径中重复使用同一个 triplet

# 为了避免循环路径或无意义重复扩展，需要检查：

# 当前 triplet 是否已经在 path_so_far 中
# 若已出现，则不再加入当前路径

# 例如：

# if triple not in path_so_far:
#     ...
# 考察点
# 路径去重
# 图搜索中的环处理
# 7. 保存完整路径

# 当出现以下情况时，应将当前路径加入结果：

# 当前节点没有可继续扩展的邻居
# 当前深度达到 hop
# 当前路径非空

# 例如：

# if not neighbors or depth >= hop:
#     if path_so_far:
#         completed_paths.append(path_so_far)
#     return
# 考察点
# 路径结束条件
# 搜索结果组织
# 8. 返回每个实体对应的所有路径

# 最终返回：

# entity_path_dict[entity] = completed_paths

# 整体格式为：

# Dict[str, List[List[Tuple[str, str, str]]]]
# 考察点
# 多实体批量处理
# 结果格式设计
# 输入输出示例
# 输入
# entities = ["Tom Hanks"]
# hop = 2
# 可能输出
# {
#     "Tom Hanks": [
#         [
#             ("Tom Hanks", "acted_in", "Forrest Gump"),
#             ("Forrest Gump", "directed_by", "Robert Zemeckis")
#         ],
#         [
#             ("Tom Hanks", "acted_in", "Cast Away")
#         ]
#     ]
# }
# 验收标准

# 补全后，函数应满足：

# 能处理多个起始实体
# 能正确做 DFS 遍历
# 能限制最大 hop
# 能输出 triplet path
# 返回格式正确
# 对不存在实体有合理处理
# 不出现明显重复路径
###########################################

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

##################################################

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
