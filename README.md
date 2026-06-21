# RAG 面试题项目

本项目是一套 **RAG（检索增强生成）面试题**。面试者需要在已经搭好的工程骨架上，补全两条
RAG 检索链路的核心代码，让它们能在 **RGB 数据集**上跑通，并理解每个模块的职责。

两道主线考题：

1. **考题 1 — VectorRAG**：构造向量知识库，跑通基于 FAISS 的向量检索问答。
2. **考题 2 — GraphRAG**：构造图知识库，跑通基于知识图谱的多跳检索问答。
3. **考题 3（可选）— Benchmark**：用 RAGAS 对上面两条链路的输出做自动评测。

> 说明：仓库当前版本在考题函数里**保留了一份参考实现**，并在每个函数上方写明了
> 「该做什么、考察什么」。正式面试时可以把这些函数体挖空成 stub，让面试者从零补全。

---

## 一、整体架构

```
                     ┌─────────────────────────────────────────────┐
   RGB 数据集         │  dataset/rgb.py                              │
  (问题/答案/语料/三元组) │   get_rgb_info()  → texts/questions/answers │
                     │   get_triplets()  → 知识图谱三元组            │
                     └───────────────┬─────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                         ▼
┌──────────────────────┐                          ┌────────────────────────────────┐
│  考题1 VectorRAG       │                          │  考题2 GraphRAG                  │
│                      │                          │                                │
│  FaissDB (向量库)      │                          │  IGraph (知识图谱)               │
│   建库 → 向量检索 top-k │                          │   实体召回(EntitiesDB)            │
│         │             │                          │    → 多跳子图抽取(IGraph DFS)     │
│         ▼             │                          │     → 语义剪枝(Pruning)           │
│   LLM 基于检索文本生成   │                          │      → LLM 基于路径生成            │
└──────────┬───────────┘                          └───────────────┬────────────────┘
           │                                                       │
           └──────────────────────┬────────────────────────────────┘
                                  ▼
                    ragas 格式 json 日志 → benchmark.py (考题3) → csv
```

两条链路共用底层基建：**embedding（DashScope）+ LLM（Qwen/Zhipu）+ FAISS + 计时/日志工具**。

---

## 二、目录与文件职责

### `dataset/` 数据集
| 文件 | 关键类/函数 | 职责 |
| --- | --- | --- |
| `rgb.py` | `get_rgb_info(file="en")`、`get_triplets()` | **本项目实际使用的数据集**。读取 `RGB/en.json`（300 条问答 + 正/负向语料），以及 `RGB/rgb_triplets.json`（74394 条知识图谱三元组）。 |
| `metaqa.py` | `get_metaqa_info()`、`get_triplets()` | MetaQA 数据集加载（多跳 QA + KB），GraphRAG 可选数据源。 |
| `freebase / cwq / webqsp / webquestions / grailqa / multihop / dragonball.py` | — | 其它 KBQA 数据集的预处理脚本，本面试不强制使用。 |
| `RGB/` | — | RGB 原始数据与三元组。`en.json` 为主问答文件。 |

### `database/` 存储层
| 文件 | 关键类 | 职责 |
| --- | --- | --- |
| `faissdb.py` | **`FaissDB`** | **考题 1 主体**。文本向量库：编码文本 → 写入 FAISS `IndexFlatIP` → 相似度检索 top-k。支持索引持久化与 GPU。 |
| `igraph.py` | **`IGraph`** | **考题 2 主体之一**。基于 `igraph` 的有向知识图谱：三元组增删、实体/三元组查询、多跳子图抽取（simple paths / DFS）、路径字符串化。 |
| `entitiesdb.py` | `EntitiesDB` | GraphRAG 的**实体召回**库：把图中所有实体做向量化，按问题检索最相似的若干实体作为图检索入口（已实现）。 |

### `utils/` 基建
| 文件 | 关键类/函数 | 职责 |
| --- | --- | --- |
| `embedding.py` | `EmbeddingEnv` | DashScope `text-embedding-v4` 文本向量化封装（L2 归一化、批处理、重试）。 |
| `remote_llm.py` | `LLMEnv` | 统一的 LLM 封装，支持 `qwen`（DashScope OpenAI 兼容）和 `zhipu`（GLM）后端，提供 `prompt_complete` / `prompt_complete_batch`。 |
| `prompts.py` | `QA_SYSTEM`、`QA_USER` | QA 提示词模板（基于检索上下文作答）。 |
| `pruning.py` | `Pruning` | **语义剪枝**：用 embedding + FAISS 计算「问题 vs 候选路径」相似度，保留 top-k 路径，控制送入 LLM 的上下文规模。 |
| `base.py` | `checkanswer`、`get_accuracy`、`read/save_json`、`get_base_dir` 等 | 通用工具：答案判分、准确率统计、json 读写、路径管理、彩色打印。 |
| `timer.py` / `logger.py` | `Timer` / `Logger` | 分阶段计时统计、控制台+文件日志。 |

### `rag/` 检索链路
| 文件 | 关键类/函数 | 职责 |
| --- | --- | --- |
| `vectorrag_faiss.py` | `prepare_faiss_db`、`vectorrag_with_faiss` | **考题 1 流程**。建向量库 → 逐题检索 → 拼上下文 → 调 LLM → 判分 → 落盘 ragas 格式 json。 |
| `graphrag_pipeline.py` | **`GraphRAGPipeline`** | **考题 2 流程**。三级流水线（3 线程 + 3 队列）：子图抽取 → 路径剪枝 → 生成。 |
| `graphrag_pipeline_process.py` | `GraphRAGPipelineProcess` | GraphRAG 的多进程版本变体（参考，依赖较多，不作为主考题）。 |
| `benchmark.py` | `convert_log_to_ragas_samples`、`main` | **考题 3**。用 RAGAS（faithfulness / context_precision / context_recall）评测产出日志。 |
| `base.py` | `RAG`（抽象基类） | RAG 的抽象接口（`retrieve` / `generate` / `run`），描述统一范式。 |

---

## 三、环境准备

### 1. 安装依赖
```bash
pip install -r requirements.txt
```
> 注：`requirements.txt` 含 `faiss-cpu`、`dashscope`、`openai`、`zai`、`igraph` 等。
> RAGAS 评测（考题 3）还需要额外安装 `ragas`、`datasets`、`pandas`、`langchain-core`。

### 2. 配置 API Key
本项目调用云端 embedding 与 LLM，需要配置对应的 Key（**推荐用环境变量，不要写死在代码里**）：

```bash
# Qwen / DashScope（embedding + qwen LLM 都用这个）
export DASHSCOPE_API_KEY="sk-xxxxxxxx"

# 若使用智谱 GLM 后端，额外配置
export ZHIPU_API_KEY="xxxxxxxx"
```

> ⚠️ 安全提示：仓库里几个脚本的 `--api_key` 默认值塞了一个明文 key。请改为从环境变量读取，
> 并轮换掉已泄露的 key，不要把真实 key 提交到 git。

---

## 四、考题 1 — VectorRAG（向量知识库）

### 目标
构造文本向量库，对每个问题检索最相关的 top-k 文本作为上下文，让 LLM 基于上下文作答，并统计准确率。

### 需要补全的代码
1. **`database/faissdb.py` 中 `FaissDB` 的 4 个核心函数**（函数上方有逐条要求）：
   - `get_embedding(query, is_query=True)`：文本→向量，支持单条/批量；query 侧加检索指令前缀；返回 `np.float32`。
   - `insert(embeddings)`：向量写入 FAISS，处理 float32 与一维→二维。
   - `generate_embedding_and_insert()`：按 `batch_size` 分批建库（tqdm 进度）。
   - `search(queries, top_k=5)`：检索 top-k，id→原文映射，处理非法 id（-1），统一单/多 query 返回结构。
2. **`rag/vectorrag_faiss.py` 中 `prepare_faiss_db`**：从 `rgb_data` 整理 `texts/metadata/ids` → 初始化 `FaissDB` → 挂回 metadata/ids。文件中 `vectorrag_with_faiss` 的检索-生成主循环已用注释标出每一步。

### 考察点
query/doc 双塔输入差异、批量编码、FAISS 输入 shape 要求、top-k 结果解析与 id 映射、检索-生成-评测完整闭环。

### 运行
```bash
python -m rag.vectorrag_faiss --backend qwen --top_k 3
```
首次运行会建库（在 `database/faiss_db/faiss_rgb_data/` 下生成 `.index` 和 `_meta.npy`）；
之后 `FaissDB(overwrite=False)` 会直接复用索引。结果 json 落在 `rag/log/`。

---

## 五、考题 2 — GraphRAG（图知识库）

### 目标
基于 RGB 三元组构造知识图谱，对每个问题：召回相关实体 → 从实体出发做多跳子图抽取 → 语义剪枝路径 → 让 LLM 基于推理路径作答。

### 需要补全的代码
1. **`database/igraph.py` 中 `IGraph.subgraph_extraction_to_paths_dfs(entities, hop)`**（考题 2-A）：
   用 DFS 从给定实体出发抽取 `hop` 跳以内的所有路径，返回 `Dict[entity, List[List[triplet]]]`。
   函数上方有 8 步详细要求（节点查找、index 映射、DFS、triplet 构造、hop 控制、路径去重、终止条件、多实体返回）以及输入输出示例。
2. **`rag/graphrag_pipeline.py` 中 `GraphRAGPipeline` 的 3 个 worker**（考题 2-B/C/D）：
   - `_subgraph_worker()`：调 `IGraph` 的 DFS 抽取 + 路径字符串化，结果入 `prune_q`。
   - `_pruning_worker()`：调 `Pruning.semantic_pruning_triplets_batch` 做语义剪枝，整理成 context 入 `gen_q`。
   - `_generation_worker()`：跨批次累积 context → 批量调 LLM → 判分 → early-stop → 写 ragas 记录。
   每个函数的 docstring 里有分步要求。注意三个队列靠 `task_done()` + `join()` 配平，漏调会死锁。

### 考察点
图节点/边属性读取、DFS 多跳遍历与边界控制、路径去重、生产者-消费者队列、语义剪枝、跨批次上下文累积、提前停止。

### 🌟 加分项 — 更优的子图检索算法
当前的 `subgraph_extraction_to_paths_simple` / `_dfs` 都是「从实体出发、穷举 hop 跳以内所有路径」的
暴力遍历，在高度数、大 hop 的真实图谱上会路径爆炸。加分项要求面试者思考：

> 除了 DFS / simple-path 这类穷举遍历，还有哪些更好的图查询 / 检索算法？

在 `database/igraph.py` 中预留了空函数 **`subgraph_extraction_to_paths_advanced(entities, hop)`**（`raise NotImplementedError`），
面试者可自选一种或多种思路实现。一些可参考方向（提示，不限于此）：

| 思路 | 要点 |
| --- | --- |
| **BFS 分层遍历** | 按 hop 逐层扩展，便于每层剪枝 / 早停 |
| **双向搜索 (bidirectional)** | 已知问题实体与候选答案实体时，两端同时扩展、中间汇合，远快于单向枚举 |
| **最短路 / k-最短路** | 只取实体对间最短的若干条路径（Dijkstra / k-shortest paths） |
| **带权扩展 + beam search** | 每跳用 embedding/LLM 给「关系-邻居」打分，只保留 top-b 分支，把检索与剪枝合一 |
| **Personalized PageRank / RWR** | 以问题实体为种子随机游走，取访问概率最高的子图（许多 GraphRAG/KBQA 系统的做法） |
| **Steiner 树 / 最小连通子图** | 求能连通多个问题实体的最小子图 |

建议返回结构与 `_dfs` 对齐（`Dict[entity, List[List[triplet]]]`），即可直接复用
`convert_triplet_lists_to_paths` 做路径字符串化，并无缝接入 `GraphRAGPipeline._subgraph_worker`。

**考察点**：图算法选型与复杂度权衡、召回与上下文规模的平衡、如何把语义相关性融入图遍历本身
（而非先全量枚举再事后剪枝）。请在作答时说明选型理由与复杂度分析。

### 数据流（`run_batch`）
按 batch 取问题 → `EntitiesDB.search` 召回 top-`ent` 实体 → 实体按 `ratio` 拆成两批分别投入 `subgraph_q`
→ 子图抽取 → 剪枝（top-`pruning`）→ 生成。同一问题的两批 context 在生成阶段累积。

### 运行
```bash
python -m rag.graphrag_pipeline --backend qwen --dataset rgb --num 10 \
    --ent 10 --hop 2 --pruning 30 --batch_size 8
```
首次会用 `IGraph` 建图（落 `igraph_db/*.gml`）并用 `EntitiesDB` 建实体向量库。结果 json + log 落在 `rag/log/`。

---

## 六、考题 3（可选）— Benchmark（RAGAS 评测）

### 目标
把考题 1/2 落盘的 ragas 格式 json，用 RAGAS 三大指标做自动评测。

### 需要补全的代码
`rag/benchmark.py` 中把现有 `main()` 的流程抽象成 `benchmark_rag(input_json, output_dir, ...)`（函数上方有分步要求）：
加载 json → `convert_log_to_ragas_samples` → `build_dataset` → `evaluate`（faithfulness / context_precision / context_recall）→ 输出 summary + details csv。

### 运行
```bash
# 修改 benchmark.py 顶部的 INPUT_JSON 指向你的产出 json，然后：
python -m rag.benchmark
```
结果落在 `rag/eval_results/`（`*_summary.csv`、`*_details.csv`）。

---

## 七、答案判分说明

`utils/base.py::checkanswer(prediction, ground_truth)` 用**子串匹配**判分：
- `ground_truth` 是嵌套 list，外层每个元素是「必须命中的一个答案点」，内层 list 是「该答案点的同义表述（命中任一即可）」。
- `get_accuracy` 要求某条样本的所有答案点都命中才算正确。

RGB 的 answer 形如 `[["January 2 2022", "Jan 2, 2022", ...]]`，即单个答案点 + 多种写法。

---

## 八、快速上手顺序（建议给面试者）

1. 读 `dataset/rgb.py` 和 `utils/`，搞清数据长什么样、有哪些现成工具。
2. 做考题 1：先补 `FaissDB` 四个函数，单独 `python -m database.faissdb` 用内置 demo 跑通，再跑 `rag.vectorrag_faiss`。
3. 做考题 2：先补 `IGraph` 的 DFS（`python -m database.igraph` 有基础测试），再补三个 worker，跑 `rag.graphrag_pipeline`。
4. 做考题 3：补 `benchmark_rag`，对两条链路的产出做评测、对比指标。
