# Paper Agent V1 — 实现总 Plan

---

## 一、模块拆分与依赖关系

共 8 个模块，按依赖顺序排列（上游先实现）：

```
M1 core        ← 无依赖（数据结构 + 配置）
M2 parsing      ← M1
M3 indexing     ← M1, M2
M4 retrieval    ← M1, M3
M5 agent        ← M1, M4
M6 generation   ← M1
M7 api+service  ← M5, M6
M8 evaluation   ← M7
```

实现顺序：M1 → M2 → M3 → M4 → M6 → M5 → M7 → M8

---

## 二、每个模块的输入 / 输出 / 职责

### M1: core（基础层）

| 文件                    | 职责             | 输入            | 输出                                                                                                                     |
| ----------------------- | ---------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `app/core/schemas.py`   | 全局数据结构定义 | —               | `PaperMetadata`, `SectionChunk`, `AgentState`, `AgentResponse`, `IntentType(Enum)`                                       |
| `app/core/config.py`    | 配置管理         | 环境变量 / .env | `Settings` (Pydantic BaseSettings): GROBID URL, embedding model path, FAISS index path, LLM API key, reranker model path |
| `app/core/constants.py` | 常量定义         | —               | `SECTION_TYPES`, `INTENT_TYPES`, `DEFAULT_TOP_K`, prompt 模板路径                                                        |
| `app/core/logger.py`    | 日志             | —               | 统一 logger                                                                                                              |

**schemas.py 核心定义：**

```python
from dataclasses import dataclass, field
from enum import Enum

class IntentType(str, Enum):
    PAPER_SEARCH = "paper_search"       # 模糊找论文
    PAPER_READING = "paper_reading"     # 指定论文解读
    SECTION_QA = "section_qa"           # 章节追问

class AgentStateEnum(str, Enum):
    ROUTING = "routing"
    SEARCHING = "searching"
    LOCKED = "locked"
    READING = "reading"
    ANSWERING = "answering"

@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str
    keywords: list[str]
    section_titles: list[str]

@dataclass
class SectionChunk:
    chunk_id: str
    paper_id: str
    section_type: str           # 归一化类型: method / experiment / conclusion / other ...
    section_title: str          # 原始标题，如 "3.1 Model Architecture"
    section_path: str           # 完整路径，如 "3 Methodology > 3.1 Model Architecture"
    text: str
    page_start: int
    page_end: int
    order_in_paper: int         # 在全文中的全局顺序，用于邻接窗口
    level: int                  # 层级深度，0 = 顶级 section，1 = subsection
    parent_chunk_id: str | None # 父 chunk id，顶级为 None
    granularity: str            # "coarse"（顶级大节聚合）| "fine"（子节/段落）

@dataclass
class AgentState:
    current_state: AgentStateEnum = AgentStateEnum.ROUTING
    candidate_papers: list[str] = field(default_factory=list)
    current_paper_id: str | None = None
    current_focus_section: str | None = None
    last_intent: IntentType | None = None

@dataclass
class Evidence:
    text: str
    paper_id: str
    section_type: str
    section_title: str
    page: int | None = None

@dataclass
class AgentResponse:
    answer: str
    evidences: list[Evidence]
    intent: IntentType
    updated_state: AgentState
    candidate_papers: list[PaperMetadata] | None = None  # 找论文时返回
```

---

### M2: parsing（PDF 解析层）

| 文件                              | 职责                      | 输入               | 输出                                          |
| --------------------------------- | ------------------------- | ------------------ | --------------------------------------------- |
| `app/parsing/grobid_runner.py`    | 调用 GROBID REST API      | `pdf_path: str`    | `tei_xml: str` (原始 TEI/XML)                 |
| `app/parsing/tei_parser.py`       | 解析 TEI/XML 为结构化数据 | `tei_xml: str`     | `PaperMetadata` + `list[SectionChunk]`        |
| `app/parsing/paper_normalizer.py` | section_type 归一化       | 原始 section title | 标准 section_type (`method`/`experiment`/...) |

**grobid_runner.py 核心逻辑：**

```
1. 读取 PDF 文件为 bytes
2. POST 到 GROBID /api/processFulltextDocument
3. 返回 TEI/XML string
4. 异常处理：GROBID 服务不可用 / PDF 解析失败
```

**tei_parser.py 核心逻辑：**

```
1. 用 lxml 解析 TEI/XML
2. 从 <teiHeader> 提取: title, authors, abstract, keywords, year, venue
3. 从 <body> 递归遍历所有嵌套 <div>，构建 section tree:
   a. 顶级 <div>（body 的直接子节点）→ 生成 coarse chunk
      - coarse chunk 的 text = 该 div 下所有 <p> 的递归文本（含子 div 的所有段落）
      - level = 0, parent_chunk_id = None, granularity = "coarse"
   b. 子 <div>（嵌套在顶级 div 内）→ 生成 fine chunk
      - fine chunk 的 text = 只取该 <div> 直接子 <p>（避免重复，子 div 各自负责自己）
      - level = 1（或更深），parent_chunk_id = 父 coarse chunk 的 chunk_id
      - granularity = "fine"
   c. section_path 格式: 父标题 + " > " + 当前标题，如 "3 Methodology > 3.1 Model"
   d. order_in_paper 按全局遍历顺序递增（coarse 和 fine 共用一个全局计数器）
4. 过滤规则：
   - references / acknowledgment / appendix → section_type = "other"，依然入库但标记低优先级
   - text 为空的 chunk 不生成
5. 生成 paper_id (基于 title hash 或 PDF 文件名)
```

**paper_normalizer.py 核心逻辑：**

```
映射表 + 模糊匹配：
"Introduction" / "1. Introduction" / "INTRODUCTION" → "introduction"
"Method" / "Methodology" / "Approach" / "Our Method" / "Proposed Method" → "method"
"Experiment" / "Experiments" / "Experimental Setup" / "Evaluation" → "experiment"
"Related Work" / "Background" → "related_work"
"Conclusion" / "Conclusions" / "Summary" → "conclusion"
"Abstract" → "abstract"
其他 → "other"
```

---

### M3: indexing（索引层）

| 文件                            | 职责                 | 输入                         | 输出                                   |
| ------------------------------- | -------------------- | ---------------------------- | -------------------------------------- |
| `app/indexing/vector_store.py`  | FAISS 向量索引封装   | embedding vectors + metadata | FAISS index (持久化到 `data/indexes/`) |
| `app/indexing/bm25_index.py`    | BM25 稀疏索引封装    | 文本 + doc_ids               | BM25 index (pickle 持久化)             |
| `app/indexing/paper_index.py`   | 论文级索引构建与检索 | `list[PaperMetadata]`        | 论文级 dense + BM25 索引               |
| `app/indexing/section_index.py` | 章节级索引构建与检索 | `list[SectionChunk]`         | 章节级 dense + BM25 索引               |

**vector_store.py 接口：**

```python
class VectorStore:
    def __init__(self, index_path: str, embedding_model: str)
    def add(self, texts: list[str], metadatas: list[dict]) -> None
    def search(self, query: str, top_k: int = 10) -> list[dict]  # [{text, metadata, score}]
    def save(self) -> None
    def load(self) -> None
```

**bm25_index.py 接口：**

```python
class BM25Index:
    def __init__(self, index_path: str)
    def add(self, texts: list[str], doc_ids: list[str]) -> None
    def search(self, query: str, top_k: int = 10) -> list[dict]  # [{doc_id, score}]
    def save(self) -> None
    def load(self) -> None
```

**paper_index.py 接口：**

```python
class PaperIndex:
    def __init__(self, vector_store: VectorStore, bm25: BM25Index, metadata_store: dict)
    def build(self, papers: list[PaperMetadata]) -> None
    def search(self, query: str, top_k: int = 5) -> list[PaperMetadata]
    def get_by_id(self, paper_id: str) -> PaperMetadata | None
```

- dense 检索字段: `title + " " + abstract + " " + " ".join(keywords)`
- BM25 检索字段: 同上
- metadata_store: `dict[paper_id, PaperMetadata]` (SQLite 或 JSON 文件)

**section_index.py 接口：**

```python
class SectionIndex:
    def __init__(self, vector_store: VectorStore, bm25: BM25Index)
    def build(self, chunks: list[SectionChunk]) -> None
    def search(self, query: str, paper_id: str | None = None,
               target_sections: list[str] | None = None,
               top_k: int = 5) -> list[SectionChunk]
```

- dense 检索字段: `section_title + ": " + text`
- 支持按 paper_id 过滤、按 section_type 过滤

---

### M4: retrieval（检索策略层）

| 文件                                 | 职责                  | 输入                               | 输出                           |
| ------------------------------------ | --------------------- | ---------------------------------- | ------------------------------ |
| `app/retrieval/paper_retriever.py`   | 论文级检索策略        | query                              | `list[PaperMetadata]` (ranked) |
| `app/retrieval/section_retriever.py` | 章节级检索策略        | query + paper_id + target_sections | `list[SectionChunk]` (ranked)  |
| `app/retrieval/fusion.py`            | dense + BM25 分数融合 | 两路召回结果                       | 融合排序结果                   |
| `app/retrieval/reranker.py`          | reranker 封装         | query + candidates                 | reranked candidates            |

**fusion.py 核心逻辑：**

```
Reciprocal Rank Fusion (RRF):
  score(doc) = Σ 1 / (k + rank_i(doc))  for each retriever i
  k = 60 (常用默认值)
```

**reranker.py 接口：**

```python
class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3")
    def rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[int]  # reranked indices
```

**paper_retriever.py 核心流程：**

```
1. dense 检索 paper_index top_20
2. BM25 检索 paper_index top_20
3. RRF 融合 → top_10
4. reranker rerank → top_3
5. 返回 list[PaperMetadata]
```

**section_retriever.py 核心流程：**

```
1. 在 section_index 中按 paper_id 过滤
2. 如果有 target_sections, 按 section_type 过滤
3. dense + BM25 双路召回
4. RRF 融合
5. reranker rerank → top_k
6. 返回 list[SectionChunk]
```

---

### M5: agent（Agent 决策层）

| 文件                         | 职责             | 输入                    | 输出                 |
| ---------------------------- | ---------------- | ----------------------- | -------------------- |
| `app/agent/intent_router.py` | 意图识别         | user query + AgentState | `IntentType`         |
| `app/agent/state_manager.py` | 状态流转 + 回退  | IntentType + AgentState | updated `AgentState` |
| `app/agent/tools.py`         | 3 个工具函数封装 | 各工具参数              | 工具返回结果         |
| `app/agent/orchestrator.py`  | Agent 主循环     | user query + AgentState | `AgentResponse`      |

**intent_router.py 核心逻辑：**

```
使用 LLM 做 few-shot 分类:

system_prompt = """你是一个意图分类器。根据用户问题和当前会话状态，判断意图类型。
只返回以下三种之一: paper_search / paper_reading / section_qa

规则:
- 如果用户在找/搜索某篇论文 → paper_search
- 如果用户想了解已锁定论文的全貌 → paper_reading
- 如果用户针对已锁定论文的某个章节/方面追问 → section_qa
- 如果已锁定论文且用户问了一个新论文 → paper_search (触发回退)
"""

输入: query + state.current_paper_id + state.last_intent
输出: IntentType
```

**state_manager.py 状态转移表：**

```
当前状态        意图              新状态        附加动作
─────────────────────────────────────────────────────────────
ROUTING       paper_search    → SEARCHING     清空 candidate_papers
ROUTING       paper_reading   → LOCKED        需 current_paper_id 已存在
ROUTING       section_qa      → READING       需 current_paper_id 已存在

SEARCHING     (候选返回)       → SEARCHING     更新 candidate_papers
SEARCHING     (用户确认)       → LOCKED        设置 current_paper_id

LOCKED        paper_reading   → READING       —
LOCKED        section_qa      → READING       设置 current_focus_section
LOCKED        paper_search    → SEARCHING     清空 current_paper_id (回退)

READING       —               → ANSWERING     —
ANSWERING     —               → ROUTING       保留 paper_id (等下轮)

回退规则:
- 任意状态 + paper_search 且 current_paper_id 指向不同论文 → SEARCHING + 清空
- ANSWERING 后 + section_qa → READING (跳过 ROUTING 直接进)
```

**tools.py 三个工具函数：**

```python
def search_papers(query: str, top_k: int = 3) -> list[PaperMetadata]:
    """调用 paper_retriever"""

def get_paper_metadata(paper_id: str) -> PaperMetadata | None:
    """调用 paper_index.get_by_id"""

def retrieve_sections(paper_id: str, query: str,
                      target_sections: list[str] | None = None,
                      top_k: int = 5) -> list[SectionChunk]:
    """调用 section_retriever"""
```

**orchestrator.py 主循环（单轮）：**

```
def run_agent(query: str, state: AgentState) -> AgentResponse:
    # 1. 意图识别
    intent = intent_router.classify(query, state)

    # 2. 状态流转
    state = state_manager.transition(state, intent, query)

    # 3. 根据意图执行工具
    if intent == PAPER_SEARCH:
        candidates = tools.search_papers(query)
        state.candidate_papers = [p.paper_id for p in candidates]
        if len(candidates) == 1:
            state.current_paper_id = candidates[0].paper_id
            state = state_manager.force_state(state, LOCKED)
        return AgentResponse(
            answer=format_candidates(candidates),
            evidences=[],
            intent=intent,
            updated_state=state,
            candidate_papers=candidates
        )

    elif intent == PAPER_READING:
        # 受控全文解读: 先读 abstract + conclusion
        sections = tools.retrieve_sections(
            state.current_paper_id, query,
            target_sections=["abstract", "conclusion"]
        )
        # 补读 method + experiment
        sections += tools.retrieve_sections(
            state.current_paper_id, query,
            target_sections=["method", "experiment"]
        )
        evidences = build_evidences(sections)
        answer = generation.generate_paper_reading(query, evidences, state)
        state.current_state = ANSWERING
        return AgentResponse(answer=answer, evidences=evidences, ...)

    elif intent == SECTION_QA:
        # 章节定向: 根据 query 检索相关 section
        sections = tools.retrieve_sections(
            state.current_paper_id, query, top_k=3
        )
        evidences = build_evidences(sections)
        answer = generation.generate_section_qa(query, evidences, state)
        state.current_state = ANSWERING
        return AgentResponse(answer=answer, evidences=evidences, ...)
```

---

### M6: generation（生成层）

| 文件                                   | 职责              | 输入                      | 输出                     |
| -------------------------------------- | ----------------- | ------------------------- | ------------------------ |
| `app/generation/prompts.py`            | prompt 模板       | —                         | 模板字符串               |
| `app/generation/answer_builder.py`     | 调用 LLM 生成回答 | query + evidences + state | 结构化 answer string     |
| `app/generation/citation_formatter.py` | 引用格式化        | evidences                 | 带 [1][2] 标记的引用文本 |

**prompts.py 核心模板：**

```python
PAPER_READING_PROMPT = """你是一个学术论文解读助手。根据下面提供的论文片段，生成结构化解读。

<paper>
  <metadata>
    <title>{title}</title>
    <authors>{authors}</authors>
    <year>{year}</year>
  </metadata>
  {sections_xml}
</paper>

请按以下结构回答：
1. 研究问题：这篇论文要解决什么问题？
2. 核心贡献：论文的主要贡献是什么？
3. 方法概述：用了什么方法？
4. 实验结论：实验结果说明了什么？
5. 局限性：有什么局限？

要求：
- 每个要点必须基于上面提供的论文片段，不要编造
- 在关键陈述后标注证据来源 [section_type]
- 如果某个方面信息不足，明确说明"原文未提供足够信息"
"""

SECTION_QA_PROMPT = """你是一个学术论文解读助手。用户正在阅读一篇论文并对某个方面追问。

<paper>
  <metadata>
    <title>{title}</title>
  </metadata>
  {sections_xml}
</paper>

用户问题：{query}

请基于提供的论文片段回答：
- 直接回答用户问题
- 在关键陈述后标注来源 [section_type]
- 如果片段中没有足够信息，明确告知用户
"""

INTENT_CLASSIFICATION_PROMPT = """...(见 intent_router.py 描述)"""
```

**answer_builder.py 接口：**

```python
class AnswerBuilder:
    def __init__(self, llm_client)
    def generate_paper_reading(self, query: str, evidences: list[Evidence],
                                metadata: PaperMetadata) -> str
    def generate_section_qa(self, query: str, evidences: list[Evidence],
                             metadata: PaperMetadata) -> str
    def generate_candidates_summary(self, candidates: list[PaperMetadata],
                                     query: str) -> str
```

---

### M7: api + services（服务层）

| 文件                             | 职责         | 输入               | 输出                                     |
| -------------------------------- | ------------ | ------------------ | ---------------------------------------- |
| `app/services/ingest_service.py` | 论文入库流程 | PDF 目录路径       | 解析 + 索引完成                          |
| `app/services/qa_service.py`     | 问答主服务   | query + session_id | AgentResponse                            |
| `app/api/routes.py`              | FastAPI 路由 | HTTP request       | HTTP response                            |
| `app/api/deps.py`                | 依赖注入     | —                  | 全局单例 (indexes, llm_client, reranker) |

**ingest_service.py 流程：**

```
def ingest_papers(pdf_dir: str):
    1. glob 所有 *.pdf
    2. 对每个 PDF:
       a. grobid_runner.parse(pdf_path) → tei_xml
       b. tei_parser.parse(tei_xml) → PaperMetadata + list[SectionChunk]
       c. 保存解析结果到 data/parsed/{paper_id}.json
    3. 收集所有 PaperMetadata → paper_index.build(papers)
    4. 收集所有 SectionChunk → section_index.build(chunks)
    5. 保存索引到 data/indexes/
```

**qa_service.py 流程：**

```
class QAService:
    sessions: dict[str, AgentState]  # session_id → state

    def chat(self, query: str, session_id: str) -> AgentResponse:
        state = self.sessions.get(session_id, AgentState())
        response = orchestrator.run_agent(query, state)
        self.sessions[session_id] = response.updated_state
        return response
```

**routes.py 路由：**

```
POST /api/ingest         — 触发论文入库
POST /api/chat           — {query: str, session_id: str} → AgentResponse
GET  /api/papers         — 列出所有已入库论文
GET  /api/papers/{id}    — 获取单篇论文 metadata
```

---

### M8: evaluation（评测层）

| 文件                                 | 职责                         | 输入                            | 输出                                 |
| ------------------------------------ | ---------------------------- | ------------------------------- | ------------------------------------ |
| `app/evaluation/dataset_builder.py`  | 评测集加载                   | `data/eval/evalset.jsonl`       | `list[EvalSample]`                   |
| `app/evaluation/llm_judge.py`        | LLM-as-a-Judge 打分          | question + answer + gold_answer | faithfulness score + relevance score |
| `app/evaluation/baseline_compare.py` | flat RAG baseline + A/B 对比 | 评测集                          | 对比结果表                           |

**evalset.jsonl 格式：**

```json
{
  "question": "...",
  "intent_type": "paper_search",
  "gold_paper_ids": ["p1"],
  "gold_section_ids": [],
  "gold_answer": "..."
}
```

**llm_judge.py 核心逻辑：**

```python
JUDGE_PROMPT = """你是一个评测专家。请对下面的回答打分。

问题: {question}
参考答案: {gold_answer}
系统回答: {predicted_answer}

请分别对以下两个维度打 1-5 分:
1. Faithfulness (忠实度): 回答是否忠于原文，没有编造信息
2. Relevance (相关性): 回答是否切中问题，信息充分

输出格式:
faithfulness: X
relevance: X
"""

class LLMJudge:
    def __init__(self, judge_model: str = "gpt-4")
    def score(self, question, predicted, gold) -> dict[str, float]
    def batch_score(self, samples: list) -> pd.DataFrame
```

**baseline_compare.py 核心逻辑：**

```python
class FlatRAGBaseline:
    """flat chunking (500 tokens, 50 overlap) + top-5 检索 + 直接生成"""
    def __init__(self, pdf_dir, llm_client)
    def build_index(self)  # 不分层, 直接 chunk 全文
    def query(self, question: str) -> str

class ABComparison:
    def run(self, evalset, agent_service, baseline) -> pd.DataFrame
    # 输出: question | intent | agent_score | baseline_score | delta
```

---

## 三、状态流转全图

```
                    ┌─────────────────────────────────────────────┐
                    │                                             │
                    ▼                                             │
              ┌──────────┐                                        │
     query ──►│ ROUTING  │                                        │
              └────┬─────┘                                        │
                   │                                              │
          ┌────────┼────────┐                                     │
          │        │        │                                     │
    paper_search   │   section_qa                                 │
          │        │   (need paper_id)                            │
          ▼        │        │                                     │
    ┌───────────┐  │        │                                     │
    │ SEARCHING │  │        │                                     │
    └─────┬─────┘  │        │                                     │
          │        │        │                                     │
    candidates     │paper_reading                                 │
    returned       │(need paper_id)                               │
          │        │        │                                     │
    user confirms  │        │                                     │
    or auto-lock   │        │                                     │
          │        ▼        │                                     │
          │   ┌────────┐    │                                     │
          └──►│ LOCKED │◄───┘                                     │
              └───┬────┘                                          │
                  │                                               │
                  ▼                                               │
            ┌──────────┐                                          │
            │ READING  │                                          │
            └────┬─────┘                                          │
                 │                                                │
                 ▼                                                │
           ┌───────────┐    next query                            │
           │ ANSWERING │────────────────────────────────────────►─┘
           └───────────┘

回退路径 (虚线):
  LOCKED  ──paper_search(新论文)──► SEARCHING
  READING ──paper_search(新论文)──► SEARCHING
  ANSWERING ──section_qa──► READING (跳过 ROUTING)
```

---

## 四、文件级改动清单

### Phase 1: 基础设施 (Day 1-3)

```
新建  app/__init__.py
新建  app/core/__init__.py
新建  app/core/schemas.py          ← 所有 dataclass + enum
新建  app/core/config.py           ← Settings (Pydantic BaseSettings)
新建  app/core/constants.py        ← SECTION_TYPES 映射表, 默认参数
新建  app/core/logger.py           ← logging 配置
新建  app/parsing/__init__.py
新建  app/parsing/grobid_runner.py ← GROBID REST client
新建  app/parsing/tei_parser.py    ← TEI/XML → PaperMetadata + SectionChunk
新建  app/parsing/paper_normalizer.py ← section title → section_type 映射
新建  requirements.txt             ← 所有依赖
新建  .env.example                 ← 环境变量模板
新建  docker-compose.yml           ← GROBID 服务容器
新建  scripts/ingest_papers.py     ← 入库脚本入口
新建  data/raw_pdfs/.gitkeep
新建  data/parsed/.gitkeep
新建  data/indexes/.gitkeep
新建  data/eval/.gitkeep
```

### Phase 2: 索引层 (Day 4-6)

```
新建  app/indexing/__init__.py
新建  app/indexing/vector_store.py    ← FAISS + BGE embedding 封装
新建  app/indexing/bm25_index.py      ← BM25 (rank_bm25) 封装
新建  app/indexing/paper_index.py     ← 论文级索引: build + search + get_by_id
新建  app/indexing/section_index.py   ← 章节级索引: build + search (支持过滤)
新建  scripts/build_indexes.py        ← 索引构建脚本
```

### Phase 3: 检索层 (Day 7-8)

```
新建  app/retrieval/__init__.py
新建  app/retrieval/fusion.py          ← RRF 融合
新建  app/retrieval/reranker.py        ← BGE-reranker 封装
新建  app/retrieval/paper_retriever.py ← 论文级检索: dense+BM25+RRF+rerank
新建  app/retrieval/section_retriever.py ← 章节级检索: 同上 + paper_id/section_type 过滤
```

### Phase 4: Agent 层 (Day 8-12)

```
新建  app/agent/__init__.py
新建  app/agent/intent_router.py    ← LLM few-shot 意图分类
新建  app/agent/state_manager.py    ← 状态转移表 + 回退逻辑
新建  app/agent/tools.py            ← search_papers / get_paper_metadata / retrieve_sections
新建  app/agent/orchestrator.py     ← run_agent 主循环
新建  app/generation/__init__.py
新建  app/generation/prompts.py     ← 所有 prompt 模板
新建  app/generation/answer_builder.py   ← LLM 调用 + 结构化生成
新建  app/generation/citation_formatter.py ← evidence → [1] 引用标记
```

### Phase 5: 服务层 + API (Day 13-15)

```
新建  app/services/__init__.py
新建  app/services/ingest_service.py  ← 论文入库编排
新建  app/services/qa_service.py      ← 问答主服务 (含 session 管理)
新建  app/api/__init__.py
新建  app/api/routes.py               ← FastAPI 路由
新建  app/api/deps.py                 ← 依赖注入 (全局单例)
新建  app/main.py                     ← FastAPI app 入口
```

### Phase 6: 评测 (Day 16-18)

```
新建  app/evaluation/__init__.py
新建  app/evaluation/dataset_builder.py   ← 加载 evalset.jsonl
新建  app/evaluation/llm_judge.py         ← LLM-as-a-Judge
新建  app/evaluation/baseline_compare.py  ← flat RAG baseline + A/B 对比
新建  data/eval/evalset.jsonl             ← 50 条手工标注 (手动创建)
新建  scripts/build_evalset.py            ← 评测集构建辅助脚本
新建  scripts/run_eval.py                 ← 跑评测主脚本
```

### Phase 7: Demo (Day 19-20)

```
新建  app/demo.py                     ← Gradio Demo 入口
修改  README.md                       ← 项目说明
```

**共计: ~40 个新建文件**

---

## 五、测试清单

### 5.1 单元测试

```
tests/
├── test_parsing/
│   ├── test_grobid_runner.py       ← mock GROBID API, 测试正常/异常
│   ├── test_tei_parser.py          ← 用真实 TEI/XML fixture 测解析正确性
│   └── test_paper_normalizer.py    ← 测 section title → type 映射覆盖率
├── test_indexing/
│   ├── test_vector_store.py        ← 测 add/search/save/load 正确性
│   ├── test_bm25_index.py          ← 测 add/search 正确性
│   ├── test_paper_index.py         ← 测 build/search/get_by_id
│   └── test_section_index.py       ← 测 build/search + paper_id 过滤 + section_type 过滤
├── test_retrieval/
│   ├── test_fusion.py              ← 测 RRF 融合排序正确性
│   ├── test_reranker.py            ← mock reranker, 测接口正确性
│   ├── test_paper_retriever.py     ← 端到端: query → ranked papers
│   └── test_section_retriever.py   ← 端到端: query + paper_id → ranked sections
├── test_agent/
│   ├── test_intent_router.py       ← mock LLM, 测 3 类意图分类
│   ├── test_state_manager.py       ← 测所有状态转移路径 + 回退
│   ├── test_tools.py               ← mock index, 测工具返回格式
│   └── test_orchestrator.py        ← 端到端: query + state → response
├── test_generation/
│   ├── test_answer_builder.py      ← mock LLM, 测 prompt 组装 + 输出解析
│   └── test_citation_formatter.py  ← 测引用标记格式
└── test_api/
    └── test_routes.py              ← FastAPI TestClient, 测 /chat /ingest 路由
```

### 5.2 集成测试

| 测试                   | 描述                                       | 验证点                                     |
| ---------------------- | ------------------------------------------ | ------------------------------------------ |
| `test_ingest_pipeline` | 放入 1 篇真实 PDF, 走完解析+索引流程       | 解析出 metadata, sections 非空; 索引可检索 |
| `test_search_flow`     | 入库后查询 "self-rag"                      | 返回候选论文, 包含正确论文                 |
| `test_reading_flow`    | 锁定论文后 "讲讲这篇论文"                  | 返回结构化解读, evidences 非空             |
| `test_section_qa_flow` | 锁定论文后 "方法部分讲讲"                  | 返回方法相关内容                           |
| `test_multi_turn`      | 找论文 → 锁定 → 解读 → 追问方法 → 追问实验 | 状态正确流转, 每轮返回合理                 |
| `test_state_rollback`  | 在解读中途问一篇新论文                     | 状态回退到 SEARCHING                       |

### 5.3 验收测试（手动）

- [ ] GROBID Docker 启动正常
- [ ] 20+ 篇 PDF 入库成功率 > 90%
- [ ] Gradio Demo 可正常对话
- [ ] 三种场景各跑 3 个 case, 回答质量人工通过
- [ ] 多轮追问 5 轮以上不崩溃
- [ ] 评测脚本跑通, 输出 A/B 对比表

---

## 六、依赖清单 (requirements.txt)

```
# Core
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0
python-multipart>=0.0.6

# PDF Parsing
lxml>=4.9.0
requests>=2.31.0

# Embedding & Indexing
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
numpy>=1.24.0

# Reranker
FlagEmbedding>=1.2.0

# LLM Client
openai>=1.6.0

# Generation
tiktoken>=0.5.0

# Evaluation
pandas>=2.1.0

# Demo
gradio>=4.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.25.0
```

---

## 七、给 Codex 的执行 Prompt

以下 prompt 按 Phase 分段，每段可独立交给 Codex 执行。

---

### Prompt 1: Phase 1 — 基础设施 + PDF 解析

```
你在一个 Python 项目 paper-agent/ 下工作。这是一个面向学术论文 PDF 的检索与解读 Agent。

请完成以下任务:

1. 创建项目基础结构:
   - app/__init__.py (空)
   - app/core/__init__.py
   - app/core/schemas.py: 定义以下 dataclass 和 enum (使用 Python dataclasses):
     - IntentType(str, Enum): paper_search / paper_reading / section_qa
     - AgentStateEnum(str, Enum): routing / searching / locked / reading / answering
     - PaperMetadata: paper_id(str), title(str), authors(list[str]), year(int|None), venue(str|None), abstract(str), keywords(list[str]), section_titles(list[str])
     - SectionChunk: chunk_id(str), paper_id(str), section_type(str), section_title(str), text(str), page_start(int), page_end(int), order_in_paper(int)
     - AgentState: current_state(AgentStateEnum=ROUTING), candidate_papers(list[str]), current_paper_id(str|None), current_focus_section(str|None), last_intent(IntentType|None)
     - Evidence: text(str), paper_id(str), section_type(str), section_title(str), page(int|None)
     - AgentResponse: answer(str), evidences(list[Evidence]), intent(IntentType), updated_state(AgentState), candidate_papers(list[PaperMetadata]|None)
   - app/core/config.py: 使用 pydantic-settings 定义 Settings 类, 字段: grobid_url(str, default="http://localhost:8070"), embedding_model(str, default="BAAI/bge-large-zh-v1.5"), reranker_model(str, default="BAAI/bge-reranker-v2-m3"), llm_api_key(str), llm_base_url(str), llm_model(str, default="deepseek-chat"), data_dir(str, default="data"), faiss_index_dir(str, default="data/indexes")
   - app/core/constants.py: SECTION_TYPE_MAP (dict 映射, 见下方), DEFAULT_TOP_K=5, SECTION_TYPES=["abstract","introduction","method","experiment","related_work","conclusion","other"]
   - app/core/logger.py: 配置 logging, 返回 get_logger(name) 函数

2. 创建 PDF 解析模块:
   - app/parsing/__init__.py
   - app/parsing/grobid_runner.py:
     - class GrobidRunner:
       - __init__(self, grobid_url: str)
       - parse(self, pdf_path: str) -> str: 读取 PDF, POST 到 {grobid_url}/api/processFulltextDocument (multipart form, 参数 input=pdf_bytes), 返回 TEI/XML string. 处理异常: 连接失败/非200状态码 抛出 GrobidError(自定义异常).
   - app/parsing/paper_normalizer.py:
     - normalize_section_type(title: str) -> str: 将原始 section title 映射为标准 section_type. 映射逻辑:
       - 转小写后匹配关键词: "introduction"→"introduction", "method"/"approach"/"proposed"→"method", "experiment"/"evaluation"/"result"→"experiment", "related work"/"background"→"related_work", "conclusion"/"summary"→"conclusion", "abstract"→"abstract"
       - 无法匹配则返回 "other"
   - app/parsing/tei_parser.py:
     - class TeiParser:
       - parse(self, tei_xml: str, paper_id: str) -> tuple[PaperMetadata, list[SectionChunk]]:
         - 用 lxml.etree 解析 XML (注意 TEI namespace: http://www.tei-c.org/ns/1.0)
         - 从 teiHeader/fileDesc/titleStmt 提取 title
         - 从 teiHeader/fileDesc/sourceDesc/biblStruct 提取 authors (forename + surname), year, venue
         - 从 teiHeader/profileDesc/abstract 提取 abstract
         - 从 teiHeader/profileDesc/textClass/keywords 提取 keywords
         - 从 body 遍历 div 提取 sections: 每个 div 的 head 作为 section_title, 所有 p 连接作为 text
         - 用 paper_normalizer 将 section_title 映射为 section_type
         - 为每个 section 创建 SectionChunk (chunk_id = f"{paper_id}_sec_{i}", page_start/page_end 默认 0)
         - section_titles 列表从所有 SectionChunk 提取

3. 创建入库脚本:
   - scripts/ingest_papers.py:
     - 接受 --pdf-dir 参数 (默认 data/raw_pdfs)
     - glob 所有 *.pdf
     - 对每个 PDF: grobid_runner.parse → tei_parser.parse → 保存为 data/parsed/{paper_id}.json
     - paper_id 生成: 取 PDF 文件名 (不含扩展名) 的 slugify 版本

4. 创建:
   - requirements.txt (包含: fastapi, uvicorn, pydantic, pydantic-settings, python-dotenv, lxml, requests, sentence-transformers, faiss-cpu, rank-bm25, numpy, FlagEmbedding, openai, tiktoken, pandas, gradio, pytest, pytest-asyncio, httpx, python-multipart)
   - .env.example (列出所有 Settings 字段)
   - docker-compose.yml: 一个 grobid 服务, 镜像 lfoppiano/grobid:0.8.1, 端口 8070:8070

5. 创建测试:
   - tests/__init__.py
   - tests/test_parsing/__init__.py
   - tests/test_parsing/test_paper_normalizer.py: 测试至少 10 种 section title 变体的映射
   - tests/test_parsing/test_tei_parser.py: 创建一个最小 TEI/XML fixture string, 测试 parse 返回正确的 PaperMetadata 和 SectionChunk 列表
   - tests/test_parsing/test_grobid_runner.py: mock requests.post, 测试正常返回和异常处理

代码要求:
- Python 3.11, 使用 type hints
- 所有模块有 __init__.py
- 异常处理完善, 不要 bare except
- 日志使用 core.logger
- 不要写 docstring 和多余注释, 代码自解释即可
```

---

### Prompt 2: Phase 2 — 索引层

```
你在 paper-agent/ 项目中继续开发。已有模块: core (schemas/config/constants), parsing (grobid_runner/tei_parser/paper_normalizer)。

请完成索引层:

1. app/indexing/__init__.py

2. app/indexing/vector_store.py:
   - class VectorStore:
     - __init__(self, index_path: str, embedding_model: str): 加载 sentence-transformers 模型, 初始化或加载 FAISS index (IndexFlatIP, 归一化后用内积等价于余弦)
     - _encode(self, texts: list[str]) -> np.ndarray: batch encode + L2 normalize
     - add(self, texts: list[str], metadatas: list[dict]): encode → faiss.add, 同时维护一个 list 存 metadata (按 index 对应)
     - search(self, query: str, top_k: int = 10) -> list[dict]: encode query → faiss.search → 返回 [{text, metadata, score}]
     - save(self): faiss.write_index + pickle dump metadata list
     - load(self): faiss.read_index + pickle load metadata list

3. app/indexing/bm25_index.py:
   - class BM25Index:
     - __init__(self, index_path: str)
     - add(self, texts: list[str], doc_ids: list[str]): jieba/空格 分词, 构建 BM25Okapi (rank_bm25). 同时保存 doc_ids 对应关系
     - search(self, query: str, top_k: int = 10) -> list[dict]: 分词 query → bm25.get_scores → top_k → [{doc_id, score, text}]
     - save(self) / load(self): pickle

4. app/indexing/paper_index.py:
   - class PaperIndex:
     - __init__(self, data_dir: str, embedding_model: str): 初始化 VectorStore + BM25Index + metadata_store (dict[str, PaperMetadata])
     - build(self, papers: list[PaperMetadata]): 对每篇论文构造检索文本 = f"{p.title} {p.abstract} {' '.join(p.keywords)}", 分别 add 到 vector_store 和 bm25_index. metadata_store[paper_id] = paper. 最后 save.
     - search(self, query: str, top_k: int = 5) -> list[tuple[PaperMetadata, float]]: 分别从 vector_store 和 bm25 检索 top_k*2, 返回 paper_id + score 列表 (融合在 retrieval 层做)
     - search_dense(self, query, top_k) -> list[tuple[str, float]]
     - search_sparse(self, query, top_k) -> list[tuple[str, float]]
     - get_by_id(self, paper_id: str) -> PaperMetadata | None
     - save(self) / load(self)

5. app/indexing/section_index.py:
   - class SectionIndex:
     - __init__(self, data_dir: str, embedding_model: str): 同上, 额外维护 chunk_store (dict[str, SectionChunk])
     - build(self, chunks: list[SectionChunk]): 检索文本 = f"{c.section_title}: {c.text}", add 到两个 index. 保存 chunk_store.
     - search_dense(self, query, top_k) -> list[tuple[str, float]]
     - search_sparse(self, query, top_k) -> list[tuple[str, float]]
     - get_by_id(self, chunk_id: str) -> SectionChunk | None
     - get_by_paper(self, paper_id: str) -> list[SectionChunk]: 返回该论文所有 chunks
     - save(self) / load(self)

6. scripts/build_indexes.py:
   - 读取 data/parsed/*.json → 反序列化为 PaperMetadata + SectionChunk
   - 调用 paper_index.build + section_index.build

7. 测试:
   - tests/test_indexing/test_vector_store.py: 用 3 条短文本测 add → search 返回正确
   - tests/test_indexing/test_bm25_index.py: 同上
   - tests/test_indexing/test_paper_index.py: mock 5 个 PaperMetadata, build 后 search 能命中
   - tests/test_indexing/test_section_index.py: mock 10 个 SectionChunk (2 篇论文各 5 section), 测 paper_id 过滤正确

注意:
- 英文论文, 分词直接用空格 split 即可, 不需要 jieba
- VectorStore 的 embedding encode 在测试中很慢, 测试中可以 mock _encode 返回随机向量
- 所有 index 文件保存到 data/indexes/ 下, 用不同前缀区分 (paper_dense.faiss, paper_bm25.pkl, section_dense.faiss, section_bm25.pkl, paper_meta.pkl, section_meta.pkl)
```

---

### Prompt 3: Phase 3 — 检索策略层

```
你在 paper-agent/ 项目中继续开发。已有: core, parsing, indexing (vector_store, bm25_index, paper_index, section_index)。

请完成检索策略层:

1. app/retrieval/__init__.py

2. app/retrieval/fusion.py:
   - def reciprocal_rank_fusion(results_list: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
     - 输入: 多路召回结果, 每路是 [(doc_id, score)] 已按 score 降序
     - 对每个 doc_id 计算 RRF score = sum(1/(k+rank)) across all lists
     - 返回按 RRF score 降序排列的 [(doc_id, rrf_score)]

3. app/retrieval/reranker.py:
   - class Reranker:
     - __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"): 加载 FlagEmbedding 的 FlagReranker
     - rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[tuple[int, float]]: 返回 [(original_index, score)] 按 score 降序, 截取 top_k

4. app/retrieval/paper_retriever.py:
   - class PaperRetriever:
     - __init__(self, paper_index: PaperIndex, reranker: Reranker)
     - retrieve(self, query: str, top_k: int = 3) -> list[PaperMetadata]:
       1. dense = paper_index.search_dense(query, top_k=20)
       2. sparse = paper_index.search_sparse(query, top_k=20)
       3. fused = reciprocal_rank_fusion([dense, sparse])[:10]
       4. 取 fused 的 paper_ids, 获取 abstracts 作为 passages
       5. reranked = reranker.rerank(query, passages, top_k)
       6. 返回对应的 list[PaperMetadata]

5. app/retrieval/section_retriever.py:
   - class SectionRetriever:
     - __init__(self, section_index: SectionIndex, reranker: Reranker)
     - retrieve(self, query: str, paper_id: str, target_sections: list[str] | None = None, top_k: int = 5) -> list[SectionChunk]:
       1. 获取该 paper 的所有 chunks: section_index.get_by_paper(paper_id)
       2. 如果 target_sections 不为空, 过滤 section_type in target_sections
       3. 如果过滤后候选 <= top_k, 直接返回 (无需检索)
       4. 否则: 对候选 chunks 做 dense 检索 + BM25 + RRF + rerank
       5. 返回 top_k 个 SectionChunk

6. 测试:
   - tests/test_retrieval/test_fusion.py: 测 RRF 融合, 两路结果有交集和无交集的情况
   - tests/test_retrieval/test_paper_retriever.py: mock paper_index 和 reranker, 测完整流程
   - tests/test_retrieval/test_section_retriever.py: mock section_index, 测 paper_id 过滤 + target_sections 过滤

注意:
- section_retriever 里如果候选很少 (比如一篇论文只有 6 个 section), 跳过检索直接按 section_type 过滤返回即可
- reranker 在测试中 mock 掉, 直接返回原始顺序
```

---

### Prompt 4: Phase 4 — Agent + 生成层

```
你在 paper-agent/ 项目中继续开发。已有: core, parsing, indexing, retrieval。

请完成 Agent 决策层和生成层:

1. app/generation/__init__.py
2. app/generation/prompts.py:
   定义以下 prompt 模板字符串常量:

   INTENT_CLASSIFICATION_PROMPT: 系统提示 + few-shot, 输入 query + current_paper_id + last_intent, 输出仅返回 paper_search / paper_reading / section_qa 三选一. Few-shot 示例至少 6 条 (每类 2 条). 包含回退场景: 已锁定论文但用户问了新论文 → paper_search.

   PAPER_READING_PROMPT: 接收 XML 格式的论文片段, 要求输出: 研究问题/核心贡献/方法概述/实验结论/局限性. 要求标注 [section_type] 来源. 信息不足时说明.

   SECTION_QA_PROMPT: 接收 XML 格式的章节片段 + 用户问题, 直接回答, 标注来源.

   CANDIDATE_SUMMARY_PROMPT: 接收候选论文列表, 为每篇生成: 标题/作者/年份/一句话摘要/匹配理由.

3. app/generation/citation_formatter.py:
   - def format_evidences_xml(evidences: list[Evidence], metadata: PaperMetadata) -> str:
     将 evidences 格式化为 XML 标签:
     <paper><metadata>...</metadata><evidence id="1" section_type="method">text</evidence>...</paper>
   - def add_citation_marks(answer: str, evidences: list[Evidence]) -> str:
     (简单实现: 如果 answer 中已有 [section_type] 标记则保留, 否则不额外处理)

4. app/generation/answer_builder.py:
   - class AnswerBuilder:
     - __init__(self, llm_api_key, llm_base_url, llm_model): 初始化 OpenAI 兼容 client
     - _call_llm(self, system_prompt: str, user_prompt: str) -> str: 通用 LLM 调用
     - classify_intent(self, query: str, state: AgentState) -> IntentType: 调用 LLM + INTENT_CLASSIFICATION_PROMPT, 解析返回值
     - generate_paper_reading(self, query: str, evidences: list[Evidence], metadata: PaperMetadata) -> str
     - generate_section_qa(self, query: str, evidences: list[Evidence], metadata: PaperMetadata) -> str
     - generate_candidates_summary(self, candidates: list[PaperMetadata], query: str) -> str

5. app/agent/__init__.py
6. app/agent/intent_router.py:
   - class IntentRouter:
     - __init__(self, answer_builder: AnswerBuilder)
     - classify(self, query: str, state: AgentState) -> IntentType:
       调用 answer_builder.classify_intent, 额外逻辑:
       - 如果 state.current_paper_id 为 None 且返回 paper_reading/section_qa → 强制改为 paper_search
       - 如果 state.current_paper_id 存在且返回 paper_search → 保持 (回退场景)

7. app/agent/state_manager.py:
   - class StateManager:
     - transition(self, state: AgentState, intent: IntentType, query: str) -> AgentState:
       按状态转移表更新 state:
       - paper_search → current_state=SEARCHING, 清空 candidate_papers, 如果是回退场景同时清空 current_paper_id
       - paper_reading → current_state=LOCKED (如果有 paper_id), 否则 SEARCHING
       - section_qa → current_state=READING (如果有 paper_id), 否则 SEARCHING
       返回新的 state (不要修改原对象, 创建新实例)
     - lock_paper(self, state: AgentState, paper_id: str) -> AgentState: 设置 current_paper_id, current_state=LOCKED
     - to_answering(self, state: AgentState) -> AgentState: current_state=ANSWERING
     - reset_to_routing(self, state: AgentState) -> AgentState: current_state=ROUTING, 保留 paper_id

8. app/agent/tools.py:
   - class AgentTools:
     - __init__(self, paper_retriever, section_retriever, paper_index)
     - search_papers(self, query: str, top_k: int = 3) -> list[PaperMetadata]
     - get_paper_metadata(self, paper_id: str) -> PaperMetadata | None
     - retrieve_sections(self, paper_id: str, query: str, target_sections: list[str] | None = None, top_k: int = 5) -> list[SectionChunk]

9. app/agent/orchestrator.py:
   - class Orchestrator:
     - __init__(self, intent_router, state_manager, tools, answer_builder, paper_index)
     - run(self, query: str, state: AgentState) -> AgentResponse:
       逻辑:
       a. intent = intent_router.classify(query, state)
       b. state = state_manager.transition(state, intent, query)
       c. if intent == PAPER_SEARCH:
            candidates = tools.search_papers(query)
            state.candidate_papers = [c.paper_id for c in candidates]
            if len(candidates) == 1: state = state_manager.lock_paper(state, candidates[0].paper_id)
            answer = answer_builder.generate_candidates_summary(candidates, query)
            return AgentResponse(answer, [], intent, state, candidates)
       d. if intent == PAPER_READING:
            metadata = tools.get_paper_metadata(state.current_paper_id)
            sections = tools.retrieve_sections(state.current_paper_id, query, ["abstract","conclusion"])
            sections += tools.retrieve_sections(state.current_paper_id, query, ["method","experiment"])
            去重 sections (by chunk_id)
            evidences = [Evidence(text=s.text, paper_id=s.paper_id, section_type=s.section_type, section_title=s.section_title) for s in sections]
            answer = answer_builder.generate_paper_reading(query, evidences, metadata)
            state = state_manager.to_answering(state)
            return AgentResponse(answer, evidences, intent, state, None)
       e. if intent == SECTION_QA:
            metadata = tools.get_paper_metadata(state.current_paper_id)
            sections = tools.retrieve_sections(state.current_paper_id, query, top_k=3)
            evidences = [...]
            answer = answer_builder.generate_section_qa(query, evidences, metadata)
            state = state_manager.to_answering(state)
            state.current_focus_section = sections[0].section_type if sections else None
            return AgentResponse(answer, evidences, intent, state, None)

10. 测试:
    - tests/test_agent/test_intent_router.py: mock answer_builder, 测 3 类意图 + 回退强制逻辑
    - tests/test_agent/test_state_manager.py: 测所有转移路径: ROUTING→SEARCHING, ROUTING→LOCKED, ROUTING→READING, lock_paper, to_answering, 回退 (LOCKED+paper_search→SEARCHING+清空paper_id)
    - tests/test_agent/test_orchestrator.py: mock 所有依赖, 测 3 个场景的完整 run 流程, 验证返回的 AgentResponse 结构和状态更新

代码要求:
- state_manager 必须返回新 AgentState 实例, 不要 mutate 原对象
- orchestrator.run 是同步方法 (V1 不需要 async)
- LLM 调用统一走 OpenAI 兼容接口 (openai.OpenAI client)
```

---

### Prompt 5: Phase 5 — 服务层 + API + Demo

```
你在 paper-agent/ 项目中继续开发。已有: core, parsing, indexing, retrieval, agent, generation。

请完成服务层、API 和 Demo:

1. app/api/__init__.py
2. app/api/deps.py:
   - 使用 functools.lru_cache 创建全局单例:
     - get_settings() -> Settings
     - get_paper_index() -> PaperIndex (load from disk)
     - get_section_index() -> SectionIndex (load from disk)
     - get_reranker() -> Reranker
     - get_answer_builder() -> AnswerBuilder
     - get_orchestrator() -> Orchestrator (组装所有依赖)
     - get_qa_service() -> QAService

3. app/services/__init__.py
4. app/services/ingest_service.py:
   - class IngestService:
     - __init__(self, settings: Settings)
     - ingest(self, pdf_dir: str) -> dict: 执行完整入库流程:
       1. glob pdf_dir/*.pdf
       2. 逐个解析 (grobid + tei_parser)
       3. 保存 parsed JSON
       4. 构建两层索引
       5. 返回 {"total": N, "success": M, "failed": L, "errors": [...]}

5. app/services/qa_service.py:
   - class QAService:
     - __init__(self, orchestrator: Orchestrator, state_manager: StateManager)
     - sessions: dict[str, AgentState] = {}
     - chat(self, query: str, session_id: str) -> AgentResponse:
       1. state = self.sessions.get(session_id, AgentState())
       2. response = orchestrator.run(query, state)
       3. # 如果 ANSWERING, 重置为 ROUTING 以便下轮
       4. next_state = state_manager.reset_to_routing(response.updated_state) if response.updated_state.current_state == ANSWERING else response.updated_state
       5. self.sessions[session_id] = next_state
       6. return response
     - lock_paper(self, session_id: str, paper_id: str) -> AgentState:
       用户确认选择某篇论文时调用, 更新 session state
     - get_state(self, session_id: str) -> AgentState

6. app/api/routes.py:
   - FastAPI APIRouter, prefix="/api"
   - POST /chat: body={query: str, session_id: str} → AgentResponse 的 dict 表示
   - POST /chat/lock: body={session_id: str, paper_id: str} → 锁定论文
   - POST /ingest: body={pdf_dir: str} → 入库结果
   - GET /papers: 列出所有论文 metadata
   - GET /papers/{paper_id}: 单篇论文 metadata
   - GET /session/{session_id}: 获取当前 session state

7. app/main.py:
   - 创建 FastAPI app
   - include router
   - 配置 CORS (allow all origins for dev)

8. app/demo.py:
   - Gradio Blocks 界面:
     - 左侧: Chatbot 组件 (多轮对话)
     - 右侧上: 当前状态显示 (current_state, current_paper_id, last_intent)
     - 右侧下: 候选论文列表 / evidence 列表 (Markdown 或 JSON)
     - 底部: 输入框 + 发送按钮
     - session_id 用 uuid4 生成, 保存在 gr.State
     - 调用后端 /api/chat 接口
     - 每轮显示: 意图类型标签 + 回答内容 + evidence 来源

9. 更新 scripts/ingest_papers.py: 改为调用 IngestService

10. 测试:
    - tests/test_api/test_routes.py: 用 FastAPI TestClient 测:
      - POST /api/chat 返回 200 + 正确结构
      - POST /api/ingest (mock ingest_service)
      - GET /api/papers

代码要求:
- API 响应统一用 Pydantic model 序列化
- demo.py 作为独立入口: python -m app.demo 启动 (同时启动 FastAPI + Gradio, 或者 Gradio 直接调用 QAService)
- Gradio Demo 推荐直接调用 QAService (不走 HTTP), 减少一层
```

---

### Prompt 6: Phase 6 — 评测

```
你在 paper-agent/ 项目中继续开发。已有: core, parsing, indexing, retrieval, agent, generation, api, services。

请完成评测模块:

1. app/evaluation/__init__.py

2. app/evaluation/dataset_builder.py:
   - @dataclass EvalSample: question(str), intent_type(str), gold_paper_ids(list[str]), gold_section_ids(list[str]), gold_answer(str)
   - def load_evalset(path: str = "data/eval/evalset.jsonl") -> list[EvalSample]: 逐行读 JSON
   - def save_evalset(samples: list[EvalSample], path: str): 逐行写 JSON

3. app/evaluation/llm_judge.py:
   - JUDGE_PROMPT 模板: 输入 question + gold_answer + predicted_answer, 要求对 faithfulness 和 relevance 各打 1-5 分, 输出格式 "faithfulness: X\nrelevance: X"
   - class LLMJudge:
     - __init__(self, api_key, base_url, model="gpt-4")
     - score_one(self, question, predicted, gold) -> dict: {"faithfulness": float, "relevance": float}
       调用 LLM, 解析返回的分数
     - score_batch(self, results: list[dict]) -> pd.DataFrame:
       输入 [{"question":..., "predicted":..., "gold":...}]
       返回 DataFrame: question | faithfulness | relevance

4. app/evaluation/baseline_compare.py:
   - class FlatRAGBaseline:
     - __init__(self, paper_index: PaperIndex, section_index: SectionIndex, answer_builder: AnswerBuilder)
     - query(self, question: str) -> str:
       1. 从 section_index 做纯 dense top-5 检索 (不分层, 不过滤 paper_id, 不用 reranker)
       2. 拼接 top-5 chunk text 作为 context
       3. 用简单 prompt "根据以下内容回答问题: {context}\n问题: {question}" 调用 LLM
       4. 返回 answer
   - class ABComparison:
     - __init__(self, qa_service: QAService, baseline: FlatRAGBaseline, judge: LLMJudge)
     - run(self, evalset: list[EvalSample]) -> pd.DataFrame:
       对每条 sample:
       1. agent_answer = qa_service.chat(sample.question, session_id=f"eval_{i}").answer
       2. baseline_answer = baseline.query(sample.question)
       3. agent_scores = judge.score_one(sample.question, agent_answer, sample.gold_answer)
       4. baseline_scores = judge.score_one(sample.question, baseline_answer, sample.gold_answer)
       汇总为 DataFrame: question | intent | agent_faith | agent_rel | baseline_faith | baseline_rel | delta_faith | delta_rel
     - summary(self, df: pd.DataFrame) -> dict: 按 intent 分组计算平均分和总平均分

5. scripts/run_eval.py:
   - 加载评测集
   - 初始化 QAService + FlatRAGBaseline + LLMJudge
   - 跑 A/B 对比
   - 输出结果到 data/eval/results.csv + 打印 summary
   - 同时计算 intent routing accuracy: 对每条 sample 单独调用 intent_router.classify, 与 gold intent_type 比较

6. scripts/build_evalset.py:
   - 辅助脚本: 交互式构建评测集
   - 展示已入库论文列表
   - 用户输入 question + 选择 intent_type + 选择 gold_paper + 选择 gold_sections + 输入 gold_answer
   - 保存到 evalset.jsonl

7. 创建 data/eval/evalset_template.jsonl: 3 条示例数据, 每种 intent 各 1 条

代码要求:
- LLM judge 的分数解析要健壮: 如果返回格式异常, 给默认分 0 并 log warning
- baseline 必须用和 agent 不同的检索策略 (纯 dense top-5, 无分层/无 rerank), 保证对比公平
- A/B 对比的每条 sample 用新 session_id, 避免状态串扰
```

---

## 八、关键实现约束（所有 Prompt 共用）

```
通用约束 (适用于所有 Phase):

1. Python 3.11, 使用 type hints, 不使用 Any 除非必要
2. 使用 dataclass 而非 Pydantic model 做内部数据结构 (schemas.py), API 层用 Pydantic
3. 所有文件操作使用 pathlib.Path
4. 日志统一用 app.core.logger.get_logger(__name__)
5. 配置统一从 app.core.config.Settings 读取
6. 不要过度抽象, 不要创建不必要的 base class
7. 不要添加 docstring / 多余注释, 代码自解释
8. 异常处理: 只在边界捕获 (API 层, 外部服务调用), 内部让异常传播
9. 测试: 外部依赖一律 mock (GROBID API, LLM API, reranker model), 索引测试可用小数据集
10. 不要创建 __main__.py, 入口统一在 scripts/ 或 app/main.py / app/demo.py

=== 以下为强制禁止项 (违反过的问题, 绝对不能再犯) ===

11. 禁止对 requirements.txt 中的硬依赖做 try/except ImportError fallback.
    faiss, sentence-transformers, pydantic-settings, FlagEmbedding 等都是硬依赖,
    直接 import, 缺失就让程序崩溃. 不要写任何 fallback 实现 (如 hash 伪 embedding,
    手动模拟 BaseSettings 等). 这类 fallback 会制造静默错误, 线上排查极其困难.

12. 禁止在索引层 (indexing/) 做检索结果融合. dense/BM25 分数不同尺度, 不能直接比较
    或取 max. 融合 (RRF) 统一在 retrieval/ 完成.

13. 禁止用 rank_bm25 以外的方式自行实现 BM25.

14. 索引层的 Index 类不要提供融合后的便捷 search() 方法.

15. tei_parser.py 禁止只取顶级 div 的直接子 <p>. 必须递归解析嵌套 <div>, 为每个子
    章节单独生成 fine chunk. coarse chunk 的 text 需包含该顶级 section 下全部段落
    的聚合文本 (递归). fine chunk 的 text 只取自身直接子 <p>, 不重复子 div 内容.
    任何 "只取一级" 或 "只取 ./tei:p 直接子节点" 的实现都是错的.

```
