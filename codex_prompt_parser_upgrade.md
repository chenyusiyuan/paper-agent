# Codex 执行 Prompt — 解析层升级（Parser Upgrade）

> 本 Prompt 升级 Phase 1（M1 core + M2 parsing）的解析能力。
> 请严格遵守末尾的「通用约束」，违反约束项的代码必须重写。

---

## 任务概述

当前 `tei_parser.py` 存在严重的解析不完整问题：

1. `_extract_sections` 只遍历 `body` 的顶级 `./tei:div`
2. 每个 div 只取直接子 `./tei:p`，导致所有子章节（subsection）内的段落完全丢失

例如：GROBID 解析出的 Method 章节结构如下：
```
<div>                               ← 顶级 div: "3 Methodology"
  <head>3 Methodology</head>
  <p>Overview</p>
  <div>                             ← 子 div: "3.1 Model Architecture"
    <head>3.1 Model Architecture</head>
    <p>Detailed content...</p>      ← 当前完全丢失
  </div>
  <div>                             ← 子 div: "3.2 Training"
    <head>3.2 Training</head>
    <p>Training details...</p>      ← 当前完全丢失
  </div>
</div>
```

**修复方案：递归解析 + 双粒度 chunk（coarse / fine）**

---

## 需要修改的文件

### 1. `app/core/schemas.py` — SectionChunk 新增字段

将 `SectionChunk` 修改为：

```python
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
    order_in_paper: int         # 全局顺序，coarse 和 fine 共用递增计数器，用于邻接窗口
    level: int                  # 层级深度，0 = 顶级 section，1 = subsection，以此类推
    parent_chunk_id: str | None # 父 chunk_id，顶级 coarse chunk 为 None
    granularity: str            # "coarse"（顶级大节聚合）| "fine"（子节/段落）
```

**注意：** 其他 dataclass（PaperMetadata、AgentState、Evidence、AgentResponse）保持不变。

---

### 2. `app/parsing/tei_parser.py` — 递归解析重写

完整重写 `_extract_sections` 方法，实现逻辑如下：

#### 核心算法

```
_extract_sections(root, paper_id) -> list[SectionChunk]:
    body = root.find(".//tei:text/tei:body")
    if body is None: return []

    chunks = []
    order_counter = 0  # 全局计数器，coarse 和 fine 共享

    for top_div in body.findall("./tei:div"):  # 只取顶级 div
        coarse_chunk, fine_chunks, order_counter = _parse_top_div(
            top_div, paper_id, order_counter
        )
        if coarse_chunk:
            chunks.append(coarse_chunk)
        chunks.extend(fine_chunks)

    return chunks

_parse_top_div(div, paper_id, order_counter) -> (coarse_chunk, fine_chunks, order_counter):
    section_title = _extract_div_head(div)  # 顶级标题，如 "3 Methodology"
    section_type = normalize_section_type(section_title)

    # coarse chunk: 递归收集该 div 下所有段落（含子 div）的文本
    all_text = _collect_all_paragraphs(div)
    if all_text:
        coarse_chunk = SectionChunk(
            chunk_id=f"{paper_id}_sec_{order_counter}",
            paper_id=paper_id,
            section_type=section_type,
            section_title=section_title,
            section_path=section_title,
            text=all_text,
            page_start=0, page_end=0,
            order_in_paper=order_counter,
            level=0,
            parent_chunk_id=None,
            granularity="coarse",
        )
        coarse_chunk_id = coarse_chunk.chunk_id
        order_counter += 1
    else:
        coarse_chunk = None
        coarse_chunk_id = None

    # fine chunks: 每个直接子 div 生成一个 fine chunk
    fine_chunks = []
    for sub_div in div.findall("./tei:div"):  # 只取直接子 div
        sub_title = _extract_div_head(sub_div)  # 如 "3.1 Model Architecture"
        sub_path = f"{section_title} > {sub_title}" if section_title else sub_title
        # fine chunk 只取子 div 的直接 <p>，不递归（避免重复）
        sub_paragraphs = [
            " ".join(p.strip() for p in node.itertext() if p.strip())
            for node in sub_div.findall("./tei:p")
        ]
        sub_text = "\n".join(p for p in sub_paragraphs if p).strip()
        if sub_text:
            fine_chunk = SectionChunk(
                chunk_id=f"{paper_id}_sec_{order_counter}",
                paper_id=paper_id,
                section_type=normalize_section_type(sub_title or section_title),
                section_title=sub_title or section_title,
                section_path=sub_path,
                text=sub_text,
                page_start=0, page_end=0,
                order_in_paper=order_counter,
                level=1,
                parent_chunk_id=coarse_chunk_id,
                granularity="fine",
            )
            fine_chunks.append(fine_chunk)
            order_counter += 1

    return coarse_chunk, fine_chunks, order_counter

_collect_all_paragraphs(div) -> str:
    # 递归收集 div 下所有 <p> 的文本（不论层级深度）
    paragraphs = [
        " ".join(part.strip() for part in node.itertext() if part.strip())
        for node in div.findall(".//tei:p")  # 注意：这里用 .// 递归
    ]
    return "\n".join(p for p in paragraphs if p).strip()
```

#### 关于 references / acknowledgment / appendix

这些 section 的 `section_type` 会被 `normalize_section_type` 映射为 `"other"`，正常入库即可，检索层会按需过滤。不需要在 parser 层特殊处理。

#### 关于 page_start / page_end

GROBID 的 TEI/XML 中 page 信息在 `<pb>` 标签上，V1 暂时保留 `page_start=0, page_end=0` 占位，与现有行为一致。

---

### 3. 测试文件更新

#### `tests/test_parsing/test_tei_parser.py`

原有测试需适配新字段。重点补充以下测试用例：

**test_parser_generates_coarse_and_fine_chunks**
- 构造一个包含嵌套 div 的 TEI XML（顶级 div 包含 2 个子 div）
- 断言：生成了 1 个 coarse chunk + 2 个 fine chunk
- 断言：coarse chunk 的 `granularity == "coarse"`, `level == 0`, `parent_chunk_id is None`
- 断言：fine chunk 的 `granularity == "fine"`, `level == 1`, `parent_chunk_id == coarse_chunk.chunk_id`
- 断言：coarse chunk 的 text 包含子 div 的段落内容（全量聚合）
- 断言：fine chunk 的 text 只包含自身直接 `<p>` 的内容

**test_parser_section_path**
- 顶级 div 标题 "3 Methodology"，子 div 标题 "3.1 Model"
- 断言：coarse chunk `section_path == "3 Methodology"`
- 断言：fine chunk `section_path == "3 Methodology > 3.1 Model"`

**test_parser_order_in_paper_is_global**
- 生成多个 coarse + fine chunk 后，所有 chunk 的 `order_in_paper` 是严格递增的唯一序列
- 不能存在两个 chunk 有相同的 `order_in_paper`

**test_existing_tests_still_pass**
- `test_parse_returns_paper_metadata` 等原有测试：更新 SectionChunk 构造参数以包含新字段

#### 关于 Stub/Mock 的 SectionChunk 构造

`tests/test_retrieval/test_section_retriever.py` 中 `make_chunk()` 需要更新以包含新字段，使用合理默认值：
```python
def make_chunk(paper_id, section_type, order, text, granularity="fine", level=1, parent_chunk_id=None):
    return SectionChunk(
        chunk_id=f"{paper_id}_{section_type}_{order}",
        paper_id=paper_id,
        section_type=section_type,
        section_title=section_type.title(),
        section_path=section_type.title(),
        text=text,
        page_start=0, page_end=0,
        order_in_paper=order,
        level=level,
        parent_chunk_id=parent_chunk_id,
        granularity=granularity,
    )
```

同样更新 `tests/test_indexing/test_section_index.py` 中所有直接构造 `SectionChunk` 的地方，补齐新字段。

---

## 不需要修改的文件

以下文件**无需改动**，因为它们以 `SectionChunk` 为整体处理，不依赖具体字段：

- `app/indexing/section_index.py`（接受 `list[SectionChunk]`，内部按 `chunk_id` 索引）
- `app/indexing/vector_store.py`
- `app/indexing/bm25_index.py`
- `app/retrieval/section_retriever.py`（消费 `SectionChunk`，字段访问不变）
- `app/retrieval/fusion.py`
- `app/retrieval/reranker.py`
- `app/retrieval/paper_retriever.py`
- `app/parsing/grobid_runner.py`
- `app/parsing/paper_normalizer.py`
- `app/core/config.py`
- `app/core/constants.py`

---

## 通用约束（所有 Phase 共用，必须遵守）

```
1. Python 3.11, 使用 type hints, 不使用 Any 除非必要
2. 使用 dataclass 而非 Pydantic model 做内部数据结构 (schemas.py), API 层用 Pydantic
3. 所有文件操作使用 pathlib.Path
4. 日志统一用 app.core.logger.get_logger(__name__)
5. 配置统一从 app.core.config.Settings 读取
6. 不要过度抽象, 不要创建不必要的 base class
7. 不要添加 docstring / 多余注释, 代码自解释
8. 异常处理: 只在边界捕获 (API 层, 外部服务调用), 内部让异常传播
9. 测试: 外部依赖一律 mock, 索引测试可用小数据集
10. 不要创建 __main__.py, 入口统一在 scripts/ 或 app/main.py / app/demo.py

=== 强制禁止项 ===

11. 禁止对 requirements.txt 中的硬依赖做 try/except ImportError fallback.
    faiss, sentence-transformers, pydantic-settings, FlagEmbedding 等都是硬依赖,
    直接 import, 缺失就让程序崩溃.

12. 禁止在索引层 (indexing/) 做检索结果融合. 融合 (RRF) 统一在 retrieval/ 完成.

13. 禁止用 rank_bm25 以外的方式自行实现 BM25.

14. 索引层的 Index 类不要提供融合后的便捷 search() 方法.

15. tei_parser.py 禁止只取顶级 div 的直接子 <p>. 必须递归解析嵌套 <div>,
    为每个子章节单独生成 fine chunk.
    coarse chunk 的 text 需包含该顶级 section 下全部段落的聚合文本 (.//tei:p 递归).
    fine chunk 的 text 只取自身直接子 <p> (./tei:p), 不重复子 div 内容.
    任何 "只取一级" 或 "只取 ./tei:p 直接子节点" 的实现都是错的.
```
