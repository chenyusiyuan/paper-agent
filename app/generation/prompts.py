INTENT_CLASSIFICATION_PROMPT = """你是论文问答系统的意图路由器。
你的任务是根据用户 query、当前锁定论文 current_paper_id、上一次意图 last_intent，在以下三类中做单一分类：
- paper_search
- paper_reading
- section_qa

判定规则：
1. 用户在找论文、比较论文、要求推荐论文、或者当前已锁定论文但用户切换到一篇新论文时，输出 paper_search
2. 用户想让你概览、总结、提炼一篇已锁定论文时，输出 paper_reading
3. 用户针对一篇已锁定论文的具体章节、方法细节、实验设置、公式或局部问题追问时，输出 section_qa
4. 只输出三选一，不要输出解释

示例：
Query: 推荐几篇关于检索增强生成的论文
Current Paper ID: None
Last Intent: None
Output: paper_search

Query: 找一篇讲图神经网络推理的最新论文
Current Paper ID: None
Last Intent: paper_search
Output: paper_search

Query: 这篇论文主要贡献是什么
Current Paper ID: paper_123
Last Intent: paper_search
Output: paper_reading

Query: 总结一下这篇文章的方法和实验结论
Current Paper ID: paper_456
Last Intent: paper_reading
Output: paper_reading

Query: 它的训练目标函数具体怎么设计的
Current Paper ID: paper_123
Last Intent: paper_reading
Output: section_qa

Query: 这篇论文的实验部分用了哪些数据集
Current Paper ID: paper_456
Last Intent: section_qa
Output: section_qa

Query: 另外帮我找一下关于多模态检索的论文
Current Paper ID: paper_123
Last Intent: section_qa
Output: paper_search
"""

PAPER_READING_PROMPT = """请基于下面的论文证据，用中文输出结构化阅读结果。
必须覆盖：
1. 研究问题
2. 核心贡献
3. 方法概述
4. 实验结论
5. 局限性

要求：
- 每个部分尽量标注来源，如 [abstract]、[method]、[experiment]、[conclusion]
- 如果证据不足，明确写“信息不足”
- 不要编造证据中没有的信息

用户问题:
{query}

论文元信息:
标题: {title}
作者: {authors}
年份: {year}
会议/期刊: {venue}

论文证据(XML):
{evidences_xml}
"""

SECTION_QA_PROMPT = """请基于下面的论文章节证据回答用户问题。

要求：
- 用中文直接回答
- 结论尽量带上来源标记，如 [method]、[experiment]
- 如果证据不足，明确说明
- 不要使用证据之外的推测

用户问题:
{query}

论文元信息:
标题: {title}
作者: {authors}
年份: {year}
会议/期刊: {venue}

章节证据(XML):
{evidences_xml}
"""

CANDIDATE_SUMMARY_PROMPT = """请根据用户问题，用中文总结候选论文列表。

要求：
- 对每篇候选论文输出：标题 / 作者 / 年份 / 一句话摘要 / 匹配理由
- 如果候选论文为空，明确说明没有找到合适结果
- 不要编造列表中没有的信息

用户问题:
{query}

候选论文列表:
{candidates_xml}
"""
