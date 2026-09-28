# Handoff：jevtest

日期：2026-09-28。给下一个要继续跑实验的 agent。先读完这一页再调用任何 API。

仓库：https://github.com/SIRI611/agent-memory （`main`）。
本机完整实验树：`/home/xirui4/projects/aip-jjin5/xirui4/jevtest`。
方法与数字的长文在 Google Doc「jevtest：记忆选择实验的方法与结果」。本文件只写还要做什么、为什么、以及怎么跑。

Jev 用环境变量 `AI_GATEWAY_API_KEY`。

## 1. 已经成立、不要重跑的结论

答题模型全是 `deepseek-flash`，思考关闭。这些百分比不能和 InMind 论文里 GPT-5-mini 的表相减。报告在 `results/e2e/report.json`、`results/v2/report.json`、`results/heldout/*/report.json`、`results/routing/offline_report.json`、`results/query_relevance/summary.json`。

读时 Jev：问题到来后，对每条已有原文问「这条事实是否必须用来安全、正确或恰当地回答」，按概率取前 5。策略名 `v0.2-query-necessity`。`state` 只有用户请求。K = 5 在答题前锁定。

| 设定 | 读时 Jev top-5 | 对照 | Oracle |
|---|---|---|---|
| InMind 间接题，125 题，池 = 1 条目标 + 238 条背景 | 应用准确率 84%，目标在上下文 94.4% | BM25 11.3% / 4.8%；MiniLM 17.6% / 9.6% | 92% |
| 同上，但池里再加上其余 121 条目标（360 条）。只做了离线召回，没有答题 | recall@5 **37.6%**，中位排名 7 | | |
| IMLogic 开发集，100 题，每用户约 744 条 | 准确率 71%，recall@5 **17%**，目标中位排名 33 | BM25 51% / 2%；MiniLM 41% / 5% | 95% |
| LongMemEval-S，500 题 | 准确率 73.0%，recall@5 96.3% | BM25 60.4% / 85.1% | 73.4% |

另外三条已经写死，不要再当成待测想法：

- 同一个 Jev 问题放在 BM25+向量的前 20 个锚点后面，InMind 目标进上下文只有 8.8%。相似度先筛候选这条路是负结果。
- 两阶段写入门（P(用户专属) ≥ 0.5 再读时排序）在 InMind 上把应用准确率从 84% 提到 90.4%，在 LongMemEval-S 上从 73% 打到 48.8%。门会丢掉助手回复和看起来不像用户事实的轮次。不要把它写成方法。
- 写入时抽事实和规则、对比重排、DeepSeek 自己扫全库，都没有超过原文上的读时 top-5。规则 top-5 是 69.6%。IMLogic 上没有任何重排变体超过不加第二遍的 Jev top-10（开发集准确率 74%）。重排提高召回时，陷阱一起进来。

InMind 现在是开发集：125 题都看过了。IMLogic 的 100 题（`random.Random("imlogic-heldout-v1")`，每用户 5 题）也是开发集。`rerank.py` 写明真正的留出划分还没有抽。LongMemEval-S 的 500 题结果已经看过，不能再拿来设计方法。

## 2. 还要做的事

缺口不在「小池子上再把 84% 抠到 90%」。小池子里只有一条真事实，其余是闲聊，读时全量打分已经接近 Oracle。下面按值得花钱的顺序排。

### 2.0 用 GPT-5-mini 和 GPT-5.6 把生成式实验各跑一遍

现有表的答题、评判、always-in-state 更新、事实与规则抽取、`llm_scan`、`llm_rerank` 用的都是 `deepseek-flash`。这些调用换成两个模型各跑一整遍：

- GPT-5-mini。这一轮用来和 InMind 论文里 GPT-5-mini 的绝对数字对照。
- GPT-5.6。同一协议上的第二列，不和论文的表相减。

Jev 打分、BM25 和 MiniLM 不换模型，继续用本机缓存。选择器已经定下来的上下文直接复用，不要为了换答题模型重打必要性。

范围是已经落盘的全部条件，不是新方法：InMind 端到端（`e2e` 与 `e2e_v2`）、IMLogic 开发集、LongMemEval-S、重排开发集里已经答过的臂、always-in-state、事实与规则抽取。提示词、温度、K、记忆池与 DeepSeek 那一轮相同。思考模式在 API 提供开关时关闭。答题和评判用同一个模型，和 InMind 的协议一致。

落盘目录分开，禁止写入已有的 `results/e2e/`、`results/v2/`、`results/heldout/`、`results/rerank/`：

- `results/gpt-5-mini/`
- `results/gpt-5.6/`

发请求之前把两个 API 模型 id 写进新的 `configs/gpt.json` 并冻结。名字以本节为准。id 在第一笔付费调用之后不再改。

顺序：先干跑，记下每个臂的请求数。然后只跑 GPT-5-mini 的 InMind 端到端，条件与 `results/e2e/report.json` 相同。Oracle 的间接应用准确率若远离论文的 84%，或掉到无记忆附近，就停，先核对模型 id 和提示词，再开 IMLogic 和 LongMemEval。GPT-5.6 用同一套检查：它的 Oracle 应接近它自己的能力上限，而不是接近无记忆。

DeepSeek 的 100 美元上限不能当作 OpenAI 的预算。GPT 花费单独记账。

### 2.1 大记忆库里把目标送进前 5

IMLogic 上目标中位排名是 33，recall@5 是 17%，recall@10 是 30%，recall@20 是 40%。准确率 71% 对 Oracle 95%，主要是因为少把陷阱送进去（陷阱在上下文里 13%，MiniLM 是 59%），不是因为找到了目标。

下一个方法要同时移动两件事：目标 recall@5，以及陷阱在上下文里的比例。只加长 K 会把陷阱一起拉进来（开发集上 top-10 准确率 74%，top-20 的陷阱率已经到 39%）。相似度锚点在 InMind 上已经失败，不要再做「先检索再重排」。缩小步骤还要满足第 2.5 节：判断保持点式，不能把候选放进同一张列表。

全量打分在 N≈240 时可行，在 N≈744 时 top-5 够不着目标。设计时只能看第 1 节里已经公开的开发集数字。新的 IMLogic 划分抽出来之后，在出分之前把规则写进这个文件或一份新的 `configs/`，不能边看边改。

### 2.2 用已经缓存的分数，答「多事实池」

`results/query_relevance/summary.json` 的 `multi_fact` 是背景加全部不重复目标，池大小 360，读时 Jev recall@5 = 37.6%。`routing.py` 里的 `multi_fact` 是另一个更松的池（最多 15 条跨领域干扰），recall@5 = 86.4%，也不要和 37.6% 混用。端到端答题用的是 239 条的原池（`pool_for` 默认 `inmind`）。

这一步不需要新的 Jev 调用。用 `results/query_relevance/cache.jsonl` 里已有的概率，在 360 条的池上取 top-5，间接题用 DeepSeek、GPT-5-mini、GPT-5.6 各答一遍，评判用同一个模型。DeepSeek 这一列如果准确率跟着召回掉下去，第 1 节的 84% 就不能写成方法在真实记忆库上的表现。GPT 两列写进第 2.0 节的目录，不要覆盖 `results/e2e/`。

### 2.3 先不要花的钱

第 2.0 节是换答题模型，条件与 DeepSeek 那一轮相同，要跑。下面这些是新方法，已经有负结果或被更简单的信号盖过。除非 2.1 或 2.5 做出了新的分离，不要把它们再设计一遍：

- 重定 v0.1 权重。单个信号「是否用户专属」的 AUROC 是 0.965，加权优先级是 0.954。512 token 上优先级和「只挑短句」的可见率分别是 87.2% 和 84.8%。
- 再抽规则、再做对比重排、再让 DeepSeek 扫全库。
- 用两阶段门当主方法。
- 把 DeepSeek 的 always-in-state（应用准确率 38.7%，目标原文在 profile 里 1.6%）写成 InMind 论文里 68.8% 的复现。更新器没有把目标留下来。要当基线，先修到目标可见，或者从表里拿掉。

### 2.4 数字能写进论文之前还缺的测量

2.1 有了结果再做，不要插到前面：

- 无关问题。官方答题提示词要求主动提到安全事实，会抬高间接题。加一组「无关事实不要强行使用」的提示词。跨领域负例上，已存事实仍进入读时 top-5 的比例是 32.8%（P>0.5 只有 8%）。
- 费用和延迟对记忆条数 N 的曲线。这是对「全量打分失去亚线性成本」的直接回应。缓存命中不能算进这条曲线。
- 预算用 `ceil(字符数/4)`，不是答题模型的分词器。在宣称 token 预算之前换成真实分词器，只重新装箱，不重新打分。

### 2.5 保留点式潜在效用，同时避开全量打分和列表互扰

要回答的问题是：Can we preserve Jev-style potential-utility judgment while avoiding exhaustive pointwise evaluation and listwise interference?

Jev 现在的判断是点式潜在效用。一条事实单独拿来问：即使后来的问题不点名它，它会不会改变建议，或者它是不是回答当前请求所必需的。这个判断只依赖「这一条事实」和「当前请求」，不依赖同批的其他事实。

现有两条路分别坏在问题的两端：

- 全量点式评估。库里每一条都单独打一次。N≈239 时 InMind 目标进上下文 94.4%。N≈744 时 IMLogic 的 recall@5 只有 17%，调用次数还随 N 线性增长。
- 列表式评估。`rerank.py` 把第一阶段的前 M 条放进同一次判断（`jev_contrast`、`llm_rerank`），让模型在看到其他候选的情况下决定谁更关键、是否重复、是否被替代。一条事实的分数因此受列表里其他事实影响。开发集上目标召回上升，陷阱一起上升，准确率没有超过不加第二遍、只把 K 加到 10 的点式排名。

要保住的是点式潜在效用：缩小之后，每一条入选事实的分数仍然可以单独解释，换一批同伴不应改变它的分数。要去掉的是对全库每条都发一次调用。相似度锚点不算一种答案，那是把效用判断换成了相似性，InMind 上目标进上下文只有 8.8%。

可测的形状是：先用另一个点式问题把集合缩小，再对缩小后的集合问现有的必要性问题。前一步的 `state` 里不能出现其他候选。这和第 2.1 节是同一项工作上的约束，不是另一个可以靠列表重排交差的任务。规则写下来之后再打分，划分不能包含 IMLogic 开发集那 100 个 id。

## 3. 下一个 agent 的顺序

1. 确认当前目录是本机 `jevtest`，并且 `results/query_relevance/cache.jsonl` 与 `results/e2e/report.json` 都在。不在就停，不要从 GitHub 克隆重打 InMind。
2. 干跑现有入口，确认不加执行开关时请求数不变：`python -m jev_persist.e2e`、`python -m jev_persist.e2e_v2`。同时给 GPT 两个臂干跑一遍，把请求数写进 `configs/gpt.json` 旁边的说明。
3. 做第 2.0 节。先 GPT-5-mini 的 InMind 端到端。Oracle 检查通过之后，再跑该模型的其余基准，然后用同样顺序跑 GPT-5.6。结果只写进 `results/gpt-5-mini/` 和 `results/gpt-5.6/`。
4. 做第 2.2 节。新条件单独命名，例如 `jev_top5_multifact`。三个答题模型各一份报告。报告应用准确率、只看答案、目标是否在上下文。
5. 第 2.2 节落盘之后，再设计第 2.1 节，并满足第 2.5 节。先把留出划分和缩小规则写下来，再调用 Jev。划分不能包含 IMLogic 开发集那 100 个 id（`heldout_imlogic.sample()`）。
6. DeepSeek 沿用 `jev_persist/budget.py`，不要绕过 100 美元上限。OpenAI 用单独的账本和 `configs/gpt.json` 里的上限。

## 4. 实现约束

- 写入时打分的 `state` 只有记忆原文。不能放入未来问题、桥接解释、关系、实体。
- 读时打分的 `state` 只有用户请求。每个布尔问题携带一条记忆。缓存键是策略版本 + 模型名 + 问题哈希 + 记忆哈希。见 `query_relevance.py`。
- 平分用文本哈希打破，保证排名可重复。
- 端点：Jev `POST https://ai-gateway.vercel.sh/v1/evaluate`，模型 `typesafe-ai/jev`，读 `answers[id].probability`。DeepSeek `https://api.deepseek.com/chat/completions`，模型 `deepseek-flash`，`thinking` 为 `disabled`，答题最多 1024 token，评判最多 512。配置在 `configs/pilot.json`。GPT-5-mini 与 GPT-5.6 的 API 模型 id 写在 `configs/gpt.json`，结果目录见第 2.0 节。
- 向量只在 `.venv-embed` 里算，模型 `all-MiniLM-L6-v2`。
- `tests/test_framework.py` 会读 `results/framework/pilot_manifest.json`。这份文件不在 GitHub 上，测试只在本机实验树上跑。

## 5. 入口

| 命令 | 作用 |
|---|---|
| `python -m jev_persist.e2e` | 125 题端到端。`--execute` 才发 DeepSeek |
| `python -m jev_persist.e2e_v2` | 第二轮条件。`--only` 可按条件前缀过滤 |
| `python -m jev_persist.heldout_imlogic` | `--embed` / `--jev` / `--admit` / `--answer` |
| `python -m jev_persist.longmemeval` | `--jev` / `--admit` / `--embed` / `--answer` |
| `python -m jev_persist.rerank --split dev` | 只覆盖已经看过的 100 题。不要在这里选新超参再当成确认 |
| `python -m jev_persist.query_relevance` | 读时打分。`--execute` 才发 Jev。默认补缓存缺口 |
| `python -m jev_persist.offline_report` | 用缓存算召回，不花钱 |
