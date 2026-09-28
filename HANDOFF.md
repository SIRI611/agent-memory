# Handoff：jevtest

日期：2026-09-28。给下一个要继续跑实验的 agent。先读完这一页再调用任何 API。

仓库：https://github.com/SIRI611/agent-memory （`main`，代码提交 `4e30170`）。
本机完整实验树：`/home/xirui4/projects/aip-jjin5/xirui4/jevtest`。
方法与数字的长文在 Google Doc「jevtest：记忆选择实验的方法与结果」。本文件只写还要做什么、为什么、以及怎么跑。

## 0. 先用哪份目录

在这台机器上跑，用本机的 `jevtest`，不要用一份只有 GitHub 代码的新克隆。缓存、答题记录和 LongMemEval 数据都只在本机。

GitHub 上没有这些东西，`.gitignore` 也把它们排除了：

| 只在本机 | 原因 |
|---|---|
| `.env` | API 密钥。不要提交，不要打印 |
| `.venv-embed/` | MiniLM 环境，约 2.4 GB |
| `results/` | Jev 缓存、DeepSeek 答题和报告，约 108 MB |
| `novelty/papers/` | 论文 PDF |
| `data/heldout/longmemeval/longmemeval_s_cleaned.json` | 277 MB，超过 GitHub 单文件 100 MB 限制 |

密钥从本机 `.env` 读，需要 `AI_GATEWAY_API_KEY`（Jev）和 `DEEPSEEK_API_KEY`。项目账本上限 100 美元，截至 2026-09-27 的累计估计是 **16.45 美元**（`results/framework/deepseek_ledger.jsonl` 最后一行的 `spent_after_usd`）。Jev 的促销免费到 2026-09-25，现在再打分可能要花钱。已经缓存的 pair 不要重打。

工作目录必须是 `jevtest` 根目录，这样 `python -m jev_persist....` 才能找到包。不加 `--execute`（或对应的阶段开关）时，入口只计数，不发请求。先干跑，再加开关。

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

## 2. 改进空间

缺口不在「小池子上再把 84% 抠到 90%」。小池子里只有一条真事实，其余是闲聊，读时全量打分已经接近 Oracle。下面按值得花钱的顺序排。

### 2.1 大记忆库里把目标送进前 5

IMLogic 上目标中位排名是 33，recall@5 是 17%，recall@10 是 30%，recall@20 是 40%。准确率 71% 对 Oracle 95%，主要是因为少把陷阱送进去（陷阱在上下文里 13%，MiniLM 是 59%），不是因为找到了目标。

下一个方法要同时移动两件事：目标 recall@5，以及陷阱在上下文里的比例。只加长 K 会把陷阱一起拉进来（开发集上 top-10 准确率 74%，top-20 的陷阱率已经到 39%）。相似度锚点在 InMind 上已经失败，不要再做「先检索再重排」。

全量打分在 N≈240 时可行，在 N≈744 时 top-5 够不着目标。要测的是一种不靠字面或向量相似度的缩小步骤，缩小之后仍然用现有的必要性问题排序。设计时只能看第 1 节里已经公开的开发集数字。新的 IMLogic 划分抽出来之后，在出分之前把规则写进这个文件或一份新的 `configs/`，不能边看边改。

### 2.2 用已经缓存的分数，答「多事实池」

`results/query_relevance/summary.json` 的 `multi_fact` 是背景加全部不重复目标，池大小 360，读时 Jev recall@5 = 37.6%。`routing.py` 里的 `multi_fact` 是另一个更松的池（最多 15 条跨领域干扰），recall@5 = 86.4%，也不要和 37.6% 混用。端到端答题用的是 239 条的原池（`pool_for` 默认 `inmind`）。

这一步不需要新的 Jev 调用。用 `results/query_relevance/cache.jsonl` 里已有的概率，在 360 条的池上取 top-5，用现有的 DeepSeek 答题和 InMind 评判各跑一遍间接题。如果准确率跟着召回掉下去，第 1 节的 84% 就不能写成方法在真实记忆库上的表现。

### 2.3 先不要花的钱

这些已经有负结果或被单个更简单的信号盖过。除非 2.1 或 2.2 做出了新的分离，不要重跑：

- 重定 v0.1 权重。单个信号「是否用户专属」的 AUROC 是 0.965，加权优先级是 0.954。512 token 上优先级和「只挑短句」的可见率分别是 87.2% 和 84.8%。
- 再抽规则、再做对比重排、再让 DeepSeek 扫全库。
- 用两阶段门当主方法。
- 把 DeepSeek 的 always-in-state（应用准确率 38.7%，目标原文在 profile 里 1.6%）写成 InMind 论文里 68.8% 的复现。更新器没有把目标留下来。要当基线，先修到目标可见，或者从表里拿掉。

### 2.4 数字能写进论文之前还缺的测量

2.1 有了结果再做，不要插到前面：

- 无关问题。官方答题提示词要求主动提到安全事实，会抬高间接题。加一组「无关事实不要强行使用」的提示词。跨领域负例上，已存事实仍进入读时 top-5 的比例是 32.8%（P>0.5 只有 8%）。
- 费用和延迟对记忆条数 N 的曲线。这是对「全量打分失去亚线性成本」的直接回应。缓存命中不能算进这条曲线。
- 预算用 `ceil(字符数/4)`，不是 DeepSeek 的分词器。在宣称 token 预算之前换成真实分词器，只重新装箱，不重新打分。

## 3. 下一个 agent 的顺序

1. 确认当前目录是本机 `jevtest`，并且 `results/query_relevance/cache.jsonl` 与 `results/e2e/report.json` 都在。不在就停，不要从 GitHub 克隆重打 InMind。
2. 干跑现有入口，确认不加执行开关时请求数不变：`python -m jev_persist.e2e`、`python -m jev_persist.e2e_v2`。
3. 做第 2.2 节。新条件单独命名，例如 `jev_top5_multifact`，不要覆盖 `results/e2e/`。答题沿用 `render.py` 的提示词和 `e2e.py` 的评判。报告应用准确率、只看答案、目标是否在上下文。
4. 第 2.2 节的数字落盘之后，再设计第 2.1 节。先把留出划分和缩小规则写下来，再调用 Jev。划分不能包含 IMLogic 开发集那 100 个 id（`heldout_imlogic.sample()`）。
5. 每次付费运行前看账本。沿用 `jev_persist/budget.py`，不要绕过 100 美元上限。

## 4. 实现约束

- 写入时打分的 `state` 只有记忆原文。不能放入未来问题、桥接解释、关系、实体。
- 读时打分的 `state` 只有用户请求。每个布尔问题携带一条记忆。缓存键是策略版本 + 模型名 + 问题哈希 + 记忆哈希。见 `query_relevance.py`。
- 平分用文本哈希打破，保证排名可重复。
- 端点：Jev `POST https://ai-gateway.vercel.sh/v1/evaluate`，模型 `typesafe-ai/jev`，读 `answers[id].probability`。DeepSeek `https://api.deepseek.com/chat/completions`，模型 `deepseek-flash`，`thinking` 为 `disabled`，答题最多 1024 token，评判最多 512。配置在 `configs/pilot.json`。
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
