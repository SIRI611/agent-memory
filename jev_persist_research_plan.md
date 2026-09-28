# Jev-Persist Research Plan

版本 v0.2，2026-09-23。v0.1 里的大而全路线收成现在这条：先证明写入时的选择，再花钱答题。

## 0. 课题

现有长期记忆等到 query 出现后，再用 query 去检索。InMind 证明，真正关键的旧事实和当前问题常常没有字面或向量相似性，检索不会把它送进上下文。记忆直接可见时，GPT-5-mini 的间接题准确率是 84.0%；六个检索系统最好只有 16.0%。

本项目不新做一套检索，也不把「有一块常驻记忆」当作贡献。要测的是：

> 在看不到未来 query 的前提下，Jev 能否在严格 token 预算内，把那些以后会改变建议的原文事实留在始终可见的 core 里，并因此提高间接题准确率。

三个条件同时成立才算我们的方法：写入时选择、始终可见、严格预算。对照必须来自文献里已经测过的系统，高度按下面第 1 节。

论文里不能写「第一个 persistent memory」或「第一次用 Jev 做记忆」。Jev-Mem、Letta、社区项目 `jev-memory` 都已经占了这些说法。可写的是：在 InMind 这种隐式关联上，比较决策模型和文献中的写入策略，谁能在小预算下把关键事实留在可见上下文里。

## 1. 要打的高度

下面的百分比都是 InMind 用 **GPT-5-mini** 答题、**GPT-5-mini** 评判得到的（Li et al., 2026，arXiv:2607.24368）。换成 DeepSeek 之后，这些绝对数字不能直接当目标；先在同一模型上重测 Oracle 和对照，再看相对高度。

| 目标 | 文献里它是什么 | 间接题 | 目标可见 |
|---|---|---:|---:|
| Naive RAG，`text-embedding-3-large`，top-5 | 检索上限。建索引不用语言模型 | 16.0% | 6.4% |
| MemoryOS | 已有常驻 profile，靠 heat 晋升。记忆内部是 GPT-4o-mini | 14.4% | 7.2% |
| A-Mem、HippoRAG 2、A-RAG、xMemory、Mem0 | 同一张表里的其他检索系统，间接题 3.2%–9.6% | 不逐个重跑 | |
| always-in-state | 每轮用 GPT-5-mini 重写最多 200 行 / 25KB 的 profile，整份可见 | 68.8% | 直接召回 98.4% |
| Oracle / backbone | 目标事实直接放进上下文 | 84.0% | 100% |

记忆内部模型并不都是 GPT-5-mini。xMemory、Mem0、A-Mem、HippoRAG 2、MemoryOS 用 GPT-4o-mini 建记忆，再用 GPT-5-mini 答题。A-RAG 的代理和 always-in-state 的更新器都是 GPT-5-mini。Naive RAG 的向量是开源 MiniLM，或 OpenAI 的 `text-embedding-3-large`。

Generative Agents 的 relevance + recency + importance，以及 Letta 的 memory block，没有出现在这张表里，不能和上面的百分比并列。它们是主实验里要在同一答题模型下重做的启发式和语言模型选择器。

高度分三档：

1. 明显超过检索上限。停在 16% 附近没有论文。InMind 已经用一个大 profile 做到了 68.8%，所以只超过 16% 也不够。
2. 在大约 512 token 的预算下，补回 Oracle 和检索之间的大部分缺口。GPT-5-mini 上，这个「大部分」就是从 16% 回到 68.8%，离 Oracle 的 84% 大约还差 15 个点。
3. 同样的小预算下，超过 MemoryOS 的 heat 晋升和 Generative Agents 启发式。和语言模型 profile 比，准确率可以接近，但可见 token 和写入成本必须更低。

主表用同一答题模型重测，不把 DeepSeek 的分数和上表直接相减。

## 2. 已经做完的结果

只调用了 Jev（`typesafe-ai/jev`，AI Gateway `POST /v1/evaluate`）。DeepSeek 没有调用。Gateway 记录的费用是 **$0**。360 条不重复文本全部打分成功；中途的 429 已补打。请求里只有记忆原文，没有未来 query，没有桥接解释。

正例是 InMind 的 125 个目标事实（122 条不重复文本）。负例是固定 LME-s 背景里的 238 条用户发言。权重是预先写下的 v0.1，没有在这批数据上拟合：

```text
priority = 0.35 * P(stable_constraint) + 0.30 * impact + 0.20 * P(stable) + 0.15 * P(user_specific)
```

`impact` 把五档分数线性归一化到 0–1。

| 分数 | AUROC | 平均精度 | 中位排名（239 条里） |
|---|---:|---:|---:|
| v0.1 priority | 0.954 | 0.935 | 2 |
| 只看是否用户专属 | 0.965 | 0.946 | |
| 是否改变未来决策 | 0.913 | 0.892 | 2 |
| 遗忘影响 | 0.898 | 0.873 | |
| 句子更短 | 0.905 | 0.731 | 23 |
| 是否稳定约束 | 0.826 | 0.888 | |
| 是否跨会话稳定 | 0.794 | 0.712 | |
| 关键词 | 0.638 | 0.929 | 64 |

目标事实的平均优先级是 0.619，背景发言是 0.257。把背景限制到不超过最长目标（131 字符，71 条）之后，priority 的 AUROC 仍是 0.965，用户专属是 0.989。分开目标这件事不是「目标句子更短」造成的。

可见率用 `ceil(字符数/4)` 当 token，按优先级装箱，装不下就跳过，平分用文本哈希打破。125 题上目标被装进 core 的比例：

| 预算 | Jev priority | 只挑短句 | 关键词 |
|---:|---:|---:|---:|
| 128 | 71.2% | 36.0% | 33.6% |
| 512 | 87.2% | 86.4% | 32.0% |

128 token 时选择质量有差别。512 token 时 Jev 和挑短句几乎一样，所以这个预算上不能只报可见率，必须看答题。关键词不是文献方法，只留在消融里。

单个信号里，「是不是关于这个用户的持久事实」比加权公式略强。v0.1 权重因此不能写成最优。正式实验前可以在合成开发集上重定权重，不能用 InMind 测试题来调。

优先级低于 0.35 的有 16 题，写在 `results/framework/failure_cases.json`。垫底的是写的时候看不出以后会有用的事实，例如新买的 M4 MacBook、下周二体检、交友软件、去西藏、我是广告经理。Jev 把它们标成一次性事件或普通身份。这是写入时选择的天花板，不是实现错误。健康、关系和法律约束的平均优先级更高，职业和金融事实更低。

## 3. 锁定的决定

- 控制器只用 Jev。准入时禁止看到未来 query、解释、实体和关系。
- Jev 选择原文，不摘要、不改写。
- 试点答题走 DeepSeek 官方 API：`https://api.deepseek.com`，模型名 `deepseek-flash`（当前是 DeepSeek-V4.1-Flash）。思考模式显式关闭。输出上限：答题 1024 token，评判 512 token。
- 和 InMind 对绝对数字的那一轮才用 GPT-5-mini。那一轮之前，先用 DeepSeek 确认「记忆可见时模型会用」。
- 不重跑 InMind 表里的六个系统。同一模型下的检索代表只留 Naive RAG。
- 挑短句、关键词是干扰检查，不进主表。
- 不克隆 Jev-Mem，不跑 LoCoMo，直到 DeepSeek 门槛通过。
- Hobby 计划不能开 Zero Data Retention。请求里不要带这个选项。
- Jev 的促销免费到 2026-09-25。DeepSeek 未命中缓存时，非高峰大约是输入 $0.15、输出 $0.60 / 百万 token，工作日高峰加倍。

和 InMind 协议的差别写在 `configs/pilot.json`。使用官方答题提示词，它要求主动提到安全相关事实。这可能抬高间接题分数，所以通过门槛之后要加一组「无关事实不要强行使用」的提示词，并做无关问题反例。

## 4. 方法

每条候选记忆是一条用户原文。背景用 InMind 固定的 47 轮 LME-s（486 个 turn，其中 240 个用户 turn，去重后 238 条）。目标是该题的 `user_message`。每条文本单独交给 Jev，五个问题：

- 记忆类型：稳定约束、偏好、身份、长期目标、正在进行的承诺、资源、一次性事件、暂时状态、其他
- 是否会在以后的会话里仍然相关
- 是否会改变以后的建议、决定或行动，即使后来的问题不点名它
- 忘掉之后误导建议有多严重：可忽略、轻微、有意义、重大、关键
- 是否是关于用户、环境、义务、偏好或约束的持久事实

最终 core 是这些分数在预算内的装箱，不是逐轮让语言模型重写 profile。这和 InMind 的 always-in-state 不同，论文里要写明。

缓存键是 `策略版本 + Jev 模型名 + 原文`。换预算只重新装箱，不重新调用 Jev。

## 5. 已经生成、尚未发送的试点

20 题清单在 `results/framework/pilot_manifest.json`。规则在看到任何答题之前就定了：优先级低于 0.35 的题最多取 12 条最低的；其余名额先按领域补最高分，再按全局最高分补满。16 条低分里有 4 条因此没进试点。

低分一侧包括 MacBook、体检、交友软件、西藏、广告经理。高分一侧包括华法林、甲壳类过敏、鸡蛋过敏、清真饮食、深静脉血栓、认知障碍、多动症和两岁孩子。

七个条件，每个条件答直接题和间接题：

| 条件 | 作用 |
|---|---|
| oracle | 只放目标事实。DeepSeek 的能力上限 |
| no_memory | 什么都不放 |
| jev_priority，128 和 512 | 要测的方法 |
| shortness，128 和 512 | 干扰检查 |
| keyword，512 | 干扰检查 |

共 280 个答题请求、560 个评判请求，在 `results/framework/requests/`。答题上下文不含解释和问题。评判沿用 InMind 的四份提示词：直接题、间接应用、只看答案、目标是否在上下文里。本地另有一个不花钱的检查：目标原文是否出现在上下文中。

测试 `tests/test_framework.py` 已通过。不加 `--execute` 时，`python3 -m jev_persist.run_deepseek` 不会打到 DeepSeek。

这 20 题先回答两个问题。Oracle 上间接题如果经常答错，DeepSeek 不能用，换模型，不要继续解释记忆。低分记忆在 Oracle 上能答对、在 Jev core 里却不在上下文中，那才是写入时选择的失败。

## 6. 门槛通过之后的主实验

同一答题模型、同一评判、同一历史、同一分词预算。主表：

| 方法 | 从哪来 |
|---|---|
| 无记忆 | 下限 |
| Oracle | InMind backbone |
| Naive RAG | InMind 的检索上限，top-5 |
| Generative Agents 启发式 | Park et al., 2023 |
| always-in-state，但压到同一 token 预算 | InMind Algorithm 1 的准入标准，去掉 200 行的宽松上限 |
| Jev core | 本方法 |
| 全历史 | 昂贵参照，不当作唯一上界 |

MemoryOS 整套系统不必重写。要复现的是它的晋升规则：什么分数能把事实送进可见 profile。Letta 的 memory block 若接入成本过高，用同一标准的结构化 LLM 选择器代替，并在文中说明它不是 Letta 的完整运行时。

125 题上报告间接题准确率、目标可见率、token、延迟和费用。构建成本和答题成本分开。统计用配对 bootstrap 和 McNemar。125 题上差一两个样本不作结论。

然后才做：无关问题反例、事实更新、LongMemEval 上的普通记忆是否变差。Jev-Mem 只在需要一个带 Jev 的检索父系统时再接。

## 7. 停止条件

- DeepSeek 在 Oracle 上用不好已经可见的事实：停止，换答题模型。
- 512 token 上 Jev 的答题不高于挑短句，128 token 上也不高于文献启发式：可见率再高也停止，不写系统论文。
- Jev 和同预算语言模型 profile 的准确率接近，但没有明确的费用或延迟优势：改写成效率比较；如果连这个都没有，就不作为独立贡献。

## 8. 明确后置

这些不在当前门槛里：合成集上重调权重、真实 tokenizer、多样性约束、背包装箱、校准曲线、ZDR、LoCoMo、DolphinBench、六个检索系统重跑、逐轮重放 47 个 session。

## 9. 文件

```text
jevtest/
  configs/pilot.json
  data/inmind.jsonl
  data/lme_s_background.jsonl
  data/prompts/                 InMind 官方提示词
  jev_persist/                  装箱、请求生成、DeepSeek 执行入口
  results/jev_admission_cache.jsonl
  results/framework/failure_cases.json
  results/framework/pilot_manifest.json
  results/framework/packing_all_tasks.jsonl
  results/framework/requests/
  tests/test_framework.py
```

下一步只有一件事：确认 20 题的 Oracle 和 core 请求可以发送，然后运行

```bash
python3 -m jev_persist.run_deepseek --execute
```
