# 来源与复现边界

## 两套必须区分的 profile

`paper-browseconf` 是论文实验 profile：使用论文公开的 verbalized-confidence system prompt、128K context、temperature 0.6、top-p 0.95；context overflow 立即作为失败并赋 `confidence=-1`。页面证据 Summary 模型固定为 `gpt-oss-120b`，Judge 模板固定为 `gpt-4o-2024-08-06`。工具 XML/JSON schema 是文本式 OpenAI-compatible 接口所需的最小适配层，不属于论文 Table 3。

`official-deepresearch` 是固定上游 commit 的实现 profile：在 110×1024 token 时要求模型强制给最终答案，并带 `presence_penalty=1.1`、`logprobs=true` 等上游推理参数。它用于复核上游代码行为，不能代替论文 profile。

论文没有公开最大 model call、单次最大输出和墙钟限制等所有细节；`paper-browseconf` 对这些未报告项采用固定上游的 100 calls、10,000 output tokens 和 150 分钟，并写入 manifest。这是可审计近似，不是论文原文明示参数。

## 固定来源与许可

主上游是 [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)，审计和适配固定在 commit `f72f75d8c3eb842f2bbbab096a12206ff66e270f`（2026-02-27），许可证为 Apache-2.0。关键文件最后修改历史：

| 文件 | 最后影响该文件的 commit | 用途 |
|---|---|---|
| `inference/tool_search.py` | `ce278d31c0632bdd0710c451e0e4ca1d549d44d5` | Serper Search |
| `inference/tool_visit.py` | `c05398f91265a902c832eb2b8b8b95dfa1c7a511` | Jina、截断、摘要与解析 |
| `inference/prompt.py` | `ac2eea7e0a4e5097420b373d5e6f06a98414f97c` | Agent/Extractor Prompt 与工具 schema |
| `inference/react_agent.py` | `ea1f68dcd647d26ed21ea50333f1f2b84df42298` | ReAct 循环与采样参数 |
| `evaluation/evaluate_deepsearch_official.py` | `ce278d31c0632bdd0710c451e0e4ca1d549d44d5` | Judge、重试、并发与聚合 |
| `evaluation/prompt.py` | `92d94352f63a3b3ccd882439b7973ec4f49cd4c9` | BrowseComp Judge Prompt |

原始 BrowseComp 评测来源是 [OpenAI simple-evals](https://github.com/openai/simple-evals/tree/652c89d0ca9df547706735883097e9537d40dc47)，本次核验 SHA 为 `652c89d0ca9df547706735883097e9537d40dc47`，许可证为 MIT。`browsecomp_eval.py` 定义原始 grader prompt、加密数据加载和 `correct: (yes|no)` 解析。DeepResearch 使用同一 prompt 文本，但为 BrowseComp 增加 strict JSON schema，并固定 Judge 为 `gpt-4o-2024-08-06`。

DeepResearch `evaluation/README.md` 仍写着 `evaluate_all_official.py`，但完整上游 Git 历史中从未出现这个路径。首次公开发布 commit `ce278d31...` 已包含且实际使用 `evaluation/evaluate_deepsearch_official.py`；本项目以该真实文件为准，不凭 README 名称重写不存在的脚本。

本仓库 MIT 许可证不替代上游许可。适配代码注释和 manifest 保留仓库、commit、Apache-2.0 与“compatibility-adapter”标记；原始论文 Prompt 作为独立复现资产保留在 `prompts/`。

## 官方链路的实际行为

### Search

- 服务：`POST https://google.serper.dev/search`，key 来自 `SERPER_KEY_ID`，header 为 `X-API-KEY`。
- 英文 payload：`q`、`location=United States`、`gl=us`、`hl=en`；查询含基本 CJK 字符时切到 `China/cn/zh-cn`。
- 当前 `inference/tool_search.py` 不传 `num`，依赖 Serper 默认返回数量；工具 description 声称 top 10。本 adapter 明确最多渲染 10 条，避免服务默认漂移。
- 读取 `organic` 中必需的 `title/link` 和可选 `date/source/snippet`，删除固定视频错误短语，按官方文本格式返回。
- 单次网络连接失败最多重试 5 次；没有 `organic` 时返回可继续搜索的文本错误。
- schema 要求 `query` 数组，但实现也接受单字符串。当前 DeepResearch 多查询串行并以 `=======` 拼接；旧 `WebAgent/WebSailor/src/tool_search.py` 则传 `num=10`、`extendParams` 并用 3 worker 并发。本项目默认跟随当前 `inference/`，而不是旧 WebSailor 分支。

### Visit 与页面证据提取

- Jina：`GET https://r.jina.ai/{url}`，`Authorization: Bearer $JINA_API_KEYS`，单请求 timeout 50 秒、每轮 3 次，外层最多 8 轮。
- 当前上游没有缓存；本项目在 adapter 边界增加按 URL/请求 payload 的文件缓存，以满足断点与成本控制。缓存不改变返回文本，但意味着复跑不会观察实时页面变化。
- 上游用 `cl100k_base` 把正文截到 95,000 tokens。安装 `tiktoken` 时本项目完全采用该编码；未安装时使用保守字符启发式，并在 tool metadata 记录 `truncation=heuristic`。
- 正文通过单条 user message 交给 summary model；Extractor Prompt 与上游文本一致，要求 JSON `rational/evidence/summary`。实际渲染只使用 `evidence/summary`。
- summary model 温度 0.7。空或极短输出会逐次把正文缩到 70%，最终缩到 25,000 字符；JSON 解析失败再请求最多 3 次。仍失败时返回稳定的“页面不可访问/无法处理”证据与摘要文本。
- 当前上游多 URL 串行，整个列表超过 900 秒后为剩余 URL 生成失败文本；旧 WebSailor 版本用 3 worker 并发。本 adapter 跟随当前实现并保留逐 URL 错误，不让一个 URL 丢掉整批结果。
- 本项目额外拒绝非 HTTP(S) URL，并缓存成功页面；这是安全与可复现性适配，不是上游逐行复制。

### Agent

- 官方 tool 名称为 `search` 与 `visit`，调用格式为 `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`，工具结果作为 user message 放入 `<tool_response>...</tool_response>`。上游通过 `json5.loads`；本项目支持标准 JSON，并安全兼容常见单引号/尾逗号形式。
- 官方完整 DeepResearch prompt 还暴露 `PythonInterpreter`、`google_scholar`、`parse_file`。本仓库的 Web-only attempt 只暴露已实现和可审计的 Search/Visit，避免模型调用不存在或不安全的沙箱工具。
- 默认最多 100 次 LLM call；每题还有 150 分钟墙钟上限。模型请求最多重试 10 次，timeout 600 秒，stop tokens 为换行版和非换行版 `<tool_response>`。
- 当前上游默认/CLI 参数为 temperature 0.6、top_p 0.95、presence penalty 1.1、`logprobs=True`、每次最多 10,000 output tokens。`.env.example` 的脚本模板可把 temperature 覆盖成 0.85；必须在 manifest 中以实际配置为准。
- 上游用模型 tokenizer 计算上下文并在超过 `110 * 1024` tokens 时强制最后回答。本项目默认同一阈值和强制回答语义，但在没有模型 tokenizer 时使用可审计的字符估算；这是主要近似点。
- BrowseConf 要求 verbalized confidence，因此本项目只在官方 system prompt 末尾增加 `Answer/Confidence` 最终格式。这是算法所需适配，不属于 DeepResearch 原始 prompt。

### Evaluation / Judge

- DeepResearch 对 BrowseComp-en 和 BrowseComp-zh 都选择 `gpt-4o-2024-08-06`。
- 输入为一条 user message：question、完整 prediction、gold answer 填入原始 BrowseComp grader prompt。Agent 的 response 被格式化为 `Answer: ...\nConfidence: ...`。
- DeepResearch 请求 strict JSON schema：`extracted_final_answer`、`reasoning`、`correct=yes|no`、`confidence`、`strict`。BrowseComp 分支没有显式 temperature 或 max token；OpenAI API 默认温度等价于 1。本项目显式记录 temperature 1；`configs/example.json` 可设置可审计的输出上限。
- LiteLLM 内层 `num_retries=5`；外层循环写成 100，但异常分支在第五次（索引 4）就返回 Error，因此有效外层最多 5 次、每次间隔 3 秒。本项目保留 5 次外层重试，底层 OpenAI-compatible client 也可配置 5 次。
- 官方脚本用 100 worker 并发逐 round 判断。本项目 `judge --workers` 默认 100，但建议 pilot 降到 1–5；JSONL 支持断点续跑。
- 结构化 JSON 无法解析时，本项目先兼容原始 `correct: yes|no`，仍失败记 `correct=null` 和 `judge_parse_failure`，不会静默算错。原始 simple-evals 当前 `match.group(0)` 与后续裸 `yes/no` 比较存在已知不一致；DeepResearch 的 JSON 路径不受该 bug 影响。

## 差异表与集成选择

| 环节 | 改造前本项目 | 固定上游 | 当前实现 |
|---|---|---|---|
| Search key/locale | `GOOGLE_SEARCH_KEY`；固定 country/language | `SERPER_KEY_ID`；按 CJK 自动 US/CN | 官方 adapter 对齐；local 后端保留旧行为 |
| Search payload | `q,num,gl[,hl]` | `q,location,gl,hl`，无 `num` | 官方 payload；渲染上限固定 10 |
| 批量 Search | 3 worker 并发 | 当前串行；旧 WebSailor 3 worker | 官方串行，local 并发 |
| Jina | 30 秒、一次 client 重试层、150k chars | 50 秒，3×8 重试，95k tokens | 官方重试；可选 cl100k，带缓存 |
| 多 URL | 3 worker 并发 | 当前串行、900 秒总限 | 官方串行和总限，逐 URL 隔离错误 |
| Extractor | 简化二字段 Prompt、一次解析 | 三字段原 Prompt、短输出/解析重试 | 原 Prompt 与重试，输出 evidence/summary |
| Agent prompt | 自定义 Web master protocol | DeepResearch system/tool XML | 上游 profile 跟随官方；论文 profile 使用论文 Prompt 加最小工具 schema |
| Agent 限制 | 80 interactions、131072 估算 tokens | 100 LLM calls、112640 tokenizer tokens、150 分钟 | 上游 profile 为 112640+强制答案；论文 profile 为 131072+失败 |
| 采样 | temp .6/top_p .95 | 同左 + presence 1.1/logprobs/max 10000 | 已对齐并进入请求 payload |
| Judge | 缩写 Prompt、无 schema、temp 0、1024 tokens | 完整 Prompt、strict schema、GPT-4o snapshot、有效 5×内部重试 | 完整 Prompt/schema/snapshot 配置与显式失败 |
| 缓存/manifest | 本项目已有 | 上游工具没有统一缓存/manifest | 继续保留且记录 adapter/commit，不写 key |
| BrowseConf policy | Zero/Summary/Neg | 非 DeepResearch 工具层职责 | 完全保留并与 adapter 解耦 |

选择“本项目内的忠实 adapter”，而不是运行时 import 或 vendor 整仓：上游模块在 import 阶段依赖 Qwen-Agent、OpenAI SDK、Transformers、tiktoken、vLLM/本地端口并读取全局环境变量；直接依赖会把训练/推理基础设施、未使用工具和隐式副作用带入复现。当前 adapter 固定来源、接口小、可注入 mock、无需真实网络即可逐字段测试。没有把整个上游机械复制进发行包。

## BrowseConf 自研层

以下保持为本项目/论文算法逻辑，而非伪装成 DeepResearch：verbalized confidence、阈值提前停止、最多 N 次 attempt、预算耗尽后取最高 confidence（并列固定取最早 attempt）、Zero、跨轨迹 Summary、Neg、calibration、self-consistency、CISC 和 Pass@K。Search/Visit/Judge 只实现一次 attempt 和最终评分所需的外部边界。

## 不可消除的缺口与实验约束

- 动态 Web、Serper 排名和网页正文会随时间/地区/账号变化；固定代码 commit 不能固定搜索环境。缓存、时间戳、查询、URL 和原始 tool event 必须随结果保存。
- 搜索可能命中公开 benchmark 答案。不得将 gold answer 传入 Agent/Search；应标记疑似泄漏 URL，并分别报告包含和排除污染样本的结果。BrowseComp 数据自带 canary，发布结果时不要泄漏解密后的题目/答案。
- 上游没有锁定 summary model 的唯一公开 snapshot，`.env.example` 只提供 `SUMMARY_MODEL_NAME`；这会显著影响证据提取。
- BrowseConf 论文明确说明 Summary 使用 `gpt-oss-120b`，但没有公布服务商 endpoint 或精确权重 revision；配置只能锁定论文名称，正式报告仍需记录实际供应商 revision。
- 论文的 500 条 SailorFog-QA calibration 样本 ID 未公开。没有作者数据划分时只能做方法级或替代 calibration，不能声称复现相同 τ。
- 本项目没有复刻 8 个 vLLM server、模型 tokenizer、Python/Scholar/File 工具或旧 WebSailor 的并发时序，因此不是整套基础设施的逐比特复现。
- BrowseConf 论文中的某些生产搜索后端、完整运行日期、服务端采样实现和成本环境无法从公开材料完全恢复。应报告模型精确 revision、服务商、运行日期、地区、缓存状态和实际 manifest。
- 当前工作区没有 `.git` 目录，因此本次被合并/删除的旧过程文档无法依赖当前 checkout 的 Git 历史恢复；在有正式远端的版本中应通过 commit 保存本次重构前状态。
