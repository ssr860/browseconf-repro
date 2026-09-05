# 保姆级使用说明（Windows PowerShell）

本说明假设你是第一次跑 Agent 论文代码。每一步都在仓库根目录 `E:\BC` 执行。

## 0. 先弄清楚你手里的 Key

完整 Web Agent 链路包含三类服务：

| 凭证 | 做什么 | 是否必需 |
|---|---|---|
| 模型服务 Key | Agent、网页摘要、Judge | 必需；同一供应商支持所需模型时可共用一个 Key |
| `SERPER_KEY_ID` | Google 搜索结果 | 真实 Web 运行必需 |
| `JINA_API_KEYS` | 将网页转换为正文 | 代码允许留空匿名访问，但正式实验建议申请并配置 |

“只有一个模型 API Key”并不自动包含 Serper 搜索能力。如果还没有 Serper Key，只能先跑离线 mock；申请 Serper 后才能跑真实 Search。严格论文模型还要求供应商提供 `gpt-oss-120b`，官方 Judge 要求 `gpt-4o-2024-08-06`。现实中经常需要两个模型供应商；若用同一个可用模型替代三个角色，可以跑通方法，但不能称为模型级严格复现。

## 1. 安装 Python 与项目

```powershell
python --version
cd E:\BC
python -m pip install -e ".[dev,official]"
```

需要 Python 3.11 或更高版本。如果 `python` 命令不存在，先安装 Python，并在安装器中勾选 “Add Python to PATH”。

## 2. 先做零成本验证

```powershell
python -m browseconf smoke
python -m browseconf demo
python -m pytest
python -m ruff check src tests
python -m browseconf run --config configs/mock.json --dataset data/example.jsonl --limit 1 --method zero --threshold 95 --max-attempts 1 --output artifacts/mock.jsonl
```

预期：测试全部通过，最后生成 `artifacts/mock.jsonl` 及相邻 manifest。以上命令完全离线。

## 3. 建立只在本机使用的配置

```powershell
Copy-Item configs/paper-browseconf.json configs/local.json
notepad configs/local.json
```

`configs/local.json` 已被 `.gitignore` 排除。只修改 `base_url` 和供应商实际使用的 `model_id`；`api_key_env` 保持为 `MODEL_API_KEY`。绝对不要把真实 Key 写入 JSON。

严格论文配置：Agent 使用论文实验模型之一（`gpt-oss-120b` 或 `DeepSeek-V3.1`），Summary 使用 `gpt-oss-120b`，Judge 使用 `gpt-4o-2024-08-06`。如果供应商模型 ID 带命名空间，以其真实 ID 为准，并在报告中记录差异。

## 4. 在当前 PowerShell 设置环境变量

```powershell
$env:MODEL_API_KEY = "粘贴你的模型API-Key"
$env:SERPER_KEY_ID = "粘贴你的Serper-Key"
$env:JINA_API_KEYS = "粘贴你的Jina-Key"
```

同一个 `MODEL_API_KEY` 会被 Agent、Summary 和 Judge 共用。若 Judge 使用另一家供应商，把 `judge_model.api_key_env` 改成 `JUDGE_MODEL_API_KEY`，然后设置：

```powershell
$env:JUDGE_MODEL_API_KEY = "粘贴Judge供应商Key"
```

环境变量只对当前 PowerShell 有效，关闭窗口后需要重设。官方 OpenAI 文档也推荐从环境变量加载 API Key，而不是硬编码。

## 5. 花钱前运行 doctor

```powershell
python -m browseconf doctor --config configs/local.json
```

它不会联网，也不会打印 Key。只有 `ready: true` 才建议继续。常见提示：

- `MODEL_API_KEY` missing：当前窗口没有设置模型 Key。
- `SERPER_KEY_ID` missing：无法真实搜索。
- `replace placeholder base_url`：还没替换模板 URL。
- `paper mismatch`：改动了论文锁定参数；仍可运行，但不再是严格设置。

## 6. 准备数据

JSONL 是“一行一个 JSON”，每行至少包含：

```json
{"question_id":"q1","question":"问题文本","answer":"标准答案"}
```

将 20 条公开 SailorFog-QA 转成 `data/sailorfog_calibration_20.jsonl`。`answer` 只交给 Judge，不会传给 Agent 或搜索。不要用 BrowseComp 测试题调整阈值。

论文使用 500 条未完整公开的 SailorFog-QA calibration 样本。20 条只适合验证链路、费用并得到临时 `tau_pilot`，不能声称复现论文正式 τ。

## 7. 第一次真实调用：只跑一题一次

```powershell
python -m browseconf fixed-pool --config configs/local.json --dataset data/sailorfog_calibration_20.jsonl --limit 1 --rollouts 1 --workers 1 --output artifacts/pilot-1.jsonl --allow-paid-apis
```

`--allow-paid-apis` 是明确授权闸门，不是费用上限。第一次务必保留 `--limit 1 --rollouts 1 --workers 1`。然后检查：

```powershell
Get-Content artifacts/pilot-1.jsonl
Get-Content artifacts/pilot-1.jsonl.manifest.json
```

重点看 `answer`、`confidence`、`stop_reason`、`model_call_count`、`tool_events`、`usage` 和 `error`。

## 8. 对单题运行 Judge

```powershell
python -m browseconf judge --config configs/local.json --dataset data/sailorfog_calibration_20.jsonl --limit 1 --predictions artifacts/pilot-1.jsonl --workers 1 --output artifacts/pilot-1.judgements.jsonl --allow-paid-apis
Get-Content artifacts/pilot-1.judgements.jsonl
```

`correct: true/false` 表示 Judge 跑通；`correct: null` 时查看 `error`。

## 9. 用 20 条得到临时 tau_pilot

```powershell
python -m browseconf fixed-pool --config configs/local.json --dataset data/sailorfog_calibration_20.jsonl --rollouts 1 --workers 1 --output artifacts/calibration20.jsonl --allow-paid-apis
python -m browseconf judge --config configs/local.json --dataset data/sailorfog_calibration_20.jsonl --predictions artifacts/calibration20.jsonl --workers 1 --output artifacts/calibration20.judgements.jsonl --allow-paid-apis
python -m browseconf calibrate --predictions artifacts/calibration20.jsonl --judgements artifacts/calibration20.judgements.jsonl --k 10 --min-count 5 --output artifacts/tau-pilot.json
Get-Content artifacts/tau-pilot.json
```

记录输出的 `threshold`。下面用 `TAU` 代表这个整数。

## 10. 跑 BrowseConf 三个变体

先保留 `--limit 1`，确认费用正常后再改成 20 或删除：

```powershell
python -m browseconf run --config configs/local.json --dataset data/questions.jsonl --limit 1 --method zero --threshold TAU --max-attempts 10 --workers 1 --output artifacts/zero.jsonl --allow-paid-apis
python -m browseconf run --config configs/local.json --dataset data/questions.jsonl --limit 1 --method summary --threshold TAU --max-attempts 10 --workers 1 --output artifacts/summary.jsonl --allow-paid-apis
python -m browseconf run --config configs/local.json --dataset data/questions.jsonl --limit 1 --method neg --threshold TAU --max-attempts 10 --workers 1 --output artifacts/neg.jsonl --allow-paid-apis
```

把字面量 `TAU` 替换为真实数字，例如 `95`。JSONL 支持断点续跑；若更换模型、阈值或数据，请换输出文件名。

## 11. Judge 与报告

以 Zero 为例：

```powershell
python -m browseconf judge --config configs/local.json --dataset data/questions.jsonl --predictions artifacts/zero.jsonl --workers 1 --output artifacts/zero.judgements.jsonl --allow-paid-apis
python -m browseconf report --predictions artifacts/zero.jsonl --judgements artifacts/zero.judgements.jsonl --output artifacts/zero.report.json
Get-Content artifacts/zero.report.json
```

Summary 与 Neg 重复操作，仅替换文件名。

## 12. SC、CISC 和 Pass@10

固定 10 次采样明显更贵：

```powershell
python -m browseconf fixed-pool --config configs/local.json --dataset data/questions.jsonl --limit 1 --rollouts 10 --workers 1 --output artifacts/pool10.jsonl --allow-paid-apis
python -m browseconf aggregate --input artifacts/pool10.jsonl --strategy sc --output artifacts/sc.jsonl
python -m browseconf aggregate --input artifacts/pool10.jsonl --strategy cisc --temperature 10 --output artifacts/cisc.jsonl
python -m browseconf judge --config configs/local.json --dataset data/questions.jsonl --predictions artifacts/pool10.jsonl --workers 1 --output artifacts/pool10.judgements.jsonl --allow-paid-apis
python -m browseconf fixed-report --pool artifacts/pool10.jsonl --judgements artifacts/pool10.judgements.jsonl --k 10 --output artifacts/pass10.report.json
```

## 13. 正式实验检查清单

- 用独立 500 条开发数据重新估计正式 τ；拿不到论文原始划分时写明“替代 calibration”。
- 保存数据哈希、结果和 `.manifest.json`，但不要公开受限制 benchmark 内容。
- 记录模型精确 ID、供应商、日期、地区、缓存状态、τ、N 和 worker 数。
- 先依据一题的 token 和账单估算总费用。
- 根据 `tool_events` 标记疑似 benchmark 泄漏 URL，并分别报告包含/排除后的结果。

## 14. Key 与 Git 安全

可以提交 `configs/example.json` 和 `configs/paper-browseconf.json`，因为它们只有环境变量名称。不要提交 `configs/local.json`、`.env*`、含 Bearer token 的日志以及 `artifacts/`。

```powershell
git status --short
git grep --cached -n -I -E 'sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9_-]{16,}'
```

第二条没有输出通常表示未匹配到常见密钥。如果 Key 曾进入 Git 历史，删除文件仍不够，必须立即撤销并重新生成。
