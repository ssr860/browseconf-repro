# BrowseConf reproduction

这是一个可审计、可断点续跑的 BrowseConf 复现框架。它将论文算法、DeepResearch Web 工具链和 BrowseComp Judge 分成独立层，并且所有真实网络命令都需要显式传入 `--allow-paid-apis`。

上游适配固定为 [`Alibaba-NLP/DeepResearch@f72f75d8c`](https://github.com/Alibaba-NLP/DeepResearch/tree/f72f75d8c3eb842f2bbbab096a12206ff66e270f)。

## 三种后端

| 后端 | 用途 | 论文精确性 |
|---|---|---|
| `paper-browseconf` | 正式复现实验 | 使用论文 Prompt、128K context、overflow 失败、0.6/0.95、论文模型约束 |
| `official-deepresearch` | 对齐固定 commit 的上游一次 Web Agent attempt | 使用上游 110K 主动收尾、presence penalty 和 logprobs，不等于论文设置 |
| `local` | 保留项目原有的轻量并发实现 | 工程对照组 |
| `offline-mock` | 测试、教学和 CI | 完全离线、零费用 |

整体链路：

```text
dataset -> BrowseConf (Zero / Summary / Neg)
        -> Agent -> Serper Search -> Jina page -> summary model
        -> threshold stop / N 次后最高 confidence
        -> Judge -> calibration / accuracy / SC / CISC / Pass@K
```

## 五分钟零成本验证

```powershell
cd E:\BC
python -m pip install -e ".[dev,official]"
python -m browseconf smoke
python -m browseconf demo
python -m pytest
python -m ruff check src tests
python -m browseconf run --config configs/mock.json --dataset data/example.jsonl --limit 1 --method zero --threshold 95 --max-attempts 1 --output artifacts/mock.jsonl
```

这些命令不会调用外部 API。

## 下一步阅读

1. [保姆级使用说明](docs/CODE_USAGE.md)：从 API Key 到单题、20 条 calibration 和完整实验。
2. [来源与复现边界](docs/REPRODUCIBILITY.md)：官方实现、固定 commit、逐项差异和无法消除的缺口。
3. [论文 Prompt](prompts/PAPER_PROMPTS.md) 与 [Prompt 哈希](prompts/SHA256SUMS.json)。
4. [第三方许可](THIRD_PARTY_NOTICES.md)。

密钥绝不能写进 JSON。`configs/paper-browseconf.json` 和 `configs/example.json` 只保存环境变量名称；本机实际配置请复制为已被 `.gitignore` 排除的 `configs/local.json`。
