# Third-party notices

Portions of the `official-deepresearch` compatibility adapters and prompt constants are adapted from:

- Alibaba-NLP/DeepResearch, commit `f72f75d8c3eb842f2bbbab096a12206ff66e270f`
- Copyright Alibaba-NLP/DeepResearch contributors
- Licensed under the [Apache License 2.0](https://github.com/Alibaba-NLP/DeepResearch/blob/f72f75d8c3eb842f2bbbab096a12206ff66e270f/LICENSE)
- Source files: `inference/react_agent.py`, `inference/tool_search.py`, `inference/tool_visit.py`, `inference/prompt.py`, `evaluation/evaluate_deepsearch_official.py`, and `evaluation/prompt.py`
- Modifications: removed Qwen-Agent/vLLM runtime coupling; added dependency injection, cache and manifest integration, stable error records, environment-variable indirection, a paid-call authorization gate, and BrowseConf verbalized-confidence output.

The BrowseComp grading protocol is also cross-checked against OpenAI/simple-evals commit `652c89d0ca9df547706735883097e9537d40dc47`, licensed under MIT. No OpenAI dataset content or decrypted benchmark answers are redistributed here.
