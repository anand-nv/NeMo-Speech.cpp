# Mandarin G2P data

These generated runtime tables reproduce the Mandarin frontend used by the
MagpieTTS NeMo configuration. They are generated from Jieba 0.42.1, pypinyin
0.55.0, and pypinyin-dict 0.9.0 by
`scripts/tts/generate_mandarin_tokenizer_data.py`.

CppJieba, Jieba, pypinyin, and pypinyin-dict are distributed under the MIT
license. CppJieba and its limonp dependency retain their license files in their
submodule directories. Project-level attribution is consolidated in the root
`THIRD_PARTY_NOTICES.md`. Source versions and generated-file checksums are
recorded in `manifest.json`.

The `misaki_pinyin_*.tsv` siblings capture unmodified pypinyin before the
Magpie-specific CC-CEDICT override. Kokoro v1.0 loads these files to reproduce
Misaki 0.9.4's legacy `ZHG2P(version=None)` path. Both table families and their
hashes are produced in one deterministic generator invocation.

Regenerate the runtime tables with the pinned Python packages installed:

```bash
python3 -m pip install jieba==0.42.1 pypinyin==0.55.0 pypinyin-dict==0.9.0
python3 scripts/tts/generate_mandarin_tokenizer_data.py \
  --output-dir src/tts/tokenizer/mandarin_data
```
