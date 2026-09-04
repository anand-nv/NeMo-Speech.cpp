# Native Misaki port manifest

The Kokoro tokenizer is pinned to Misaki 0.9.4 commit
`fba1236595f2d2bf21d414ba6e57d25256afada3`. Misaki is licensed under
Apache-2.0. This manifest is the review boundary for updating that pin.

| Pinned Misaki source | Native counterpart | Status |
| --- | --- | --- |
| `misaki/token.py::MToken` | `kokoro_tokenizer.h::MToken` | Structure, timestamps, source spans, and selected feature fields ported; checked-in English fixtures compare every field consumed natively |
| `misaki/espeak.py::EspeakG2P` | `espeak_g2p.cpp::EspeakG2P::phonemize` | Native eSpeak-NG C API and legacy replacements ported |
| `misaki/espeak.py::EspeakFallback` | `espeak_g2p.cpp::EspeakG2P::fallback` | Native eSpeak-NG C API and ordered replacements ported |
| `kokoro/pipeline.py::ALIASES` | `KokoroTokenizer::canonicalize_language` | Ported and extended with canonical Riva aliases |
| `kokoro/pipeline.py::waterfall_last` / `en_tokenize` | `KokoroTokenizer::split_source`, `split_unit_to_fit`, and `tokenize` | Sentence/clause/whitespace/codepoint waterfall and 400/510 limits ported; exact token-level reference boundaries remain under differential test |
| `misaki/en.py::G2P.preprocess`, `tokenize`, and `fold_left` | `english_g2p.cpp::preprocess_english`, `EnglishG2P::phonemize` | `[text](feature)` removal, direct phonemes, integer/half stress, `a`/`n`/`&` number flags, multi-token folding, processed text, and raw source spans ported and covered by the pinned differential oracle |
| `misaki/en.py::Lexicon` | `english_g2p.*::EnglishG2P` | Native gold/silver lookup, deterministic coarse POS, contextual function words, stress, morphology/numbers, Unicode possessives, and eSpeak fallback ported. The exhaustive load-once differential covers every US and GB gold/silver entry in its Misaki token context: US 183,561/183,561 and GB 197,116/197,116 exact phoneme matches |
| `misaki/data/{us,gb}_{gold,silver}.json` | `conversion/kokoro.py::_load_misaki_data`, GGUF `kokoro.tokenizer.misaki.*.json`, `EnglishG2P::parse_lexicon` | Exact-commit files and SHA-256 values embedded and loaded natively |
| `misaki/cutlet.py`, `num2kana.py`, `ja.py` | `japanese_g2p.*` | Native MeCab/UniDic readings, Cutlet normalization, number-to-kana, Hepburn/digraph and small-kana fallbacks, iteration/dakuten, nasal, sokuon, long-vowel, punctuation, and spacing paths ported |
| `misaki/data/ja_words.txt` | GGUF `kokoro.tokenizer.misaki.ja_words.text`, parsed by `JapaneseG2P` | Exact pinned file and SHA-256 embedded by the converter |
| `misaki/zh.py::ZHG2P(version=None)` | `mandarin_g2p.*` | Legacy Jieba segmentation with HMM enabled, base-pypinyin phrase/character lookup, `cn2an`-equivalent integer normalization, punctuation, and IPA/tone rewrites ported |
| `misaki/transcription.py` | `mandarin_g2p.cpp::pinyin_to_ipa` | Selected first-variant path ported; original transcription rules are MIT |
| `jieba==0.42.1` dictionary/HMM and base `pypinyin==0.55.0` tables | `src/tts/tokenizer/mandarin_data/{jieba.dict.utf8,hmm_model.utf8,misaki_pinyin_chars.tsv,misaki_pinyin_phrases.tsv}` | Generated from the pinned packages before loading CC-CEDICT, recorded in `manifest.json`, and SHA-256 validated by the installed runtime |

The runtime library contains no CPython, subprocess, tokenizer-service, or
ONNX fallback. Japanese morphology uses the native vendored MeCab engine and
the pinned UniDic-lite data directory; the immutable Misaki grouping list is
loaded from the model GGUF. The checked-in exact frontend corpus currently has
22 cases, including all nine Kokoro locales, Japanese kana/rule coverage,
Mandarin polyphones and number classes, punctuation, source spans, token IDs,
and long-text boundaries.

The exhaustive English audit is reproducible with
`scripts/tts/diff_kokoro_english.py --limit 0` plus its required Misaki,
model, native-runtime, and language arguments. It runs the native frontend in
one process so dictionary construction and token-context behavior are tested
together; both dialect runs must report zero mismatches before the Misaki pin
or lexicon logic changes.
