# OmniVoice

NeMo-Speech.cpp can run the pinned `k2-fsa/OmniVoice` checkpoint entirely in
native C++/GGML. The installed runtime implements the checkpoint's Qwen2
byte-level BPE frontend, diffusion-style eight-codebook generator, Higgs Audio
V2 prompt encoder and waveform decoder. Python is used only to convert the
published checkpoint and for release evaluation.

OmniVoice supports automatic voices, instruction-based voice design, and
voice cloning from reference PCM plus its transcript. It produces mono 24 kHz
audio, with optional downsampling through the shared output resampler. The
native API provides offline synthesis, finalized-chunk output streaming, and
bidirectional UTF-8 text/finalized-audio streaming. Because the denoiser is
bidirectional, audio becomes final at a committed linguistic chunk—not at an
individual acoustic frame.

Language IDs and names follow the 646-entry model table. Pass `None` as the
language value for language-agnostic inference; an unknown value follows the
upstream warning-and-fallback behavior.

## License and pinned inputs

The OmniVoice source implementation is Apache-2.0. The published checkpoint
states CC-BY-NC, and its bundled `audio_tokenizer/LICENSE` contains the Boson
Higgs Audio 2 Community License Agreement. Treat both as applicable and review
their use and redistribution terms before downloading or converting weights.
Converted GGUFs retain source, revision, hashes, and license metadata and are
not distributed by this project.

This implementation is pinned to Hugging Face revision
`c5fdb5ccb189668d56333f77ba2629f4cd7535f4`:

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.safetensors` | 2,450,344,112 | `730839316de585f4c8298ec0e1712efc10fb19c6fa4e36eb741cb8d51ebcf6aa` |
| `audio_tokenizer/model.safetensors` | 805,665,628 | `fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d` |
| `tokenizer.json` | 11,423,986 | `408f669b7e2b045fdf54201d815bd364e6667dbd845115da81239c40bc6dcfd1` |

Download that immutable snapshot and convert both required models:

```bash
hf download k2-fsa/OmniVoice \
  --revision c5fdb5ccb189668d56333f77ba2629f4cd7535f4 \
  --local-dir models/omnivoice/upstream

python3 convert_model.py models/omnivoice/upstream \
  --architecture omnivoice --outtype f16 \
  --outfile models/omnivoice/omnivoice.f16.gguf

python3 convert_model.py models/omnivoice/upstream/audio_tokenizer \
  --architecture higgs_audio_v2_tokenizer --outtype f16 \
  --outfile models/omnivoice/higgs-audio-v2-tokenizer.f16.gguf
```

Use `--outtype f32` for the numerical reference build. F16 and F32 are the only
qualified formats. Conversion validates pinned hashes, exact architecture and
tokenizer structure, all reachable tensor shapes, and the codec stride graph.
The two GGUFs are fingerprinted and cannot be mixed across source revisions.

## Build

OmniVoice is opt-in. For CPU:

```bash
cmake -S . -B build-omnivoice -G Ninja \
  -DNEMO_SPEECH_BUILD_TTS=ON \
  -DNEMO_SPEECH_TTS_WITH_OMNIVOICE=ON \
  -DNEMO_SPEECH_BUILD_EXAMPLES=ON
cmake --build build-omnivoice -j
```

For CUDA, initialize and patch the pinned ggml checkout as described in the
main build guide, then configure with `-DGGML_CUDA=ON`. Select `cpu` or `cuda`
explicitly when qualifying a backend. The two published source files total
about 3.27 GB; loaded memory and workspace requirements are higher and depend
on dtype, backend, prompt length, and generated duration.

## CLI

Automatic voice on CPU:

```bash
build-omnivoice/bin/synthesize_text \
  --tts.omnivoice-model models/omnivoice/omnivoice.f16.gguf \
  --tts.audio-tokenizer-model models/omnivoice/higgs-audio-v2-tokenizer.f16.gguf \
  --tts.backend cpu --tts.language-code en \
  --tts.text "Hello from native OmniVoice." \
  --tts.wav-out speech.wav
```

Change `--tts.backend cpu` to `cuda` for CUDA. Voice design uses an instruction:

```bash
build-omnivoice/bin/synthesize_text \
  --tts.omnivoice-model models/omnivoice/omnivoice.f16.gguf \
  --tts.audio-tokenizer-model models/omnivoice/higgs-audio-v2-tokenizer.f16.gguf \
  --tts.language-code en --tts.instruction "male, high pitch" \
  --tts.text "This voice was designed from an instruction." \
  --tts.wav-out designed.wav
```

Voice cloning requires a transcript. Encode once and save the versioned,
model-bound prompt:

```bash
build-omnivoice/bin/synthesize_text \
  --tts.omnivoice-model models/omnivoice/omnivoice.f16.gguf \
  --tts.audio-tokenizer-model models/omnivoice/higgs-audio-v2-tokenizer.f16.gguf \
  --tts.prompt-wav reference.wav \
  --tts.prompt-text "The exact words spoken in the reference." \
  --tts.save-prompt voice.prompt

build-omnivoice/bin/synthesize_text \
  --tts.omnivoice-model models/omnivoice/omnivoice.f16.gguf \
  --tts.audio-tokenizer-model models/omnivoice/higgs-audio-v2-tokenizer.f16.gguf \
  --tts.prompt-file voice.prompt --tts.language-code fr \
  --tts.text "Bonjour depuis OmniVoice." --tts.wav-out cloned.wav
```

Treat prompt files as biometric-derived sensitive data. They contain acoustic
codes and the reference transcript, are not encrypted, and are accepted only
when their model and codec fingerprints match.

Use `--tts.bidirectional` with text or stdin to demonstrate committed-chunk
streaming. Arbitrary UTF-8 fragments are accepted; terminal punctuation
commits automatically and each input line is explicitly committed by the
example. The stable C ABI exposes the same behavior through
`nemo_speech_tts_stream_create`, `nemo_speech_tts_stream_push_text`, and
`nemo_speech_tts_stream_finish`. Concatenating output callbacks is identical to
offline output for the same seed and committed chunk sequence.

The unified CLI accepts the equivalent `nemo-speech synthesize` options
`--omnivoice-model`, `--audio-tokenizer-model`, `--instruction`,
`--prompt-wav`, `--prompt-text`, `--prompt-file`, and `--save-prompt`.

## Services

The HTTP `/v1/audio/speech` route accepts automatic voice, the OpenAI-style
`instructions` field for voice design, and OmniVoice `speed`. Ordinary `wav`
or `pcm` requests are complete responses. Set `stream: true` only with
`response_format: "pcm"`; finalized PCM chunks are sent as they are produced.
HTTP voice-clone audio upload is intentionally not exposed.

The Riva adapter advertises `omnivoice.auto`, native 24 kHz `LINEAR_PCM`, all
model language IDs, zero-shot, and voice-design capabilities. `ZeroShotData`
accepts raw little-endian PCM16 with `LINEAR_PCM`, its sample rate, and a
mandatory transcript. OmniVoice generation controls are strict
`custom_configuration` entries; unknown keys and unsupported encodings fail
with `INVALID_ARGUMENT`.

## MiniMax WER/CER release validation

The release job uses the first five non-empty lines from each of the 24
MiniMaxAI multilingual files (120 utterances). Selection hashes and ASR/data
revisions are checked in at
`scripts/tts/omnivoice_minimax_manifest.json`. Download the pinned references:

Install the release-only Python environment (choose the appropriate PyTorch
wheel for the evaluation host) and the pinned OmniVoice oracle checkout first:

```bash
python3 -m pip install -r scripts/tts/requirements-omnivoice-eval.txt
python3 -m pip install -e ../OmniVoice
```

```bash
hf download k2-fsa/TTS_eval_datasets --repo-type dataset \
  --revision 699beb854ab40a8a38bd48285e67e22697fc6e63 \
  --include minimax_multilingual_24.jsonl minimax_multilingual_24.tar.gz \
  --local-dir .eval/TTS_eval_datasets
tar -xzf .eval/TTS_eval_datasets/minimax_multilingual_24.tar.gz \
  -C .eval/TTS_eval_datasets

python3 scripts/tts/omnivoice_minimax_eval.py select \
  --source-root ../MiniMaxAI_TTS-Multilingual-Test-Set \
  --reference-manifest .eval/TTS_eval_datasets/minimax_multilingual_24.jsonl \
  --output .eval/minimax-120.jsonl
```

Generate the Python baseline and both native candidates with the same default
32-step configuration and seed:

```bash
python3 scripts/tts/omnivoice_minimax_eval.py oracle-synthesize \
  --omnivoice-root ../OmniVoice --model models/omnivoice/upstream \
  --manifest .eval/minimax-120.jsonl \
  --reference-root .eval/TTS_eval_datasets \
  --output-dir .eval/oracle --output-manifest .eval/oracle.jsonl

python3 scripts/tts/omnivoice_minimax_eval.py synthesize \
  --library build-omnivoice/bin/libnemo_speech_tts.so \
  --model models/omnivoice/omnivoice.f16.gguf \
  --codec models/omnivoice/higgs-audio-v2-tokenizer.f16.gguf \
  --backend cpu --manifest .eval/minimax-120.jsonl \
  --reference-root .eval/TTS_eval_datasets \
  --output-dir .eval/cpu --output-manifest .eval/cpu.jsonl

# Repeat with a CUDA build/library and --backend cuda.
```

Run `score` on each output manifest. It pins
`openai/whisper-large-v3-turbo` revision
`41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, forces the known language, and
uses the upstream OmniVoice normalization. It reports WER for word-delimited
languages and CER for Chinese, Cantonese, Japanese, Korean, Thai, Arabic,
Vietnamese, Hindi, and Greek. Finally, use `compare --baseline ...
--candidate ...`; the gate permits at most one absolute macro point and five
per-language points of regression, with no missing/failed ASR job.

This 120-item suite is a release/nightly job. Compact frontend, model,
prompt, offline/output-streaming, bidirectional-streaming, CPU, CUDA, legacy C
ABI, and Magpie compatibility tests remain suitable for normal CI.
