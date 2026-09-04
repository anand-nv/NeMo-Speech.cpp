# Native Kokoro TTS

This directory implements the `hexgrad/Kokoro-82M` v1.0 model with GGML and a
native C++ behavioral port of the Misaki 0.9.4 frontend. The installed runtime
does not invoke Python or ONNX Runtime. One converted GGUF contains the model,
Kokoro vocabulary, pinned Misaki lexicons, and all 54 voice tables.

## Build

Kokoro is opt-in. ICU and the eSpeak-NG library/data must be installed before
configuring:

```bash
cmake -S . -B build-kokoro \
  -DNEMO_SPEECH_BUILD_TTS=ON \
  -DNEMO_SPEECH_TTS_WITH_KOKORO=ON
cmake --build build-kokoro -j
```

English, Mandarin, and the eSpeak-backed locales use native resources bundled
or linked by the build. Japanese additionally needs the exact UniDic data from
`unidic-lite==1.0.8`. Extract that package and provide its `dicdir` at
configure time so it is also copied by `cmake --install`:

```bash
python3 -m pip install --no-deps --target /tmp/unidic-lite unidic-lite==1.0.8
cmake -S . -B build-kokoro \
  -DNEMO_SPEECH_BUILD_TTS=ON \
  -DNEMO_SPEECH_TTS_WITH_KOKORO=ON \
  -DNEMO_SPEECH_KOKORO_UNIDIC_DIR=/tmp/unidic-lite/unidic_lite/dicdir
cmake --build build-kokoro -j
```

For a non-installed development build, `NEMO_SPEECH_KOKORO_UNIDIC_DIR` can
override the dictionary location at runtime. The loader validates the pinned
dictionary artifacts before use.

## Convert and run

Conversion is a Python build-time operation. Pin a Hugging Face revision for a
reproducible artifact; F16 is the default and F32 is useful for parity tests:

```bash
python3 convert_model.py hexgrad/Kokoro-82M \
  --architecture kokoro --revision f3ff3571791e39611d31c381e3a41a3af07b4987 \
  --outfile models/kokoro-v1_0.f16.gguf --outtype f16
python3 convert_model.py hexgrad/Kokoro-82M \
  --architecture kokoro --revision f3ff3571791e39611d31c381e3a41a3af07b4987 \
  --outfile models/kokoro-v1_0.f32.gguf --outtype f32

build-kokoro/bin/nemo-speech synthesize "Hello from Kokoro." \
  --kokoro-model models/kokoro-v1_0.f16.gguf \
  --voice af_heart --language en-US --speed 1.0 --output hello.wav
build-kokoro/bin/nemo-speech synthesize "日本語です。" \
  --kokoro-model models/kokoro-v1_0.f16.gguf \
  --voice jf_alpha --language ja-JP --output japanese.wav
build-kokoro/bin/nemo-speech synthesize "你好！" \
  --kokoro-model models/kokoro-v1_0.f16.gguf \
  --voice zf_xiaobei --language zh-CN --output mandarin.wav
```

## Regenerate reference fixtures

The developer-only oracle exporter records the pinned Misaki frontend and all
major PyTorch model boundaries as little-endian tensors. Its PyTorch vocoder
run injects the runtime's `splitmix64-box-muller-v1` source generator, making
the waveform fixture independent of PyTorch RNG implementation details and
comparable across graph tiling. Create an isolated
environment from `scripts/tts/requirements-kokoro-reference.txt`, check out
Misaki and Kokoro at the commits enforced by the script, then run:

```bash
python scripts/tts/export_kokoro_frontend_reference.py \
  --misaki-root /path/to/misaki \
  --kokoro-root /path/to/kokoro \
  --model-root /path/to/Kokoro-82M-snapshot \
  --cases scripts/tts/kokoro_model_reference_case.json \
  --output /path/to/reference.json
```

Each model-enabled case must contain `id`, `text`, `kokoro_language`, and
`voice`. The JSON manifest hashes the checkpoint, config, used voice files,
and every emitted tensor. Omitting `--model-root` produces frontend-only
fixtures without loading model weights.

Configure model differential tests by setting both paths before CMake runs:

```bash
NEMO_SPEECH_TEST_KOKORO_MODEL=/path/to/kokoro-v1_0.f32.gguf \
NEMO_SPEECH_TEST_KOKORO_REFERENCE=/path/to/reference.json \
  cmake -S . -B build-kokoro-tests \
    -DNEMO_SPEECH_BUILD_TESTS=ON \
    -DNEMO_SPEECH_TTS_WITH_KOKORO=ON
```

CUDA CTest entries select the GPU explicitly and run with TF32 disabled and
F32 cuBLAS accumulation. This is the numerical-parity mode used for the
`cosine >= 0.999` and `MAE <= 1e-3` waveform release gate; normal CUDA builds
remain free to use GGML's faster default precision policy.

Supported canonical locales are `en-US`, `en-GB`, `es-ES`, `fr-FR`, `hi-IN`,
`it-IT`, `ja-JP`, `pt-BR`, and `zh-CN`. If language is omitted it is inferred
from the selected voice. Speed must be in `[0.5, 2.0]`. A fixed seed makes a
request byte-repeatable.

The public C API selects Kokoro by setting only
`nemo_speech_tts_model_config.kokoro_model`. Raw-text and unframed phoneme-ID
requests use the existing `nemo_speech_tts_synthesize_text` and
`nemo_speech_tts_synthesize_tokens` functions. Callbacks receive mono s16le PCM
at 24 kHz, or at the requested lower output rate. A linguistic chunk is emitted
in 500 ms PCM tiles. The vocoder pulls bounded decoder ranges, regenerates
harmonic/STFT and convolutional windows by absolute index, and discards overlap
before feeding a persistent bounded inverse-STFT ring. Time-global AdaIN sees
the complete bounded inference leaf, so concatenated tiles are byte-identical
to the whole-graph result for a fixed seed. Predicted sequences above the
1,064-frame generator capacity are split recursively at phoneme boundaries;
the graph cache retains one tile shape rather than growing with callback count.
Cancellation is checked before the next decoder/generator tile and never
flushes a synthetic tail.

The tokenizer source map, exact Misaki pin, immutable data hashes, and known
compatibility boundaries are recorded in
[`tokenizer/MISAKI_PORT.md`](tokenizer/MISAKI_PORT.md).
