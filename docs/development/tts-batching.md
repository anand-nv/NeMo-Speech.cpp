# TTS in-flight batching

MagpieTTS supports in-flight (continuous) batching: concurrent synthesis
requests share batched decoder steps while each one keeps its own text, voice,
sampling settings, random stream and streamed audio. A request joins the batch
as soon as a slot is free. It leaves in the step where it ends (EOS, its step
limit or a cancellation), and its slot goes to the next waiting request.
Batching is opt-in: with `tts.max-batch-size` at `1` (the default) the runtime
synthesizes one request at a time exactly as before.

## Enabling

Set `tts.max-batch-size` to the number of requests that may decode together:

```bash
nemo-speech serve --config config/tts.example.yaml --tts.max-batch-size 8
```

The HTTP server, `riva_server`, the C API (`nemo_speech_tts_runtime_config.max_batch_size`)
and the C++ `MagpieTtsRuntime` (`MagpieRuntimeConfig::max_batch_size`) all use
the same setting. Concurrent `synthesize` calls are batched automatically,
with no other client change.

To measure it:

```bash
nemo-speech bench tts --text-file utterances.txt \
    --magpie-model magpie.q8_0.gguf --codec-model codec.gguf --tokenizer-dir extracted \
    --concurrency 1,4,8 --max-batch-size 8
```

Requirements and limits:

- The local transformer must be enabled. All requests use the engine's
  `tts.use-cfg` setting: every request runs classifier-free guidance, or none
  does.
- A request's widest text window (sentence chunk plus up to 20 tokens of
  history) must fit the engine's text capacity of 512 tokens. Its decoder steps
  must fit the per-slot cache, sized for the model's `max_decoder_steps`.
  Requests outside these limits are rejected at submission.
- Memory: each slot holds an F32 self-attention cache per lane (about 45 MB per
  lane for the 357M model, twice that with CFG) and its text's cross-attention
  rows.

## Fast path and fallback

With a Q8_0 model on an sm_80+ GPU, each decoder step runs two persistent CUDA
kernels for the whole batch (`src/tts/magpietts/magpietts_batch_fused.cu`):
- the decoder step;
- the local-transformer frame, including in-kernel top-k sampling.

Each lane runs exactly the arithmetic of the single-request kernels. A request's
codes are therefore the same whether it decodes alone or with 15 others, and the
same as the serial runtime's (`test_magpietts_batch_fused`,
`test_magpietts_batch_longform`).

Otherwise (F16 models, CPU, other GPUs), steps run fixed-shape ggml graphs per
batch-size bucket (`batch_decoder.*`, `batch_lt.*`). The decoder's cached
attention uses `ggml_fused_attn_cached` over the slot arena, which also has a
CPU implementation. Sampling then happens on the host with each request's own
random generator. Results match the serial runtime to within floating-point
noise, but not bit for bit.

## Architecture

All files are in `src/tts/magpietts/batch/` unless noted.

| file | role |
|---|---|
| `batch_scheduler.*` | the scheduler thread: admission (FCFS), prefill budget, batched decode, eviction, cancellation, finishing threads |
| `slot_pool.*` | slot ownership, planar lane layout (`[cond_0..cond_{B-1} \| uncond_0..uncond_{B-1}]`), bucket choice |
| `batch_decoder.*` | slot arenas (self-attention ring caches per lane, cross-attention rows per slot) and the ggml step graph |
| `batch_lt.*` | the ggml batched local transformer with per-row host sampling |
| `batch_backend.*` | prefill into shared scratch caches and the handoff to a slot; batched steps on the fused kernels or the graphs |
| `batch_request.h` + `magpietts.cpp` | the request adapter: `MagpieRequestRunner` in engine mode, so the per-request logic is the serial runtime's |
| `batch_engine.*` | the public `submit` / `cancel` / `synthesize` API |

A request moves through these states:

```
QUEUED -> PREFILL -> DECODING --(EOS | step limit | cancel)--> draining audio -> done
                       ^   |
                       +---+  longform chunk boundary: re-prefill on the same slot
```

Each scheduler iteration does four things:
1. Admits waiting requests into free slots.
2. Runs prefill for up to `max_prefills_per_iter` requests: the text encoder, the
   baked-context decoder prefill and the first frame. While other requests are
   decoding, this waits until `max_decode_steps_before_prefill` decode steps have
   run since the last prefill.
3. Decodes one stacked frame for every decoding request in one batched step.
4. Evicts the requests that ended.

NanoCodec runs per request on its own worker thread. Codec decodes are
serialized because the requests share the codec backend.

## Performance notes (RTX A4500, v2607 Q8_0)

- **Kernel step:** 2.15 / 3.49 / 5.31 / 8.91 ms for 1 / 4 / 8 / 16 requests with
  CFG. That is 2.5x / 3.2x / 3.9x the rows per millisecond of a single request
  (no CFG: 4.5x at 8, 5.9x at 16).
- **End to end** (`bench tts`, MiniMax English, 10 utterances):

  | `--max-batch-size` | concurrency 1 / 4 / 8 | mean TTFA at 8 |
  |---|---|---|
  | 1 | RTFx 34.3 / 34.3 / 34.4 | 861 ms |
  | 8 | RTFx 34.5 / 59.2 / 67.8 | 94 ms |

- **What limits throughput today:**
  1. Serialized NanoCodec decodes across requests.
  2. Prefill, about 13 ms per request and serialized.
  3. In the kernels, F32 KV-cache reads in self-attention and the
     order-preserving per-lane-pair reductions.
