<div align="center">
  <h1>🐃 carabao.rs</h1>
  <p><strong>One token. Typed decisions. Your GGUF.</strong></p>
  <p>A small Rust System One adapter for <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a>.<br>
  Ask runtime-defined questions; get conditional option probabilities instead of generated prose.</p>
  <p><code>native GGUF</code> · <code>existing llama-server</code> · <code>managed llama-server</code> · <code>choice / noul / score</code></p>
</div>

---

## Start

Requires Rust 1.98+ (edition 2024; tested on current stable 1.98.1), a GGUF model, and CMake, a C++ compiler and libclang for the **native** build. The default binary bundles CPU llama.cpp bindings; no separate llama-server is needed. Models are not bundled. The directory/project is `carabao.rs`; Cargo uses `carabao-rs` (package names cannot contain dots) and the executable is `carabao`.

```sh
cargo build --release
./target/release/carabao --model /path/to/model.gguf --ctx-size 4096
```

Alternatively, connect to an already running llama-server or let carabao launch your installed one:

```sh
# Existing server: no model file needed on the adapter host
cargo build --release --no-default-features
./target/release/carabao -R http://127.0.0.1:8080 --model-name my-model

# Existing llama.cpp installation: carabao manages llama-server
./target/release/carabao --llama-path /path/to/llama.cpp/build/bin \
  --model /path/to/model.gguf --gpu-layers 99 --ctx-size 8192
```

`--llama-path` accepts either the `llama-server` executable or its directory. Add `--mmproj /path/to/mmproj.gguf` for supported multimodal checkpoints. It binds the managed server to `127.0.0.1:8080` by default (`--llama-listen` to change). **Use `--remote` rather than `--llama-path` if a server is already running on that port.** The managed child is stopped on SIGINT/SIGTERM; SIGKILL cannot be intercepted, so check for orphaned server processes after a forced kill.

Native GPU builds are opt-in:

```sh
cargo build --release --features rocm    # AMD: ROCm toolchain / HIP SDK required
cargo build --release --features cuda    # NVIDIA: CUDA toolkit required
cargo build --release --features vulkan  # Vulkan SDK required
cargo build --release --features metal   # macOS
./target/release/carabao -m /path/to/model.gguf --gpu-layers 99
```

If `bindgen` cannot locate libclang, set `LIBCLANG_PATH` to its library directory. On installations with nonstandard Clang resources you may also need `BINDGEN_EXTRA_CLANG_ARGS='-isystem /path/to/clang/<version>/include'`. On the development machine, `LIBCLANG_PATH=/opt/rocm-7.2.4/lib/llvm/lib` supplied **build-time libclang**; it did not enable ROCm inference by itself. Native builds compile bundled llama.cpp and will be larger than the remote-only build. For your existing ROCm-enabled llama.cpp, `--llama-path` avoids rebuilding llama.cpp entirely.

## Ask a question

```sh
curl -sS http://127.0.0.1:8090/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "I was charged twice; please refund the duplicate payment.",
    "questions": {
      "queue": {"type": "choice", "instructions": "Who should handle this?",
        "criteria": {"billing": "Payments and refunds", "technical": "Bugs"}},
      "refund": {"type": "noul", "instructions": "Is a refund requested?"},
      "urgency": {"type": "score", "instructions": "How urgent is it?",
        "criteria": ["Routine", "Soon", "Immediate"]}
    }
  }'
```

The response includes `model`, question-keyed `answers`, and `usage` (`input_tokens`, `output_tokens`). Native mode reports **zero** output tokens because it reads prompt logits without generating a token; server mode reports its completion probes. Choice returns the selected ID, conditional probabilities and normalized-entropy confidence. Noul returns the probability of **yes**. Score returns a probability-weighted **zero-based** level, probabilities, legend and confidence. The `jev-latest` input alias is accepted for compatibility; the response uses `--model-name` (`carabao-local` by default). This is not Jev or a calibrated Jev-compatible model.

### How it scores

```
state + criterion + options  →  model chat template  →  verify A…P are distinct
                                                      single tokens at the boundary
                          →  one prompt evaluation  →  read option logits
                          →  softmax over options   →  typed answer
```

The prompt follows [SemIf's direct-options approach](https://github.com/TheoLeeCJ/SemIf-OpenJev). Native mode reads logits directly from the bundled llama.cpp binding: no next-token generation or top-N serialization. Remote/server mode uses `/apply-template`, `/tokenize` and `/completion`, requesting **pre-sampling** logprobs and doubling `n_probs` until all answers are found or `--max-top-probs` is reached. It checks prompt-boundary tokenization (not just bare letters) and rejects truncated prompts. It reuses one native model/context across requests; native inference is serialized through one worker, while HTTP/remote requests can run concurrently. If your GGUF chat template is not supported by the native llama.cpp template API, use a recent llama-server via `-R` or `--llama-path`. Remote templates receive `enable_thinking=false`; the native template API has no equivalent template-kwargs setting, so reasoning-specific templates may need the server path.

**Compatibility:** `choice` takes 2–16 options (in JSON object order); `score` takes 2–10 levels; `noul` is yes/no with optional `true`/`false` rubrics. Multiple questions are scored independently. Option probabilities sum to 1 **only among the listed options**; confidence is an uncalibrated entropy statistic, not a reliability guarantee. Question IDs and option IDs may be arbitrary nonempty strings. Invalid input returns 422; inference/backend errors return 502.

## Configuration & security

| Setting | Meaning |
| --- | --- |
| `--listen 127.0.0.1:8090` | Adapter bind address; loopback by default. |
| `CARABAO_API_KEY` | If set, require `Authorization: Bearer <key>` on decision requests. `SEMIF_API_KEY` is a fallback. |
| `LLAMA_API_KEY` | Bearer token sent to a protected remote llama-server (never use it as the client key). |
| `--cors-origin https://app.example` | Allow exactly this browser origin; repeat for several. No wildcard; CORS off by default. |
| `--model-name LABEL` | Response model label, not the GGUF loading path. |
| `--ctx-size N`, `--gpu-layers N`, `--threads N` | Native context, GPU offload and CPU threads; context/GPU settings also passed to managed llama-server. |
| `--llama-startup-timeout 300` | Seconds to wait for a managed llama-server to load a model. |
| `--initial-top-probs 256`, `--max-top-probs 262144` | Remote retry range; unused by native logits. |
| `--cache-prompt` | Opt in to llama-server prompt reuse; off by default. |
| `--max-inflight 32` | Bound simultaneous HTTP requests; excess receive HTTP 503. Native inference remains serialized. |

`GET /health` does not need a key. `OPTIONS /v1/systemone` preflight works only for configured origins and permits `POST`, `Authorization`, and `Content-Type`. CORS is **not authentication**: keep the API key enabled and put TLS/authentication at your reverse proxy if exposing the service beyond loopback. Request bodies are limited to 64 MiB; raw prompts, media and secrets are not logged.

## Multimodal (server modes)

`--remote` and `--llama-path` can forward text, images, audio and video to a compatible llama-server via `/props` media markers and `multimodal_data`:

```json
{"model":"jev-latest","state":{"type":"multimodal","content":[
  {"type":"text","text":"What is pictured?"},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,<BASE64>"}}
]},"questions":{"subject":{"type":"choice","instructions":"Pick the subject",
  "criteria":{"cat":"A cat","dog":"A dog"}}}}
```

Audio/video parts use `input_audio` / `input_video` with either `{"data":"<BASE64>"}` or an inline MIME data `url`. Model capabilities must be reported in `/props`. Limits: 8 parts, 32 MiB decoded per part, 48 MiB total. Remote/file media URLs are not fetched. **Native mode is text-only**; for a local multimodal model, use `--llama-path --mmproj` instead. Video may additionally need `ffmpeg`/`ffprobe` on the llama-server host. See the [llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) and [multimodal guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

## Verify & compare

```sh
cargo test                             # includes native bindings
cargo test --no-default-features       # remote-only build
cargo build --release --no-default-features
python3 benches/compare.py --requests 200 --go ../semif-go/semif-go
```

The comparison uses the **same mock upstream** with short-lived connections for both adapters, warms up each server, then reports median/p95 serial request latency, RSS and stripped binary size. It measures *adapter overhead only*, not model throughput or quality. A run on this machine (Linux x86-64, AMD Ryzen 7 8845HS, 200 requests, remote-only Rust release vs the existing Go binary):

| Adapter | Binary | RSS | p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| carabao (remote-only) | 2.76 MiB | 4.6 MiB | 1.72 ms | 2.29 ms |
| semif-go | 9.09 MiB | 16.2 MiB | 3.07 ms | 3.82 ms |

These are one run, not a hardware-independent speed claim. A stripped CPU-native release binary on this machine was **7.2 MiB** (bundled llama.cpp); the remote-only build is smaller. A local Wendi 2B Q4_K_M GGUF smoke test returned a Noul result in 0.66 s on CPU with four threads (one run, not a benchmark). The ROCm feature compiled with this machine's HIP toolchain; GPU execution has not been measured. To compare real inference, use the same GGUF, llama.cpp backend, hardware, context size and prompt; the native/server paths may differ in template handling. The Go baseline is the existing `semif-go` binary from this workspace; no Go source files were modified.

## Attribution

Independent MIT-licensed adapter; see [LICENSE](LICENSE). Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) through [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs), and inspired by [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev)'s option-logit method and [System One / Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)'s typed decision interface. Not affiliated with those projects. Your model's license applies separately.
