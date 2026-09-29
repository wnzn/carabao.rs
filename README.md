<div align="center">
  <h1>🐃 carabao.rs</h1>
  <p><strong>One token. Typed decisions. Your GGUF.</strong></p>
  <p>A small Rust System One adapter for <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> and <a href="https://github.com/vllm-project/vllm">vLLM</a>.<br>
  Ask runtime-defined questions; get conditional option probabilities instead of generated prose.</p>
  <p><code>native GGUF</code> · <code>llama-server</code> · <code>vLLM</code> · <code>choice / noul / score</code></p>
</div>

---

## Start

Requires Rust 1.98+ (edition 2024; tested on current stable 1.98.1), a GGUF model, and CMake, a C++ compiler and libclang for the **native** build. The default binary bundles CPU llama.cpp bindings; no separate llama-server is needed. Models are not bundled. The directory/project is `carabao.rs`; Cargo uses `carabao-rs` (package names cannot contain dots) and the executable is `carabao`.

```sh
cargo build --release
./target/release/carabao --model /path/to/model.gguf --ctx-size 4096
```

Alternatively, connect to an already running llama-server or vLLM, or let carabao launch your installed llama-server:

```sh
# Existing llama-server: -rl is short for --remote-llama; no local model needed
cargo build --release --no-default-features
./target/release/carabao -rl http://127.0.0.1:8080 --model-name my-model

# Existing llama.cpp installation: carabao manages llama-server
./target/release/carabao --llama-path /path/to/llama.cpp/build/bin \
  --model /path/to/model.gguf --gpu-layers 99 --ctx-size 8192

# Existing vLLM server: use its served model ID; -rv is short for --remote-vllm
./target/release/carabao -rv http://127.0.0.1:8000 \
  --vllm-model Qwen/Qwen3-0.6B --model-name qwen3
```

For vLLM, start a separately installed server with `vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 --generation-config vllm --logprobs-mode raw_logprobs`. The vLLM URL is the **server root**, not `/v1`. `--vllm-model` must match the served model ID; `--model-name` remains the label carabao returns to clients. Set `VLLM_API_KEY` for vLLM's upstream bearer key, separately from `CARABAO_API_KEY`. vLLM support is **text-only** for now; its multimodal chat content format cannot use llama-server's media-marker path. The backend needs a vLLM release with `logprob_token_ids` in chat completions and `return_token_ids`; incompatible servers return an error rather than a guessed answer. Some model chat templates cannot make an open assistant answer match the generation prefix; carabao rejects those instead of assigning the wrong answer-token IDs. See [vLLM's chat API](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/#chat-api) and [logprob modes](https://docs.vllm.ai/en/stable/configuration/engine_args/#--logprobs-mode). Its [GGUF support](https://docs.vllm.ai/en/stable/features/quantization/gguf/) is experimental and requires a separate plugin. The older `--vllm-remote` / `-v` options remain supported as aliases.

`--remote-llama` / `-rl` and `--remote-vllm` / `-rv` are mutually exclusive. The older `--remote` / `-R` spelling for llama-server remains supported as an alias. `--llama-path` accepts either the `llama-server` executable or its directory. Add `--mmproj /path/to/mmproj.gguf` for supported multimodal checkpoints. It binds the managed server to `127.0.0.1:8080` by default (`--llama-listen` to change). **Use `--remote-llama` rather than `--llama-path` if a server is already running on that port.** The managed child is stopped on SIGINT/SIGTERM; SIGKILL cannot be intercepted, so check for orphaned server processes after a forced kill.

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
curl -sS http://127.0.0.1:8090/decisions \
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

`POST /decisions` is the primary endpoint; `/v1/decisions` is also accepted. `/systemone` and `/v1/systemone` remain supported as legacy aliases with the same request and response shape. The response includes `model`, question-keyed `answers`, and `usage` (`input_tokens`, `output_tokens`). Native mode reports **zero** output tokens because it reads prompt logits without generating a token; llama-server reports its completion probes and vLLM reports one generated probe token. Choice returns the selected ID, conditional probabilities and normalized-entropy confidence. Noul returns the probability of **yes**. Score returns a probability-weighted **zero-based** level, probabilities, legend and confidence. The `jev-latest` input alias is accepted for compatibility; the response uses `--model-name` (`carabao-local` by default). This is not Jev or a calibrated Jev-compatible model.

### How it scores

```
state + criterion + options  →  model chat template  →  verify A…P are distinct
                                                      single tokens at the boundary
                          →  one prompt evaluation  →  read option logits
                          →  softmax over options   →  typed answer
```

The prompt follows [SemIf's direct-options approach](https://github.com/TheoLeeCJ/SemIf-OpenJev). Native mode reads logits directly from the bundled llama.cpp binding: no next-token generation or top-N serialization. The llama-server backend uses `/apply-template`, `/tokenize` and `/completion`, requesting **pre-sampling** logprobs and doubling `n_probs` until all answers are found or `--max-top-probs` is reached. The vLLM backend uses `/tokenize` to check the assistant answer boundary and `/v1/chat/completions` with `logprob_token_ids` to obtain precisely the requested labels; it verifies that vLLM actually scored the same, untruncated prompt. Both server modes generate one token per scoring probe. It reuses one native model/context across requests; native inference is serialized through one worker, while HTTP/remote requests can run concurrently. If your GGUF chat template is not supported by the native llama.cpp template API, use a recent llama-server via `-rl` or `--llama-path`. Server templates receive `enable_thinking=false`; the native template API has no equivalent template-kwargs setting, so reasoning-specific templates may need a server path.

**Compatibility:** `choice` takes 2–16 options (in JSON object order); `score` takes 2–10 levels; `noul` is yes/no with optional `true`/`false` rubrics. Multiple questions are scored independently. Option probabilities sum to 1 **only among the listed options**; confidence is an uncalibrated entropy statistic, not a reliability guarantee. Question IDs and option IDs may be arbitrary nonempty strings. Invalid input returns 422; inference/backend errors return 502.

## Configuration & security

| Setting | Meaning |
| --- | --- |
| `--listen 127.0.0.1:8090` | Adapter bind address; loopback by default. |
| `CARABAO_API_KEY` | If set, require `Authorization: Bearer <key>` on decision requests. `SEMIF_API_KEY` is a fallback. |
| `LLAMA_API_KEY` | Bearer token sent to a protected remote llama-server (never use it as the client key). |
| `VLLM_API_KEY` | Bearer token sent to vLLM (`--remote-vllm` / `-rv`); independent of the client key. |
| `--log-verbosity LEVEL`, `-lv LEVEL` | stderr log level: `off`, `error`, `warn`, `info` (default), `debug`, `trace`. Logs status; request latency at `debug`, per-question token counts/timing at `trace`. Never logs prompts, media or keys. |
| `--cors-origin https://app.example` | Allow exactly this browser origin; repeat for several. No wildcard; CORS off by default. |
| `--model-name LABEL` | Response model label, not the GGUF loading path. |
| `--ctx-size N`, `--gpu-layers N`, `--threads N` | Native context, GPU offload and CPU threads; context/GPU settings also passed to managed llama-server. |
| `--llama-startup-timeout 300` | Seconds to wait for a managed llama-server to load a model. |
| `--initial-top-probs 256`, `--max-top-probs 262144` | llama-server retry range; unused by native or vLLM. |
| `--cache-prompt` | Opt in to llama-server prompt reuse; off by default. |
| `--max-inflight 32` | Bound simultaneous HTTP requests; excess receive HTTP 503. Native inference remains serialized. |

`GET /health` does not need a key. `OPTIONS` preflight works on all decision routes only for configured origins and permits `POST`, `Authorization`, and `Content-Type`. CORS is **not authentication**: keep the API key enabled and put TLS/authentication at your reverse proxy if exposing the service beyond loopback. vLLM's own `--api-key` [does not protect every endpoint](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/); keep its server on loopback or behind a properly configured reverse proxy. Request bodies are limited to 64 MiB; raw prompts, media and secrets are not logged.

## Multimodal (llama-server modes)

`--remote-llama` and `--llama-path` can forward text, images, audio and video to a compatible llama-server via `/props` media markers and `multimodal_data`; **`--remote-vllm` is text-only**:

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
| carabao (remote-only) | 2.88 MiB | 4.6 MiB | 1.69 ms | 2.15 ms |
| semif-go | 9.09 MiB | 16.2 MiB | 2.95 ms | 3.58 ms |

These are one run, not a hardware-independent speed claim. A stripped CPU-native release binary on this machine was **7.2 MiB** (bundled llama.cpp); the remote-only build is smaller. A local Wendi 2B Q4_K_M GGUF smoke test returned a Noul result in 0.66 s on CPU with four threads (one run, not a benchmark). The ROCm feature compiled with this machine's HIP toolchain; GPU execution has not been measured. To compare real inference, use the same GGUF, llama.cpp backend, hardware, context size and prompt; the native/server paths may differ in template handling. The Go baseline is the existing `semif-go` binary from this workspace; no Go source files were modified.

## Attribution

Independent MIT-licensed adapter; see [LICENSE](LICENSE). Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) through [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs), and inspired by [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev)'s option-logit method and [System One / Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)'s typed decision interface. Not affiliated with those projects. Your model's license applies separately.
