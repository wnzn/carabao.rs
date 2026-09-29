<div align="center">
  <h1>🐃 carabao.rs</h1>
  <p><strong>Typed decisions from your models.</strong></p>
  <p>A small Rust adapter for <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> and <a href="https://github.com/vllm-project/vllm">vLLM</a>.<br>
  Ask runtime-defined questions; get option probabilities instead of generated prose.</p>
</div>

## llama.cpp

Run a GGUF model directly (Rust 1.98+, CMake, a C++ compiler and libclang required):

```sh
cargo build --release
./target/release/carabao -m /path/to/model.gguf
```

Or connect to an existing llama-server without building native bindings:

```sh
cargo build --release --no-default-features
./target/release/carabao -rl http://127.0.0.1:8080
```

`-rl` is short for `--remote-llama`. To launch an installed llama-server instead, see [`--llama-path` in the detailed guide](docs/guide.md#installation--backends).

## vLLM

Start vLLM with a supported model:

```sh
vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 \
  --generation-config vllm --logprobs-mode raw_logprobs
```

Then, in another terminal:

```sh
cargo build --release --no-default-features
./target/release/carabao -rv http://127.0.0.1:8000 \
  --vllm-model Qwen/Qwen3-0.6B
```

`-rv` is short for `--remote-vllm`. This backend is text-only and needs vLLM support for `logprob_token_ids` and `return_token_ids`.

## Make a decision

Both backends serve `POST /decisions` on `127.0.0.1:8090` by default:

```sh
curl -sS http://127.0.0.1:8090/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"carabao-local","state":"The parcel arrived damaged.",
       "questions":{"damaged":{"type":"noul","instructions":"Was the parcel damaged?"}}}'
```

Choice, noul and score questions, API keys, logging (`-lv`), CORS, multimodal input, GPU builds and benchmarks are covered in the **[detailed guide](docs/guide.md)**.
