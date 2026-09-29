use crate::backend::{Engine, Media, Score, verify_boundary};
use crate::decision::{Decision, SYSTEM, softmax};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaModel, params::LlamaModelParams},
};
use std::{
    num::NonZeroU32,
    path::Path,
    sync::mpsc::{self, Sender},
};

struct Job {
    prompt: String,
    letters: Vec<char>,
    reply: Sender<Result<Score, String>>,
}
pub struct Native {
    sender: Sender<Job>,
}

impl Native {
    pub fn load(path: &Path, ctx_size: u32, gpu_layers: u32, threads: i32) -> Result<Self, String> {
        if !path.is_file() {
            return Err(format!("GGUF model not found: {}", path.display()));
        }
        let (sender, receiver) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let path = path.to_owned();
        std::thread::spawn(move || {
            let backend = match LlamaBackend::init() {
                Ok(x) => x,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };
            let model = match LlamaModel::load_from_file(
                &backend,
                &path,
                &LlamaModelParams::default().with_n_gpu_layers(gpu_layers),
            ) {
                Ok(x) => x,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };
            let template = match model.chat_template(None) {
                Ok(x) => x,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };
            let mut params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(ctx_size))
                .with_n_batch(ctx_size.min(512));
            if threads > 0 {
                params = params.with_n_threads(threads).with_n_threads_batch(threads);
            }
            let mut context = match model.new_context(&backend, params) {
                Ok(x) => x,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(()));
            for job in receiver {
                let answer = (|| -> Result<Score, String> {
                    let messages = [
                        LlamaChatMessage::new("system".into(), SYSTEM.into())
                            .map_err(|e| e.to_string())?,
                        LlamaChatMessage::new("user".into(), job.prompt)
                            .map_err(|e| e.to_string())?,
                    ];
                    let prompt = model
                        .apply_chat_template(&template, &messages, true)
                        .map_err(|e| e.to_string())?;
                    let (ids, slots) = verify_boundary(&prompt, job.letters.into_iter(), |s| {
                        model
                            .str_to_token(s, AddBos::Never)
                            .map(|v| v.into_iter().map(|t| t.0).collect())
                            .map_err(|e| e.to_string())
                    })?;
                    if ids.len() >= context.n_ctx() as usize {
                        return Err("prompt exceeds local context window (--ctx-size)".into());
                    }
                    context.clear_kv_cache();
                    let batch_size = context.n_batch() as usize;
                    let mut batch = LlamaBatch::new(batch_size, 1);
                    for (start, chunk) in ids.chunks(batch_size).enumerate() {
                        batch.clear();
                        let offset = start * batch_size;
                        for (i, &token) in chunk.iter().enumerate() {
                            batch
                                .add(
                                    llama_cpp_2::token::LlamaToken(token),
                                    (offset + i) as i32,
                                    &[0],
                                    offset + i + 1 == ids.len(),
                                )
                                .map_err(|e| e.to_string())?;
                        }
                        context.decode(&mut batch).map_err(|e| e.to_string())?;
                    }
                    let logits = context.get_logits_ith(((ids.len() - 1) % batch_size) as i32);
                    let selected: Vec<f64> = slots
                        .iter()
                        .map(|&id| {
                            logits
                                .get(id as usize)
                                .map(|v| *v as f64)
                                .ok_or("invalid answer token ID".into())
                        })
                        .collect::<Result<_, String>>()?;
                    Ok(Score {
                        probabilities: softmax(&selected)?,
                        input_tokens: ids.len(),
                        output_tokens: 0,
                    })
                })();
                let _ = job.reply.send(answer);
            }
        });
        ready_rx
            .recv()
            .map_err(|_| "local inference worker exited".to_string())??;
        Ok(Self { sender })
    }
}

impl Engine for Native {
    fn score(&self, decision: &Decision, media: Option<&Media>) -> Result<Score, String> {
        if media.is_some() {
            return Err("native multimodal inference is not available; use --llama-path with --mmproj or --remote".into());
        }
        let (reply, recv) = mpsc::channel();
        self.sender
            .send(Job {
                prompt: decision.prompt.clone(),
                letters: decision.letters().collect(),
                reply,
            })
            .map_err(|_| "local inference worker exited")?;
        recv.recv()
            .map_err(|_| "local inference worker exited".to_string())?
    }
}
