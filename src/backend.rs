use crate::decision::{Decision, SYSTEM, softmax};
use serde_json::{Value, json};
use std::time::Duration;

pub struct Score {
    pub probabilities: Vec<f64>,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

pub trait Engine: Send + Sync {
    fn score(&self, decision: &Decision, media: Option<&Media>) -> Result<Score, String>;
    fn props(&self) -> Result<Props, String> {
        Err("multimodal input requires llama-server (--remote or --llama-path)".into())
    }
}

#[derive(Clone)]
pub struct Props {
    pub marker: String,
    pub modalities: Value,
}
pub struct Media {
    pub data: Vec<String>,
    pub marker: String,
}

pub fn verify_boundary<F>(
    prompt: &str,
    letters: impl Iterator<Item = char>,
    mut tokenize: F,
) -> Result<(Vec<i32>, Vec<i32>), String>
where
    F: FnMut(&str) -> Result<Vec<i32>, String>,
{
    let tokens = tokenize(prompt)?;
    if tokens.is_empty() {
        return Err("chat template produced no prompt tokens".into());
    }
    let mut slots = Vec::new();
    let mut combined = String::with_capacity(prompt.len() + 1);
    combined.push_str(prompt);
    for letter in letters {
        combined.push(letter);
        let joined = tokenize(&combined)?;
        combined.pop();
        if joined.len() != tokens.len() + 1 || joined[..tokens.len()] != tokens[..] {
            return Err(format!(
                "answer slot {letter} is not one token at the prompt boundary"
            ));
        }
        let id = joined[tokens.len()];
        if slots.contains(&id) {
            return Err("answer slot tokens collide".into());
        }
        slots.push(id);
    }
    Ok((tokens, slots))
}

pub struct Remote {
    base: String,
    key: String,
    agent: ureq::Agent,
    pub initial: usize,
    pub maximum: usize,
    pub cache_prompt: bool,
}

impl Remote {
    pub fn new(
        base: &str,
        initial: usize,
        maximum: usize,
        cache_prompt: bool,
    ) -> Result<Self, String> {
        let host = base
            .strip_prefix("http://")
            .or_else(|| base.strip_prefix("https://"));
        if host.is_none_or(|h| h.is_empty() || h.starts_with('/') || h.contains(['?', '#', '@'])) {
            return Err("remote must be an HTTP(S) base URL".into());
        }
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(300)))
            .build();
        Ok(Self {
            base: base.trim_end_matches('/').into(),
            key: std::env::var("LLAMA_API_KEY").unwrap_or_default(),
            agent: config.into(),
            initial,
            maximum,
            cache_prompt,
        })
    }

    pub fn ready(&self) -> bool {
        self.agent
            .get(&format!("{}/health", self.base))
            .call()
            .is_ok()
    }

    fn post(&self, route: &str, data: Value) -> Result<Value, String> {
        let url = format!("{}{}", self.base, route);
        let large_response = data["n_probs"].as_u64().is_some_and(|top| top > 8192);
        let mut retried = false;
        let mut response = loop {
            let mut request = self.agent.post(&url);
            if !self.key.is_empty() {
                request = request.header("Authorization", &format!("Bearer {}", self.key));
            }
            match request.send_json(&data) {
                Ok(response) => break response,
                Err(ureq::Error::Io(_)) if !retried => {
                    retried = true;
                }
                Err(error) => return Err(format!("llama-server {route}: {error}")),
            }
        };
        if large_response {
            response
                .body_mut()
                .with_config()
                .limit(128 << 20)
                .read_json()
                .map_err(|e| format!("llama-server {route} response: {e}"))
        } else {
            response
                .body_mut()
                .read_json()
                .map_err(|e| format!("llama-server {route} response: {e}"))
        }
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>, String> {
        let result = self.post(
            "/tokenize",
            json!({"content":text,"add_special":false,"parse_special":true}),
        )?;
        let tokens = result["tokens"]
            .as_array()
            .ok_or("llama-server returned no tokens")?;
        tokens
            .iter()
            .map(|v| {
                v.as_i64()
                    .and_then(|x| i32::try_from(x).ok())
                    .ok_or("invalid token ID".into())
            })
            .collect()
    }
}

impl Engine for Remote {
    fn props(&self) -> Result<Props, String> {
        let url = format!("{}/props", self.base);
        let mut request = self.agent.get(&url);
        if !self.key.is_empty() {
            request = request.header("Authorization", &format!("Bearer {}", self.key));
        }
        let mut response = request
            .call()
            .map_err(|e| format!("llama-server /props: {e}"))?;
        let value: Value = response.body_mut().read_json().map_err(|e| e.to_string())?;
        let marker = value["media_marker"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("llama-server /props has no media_marker")?;
        Ok(Props {
            marker: marker.into(),
            modalities: value["modalities"].clone(),
        })
    }

    fn score(&self, decision: &Decision, media: Option<&Media>) -> Result<Score, String> {
        let rendered = self.post("/apply-template", json!({
            "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":decision.prompt}],
            "add_generation_prompt":true, "chat_template_kwargs":{"enable_thinking":false}
        }))?;
        let prompt = rendered["prompt"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("empty chat template prompt")?;
        if let Some(m) = media
            && prompt.matches(&m.marker).count() != m.data.len()
        {
            return Err("chat template changed media markers".into());
        }
        let (tokens, slots) = verify_boundary(prompt, decision.letters(), |s| self.tokenize(s))?;
        let scoring_prompt = if let Some(m) = media {
            json!({"prompt_string": prompt, "multimodal_data": m.data})
        } else {
            json!(tokens)
        };
        let mut top = self.initial;
        let mut probes = 0;
        loop {
            let result = self.post(
                "/completion",
                json!({
                    "prompt":scoring_prompt, "n_predict":1, "n_probs":top,
                    "post_sampling_probs":false,"cache_prompt":self.cache_prompt
                }),
            )?;
            if result["truncated"] == true {
                return Err("llama-server truncated the prompt".into());
            }
            let entries = result["completion_probabilities"][0]["top_logprobs"]
                .as_array()
                .filter(|a| !a.is_empty())
                .ok_or("llama-server returned no next-token probabilities")?;
            probes += result["completion_probabilities"]
                .as_array()
                .map_or(1, Vec::len);
            let mut values = vec![None; slots.len()];
            for entry in entries {
                if let Some(i) = slots
                    .iter()
                    .position(|id| entry["id"].as_i64() == Some(*id as i64))
                {
                    values[i] = entry["logprob"].as_f64();
                }
            }
            if let Some(logits) = values.into_iter().collect::<Option<Vec<_>>>() {
                let usage = result["tokens_evaluated"]
                    .as_u64()
                    .unwrap_or(tokens.len() as u64) as usize;
                if media.is_some() && usage == 0 {
                    return Err("llama-server returned no multimodal token usage".into());
                }
                return Ok(Score {
                    probabilities: softmax(&logits)?,
                    input_tokens: usage,
                    output_tokens: probes,
                });
            }
            if top >= self.maximum {
                return Err(format!(
                    "an answer token is outside the top {top} next tokens; raise --max-top-probs"
                ));
            }
            top = top.saturating_mul(2).min(self.maximum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn boundary_rejects_merges() {
        assert!(
            verify_boundary("hi", "AB".chars(), |s| Ok(s
                .bytes()
                .map(i32::from)
                .collect()))
            .is_ok()
        );
        assert!(
            verify_boundary("hi", "A".chars(), |s| Ok(if s.ends_with('A') {
                vec![9]
            } else {
                vec![1]
            }))
            .is_err()
        );
    }
    #[test]
    fn remote_retries_missing_option_and_keeps_pre_sampling() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let base = format!("http://{}", server.server_addr());
        let worker = std::thread::spawn(move || {
            let mut tops = Vec::new();
            for mut request in server.incoming_requests().take(6) {
                let mut body = String::new();
                std::io::Read::read_to_string(request.as_reader(), &mut body).unwrap();
                let input: Value = serde_json::from_str(&body).unwrap();
                let result = match request.url() {
                    "/apply-template" => {
                        json!({"prompt":format!("<assistant>{}<answer>", input["messages"][1]["content"].as_str().unwrap())})
                    }
                    "/tokenize" => {
                        json!({"tokens":input["content"].as_str().unwrap().bytes().collect::<Vec<_>>()})
                    }
                    "/completion" => {
                        assert_eq!(input["n_predict"], 1);
                        assert_eq!(input["post_sampling_probs"], false);
                        assert_eq!(input["cache_prompt"], true);
                        tops.push(input["n_probs"].as_u64().unwrap());
                        if tops.len() == 1 {
                            json!({"completion_probabilities":[{"top_logprobs":[{"id":65,"logprob":-1.0}]}]})
                        } else {
                            json!({"completion_probabilities":[{"top_logprobs":[{"id":65,"logprob":-1.0},{"id":66,"logprob":-2.0}]}]})
                        }
                    }
                    path => panic!("unexpected path {path}"),
                };
                request
                    .respond(tiny_http::Response::from_string(result.to_string()))
                    .unwrap();
            }
            tops
        });
        let engine = Remote::new(&base, 2, 4, true).unwrap();
        let d = Decision::prepare(
            &json!("hello"),
            &json!({"type":"noul","instructions":"Is it true?"}),
        )
        .unwrap();
        let score = engine.score(&d, None).unwrap();
        assert!((score.probabilities[0] - 0.7310585786).abs() < 1e-8);
        assert_eq!(score.output_tokens, 2);
        assert_eq!(worker.join().unwrap(), [2, 4]);
    }
}
