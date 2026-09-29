use crate::{
    backend::{Engine, Media, Score, check_slot},
    decision::{Decision, SYSTEM, softmax},
};
use serde_json::{Value, json};
use std::time::Duration;

pub struct Vllm {
    base: String,
    model: String,
    key: String,
    agent: ureq::Agent,
}

impl Vllm {
    pub fn new(base: &str, model: &str) -> Result<Self, String> {
        let normalized = base.trim_end_matches('/');
        let host = normalized
            .strip_prefix("http://")
            .or_else(|| normalized.strip_prefix("https://"));
        if host.is_none_or(|h| {
            h.is_empty()
                || !h
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b".:-[]".contains(&b))
        }) {
            return Err("vLLM remote must be an HTTP(S) base URL without /v1".into());
        }
        if model.trim().is_empty() {
            return Err("--vllm-model must name the served vLLM model".into());
        }
        Ok(Self {
            base: normalized.to_owned(),
            model: model.to_owned(),
            key: std::env::var("VLLM_API_KEY").unwrap_or_default(),
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs(300)))
                .build()
                .into(),
        })
    }

    fn post(&self, route: &str, data: Value) -> Result<Value, String> {
        let url = format!("{}{route}", self.base);
        let mut retried = false;
        let mut response = loop {
            let mut request = self.agent.post(&url);
            if !self.key.is_empty() {
                request = request.header("Authorization", &format!("Bearer {}", self.key));
            }
            match request.send_json(&data) {
                Ok(response) => break response,
                Err(ureq::Error::Io(_)) if !retried => retried = true,
                Err(e) => return Err(format!("vLLM {route}: {e}")),
            }
        };
        response
            .body_mut()
            .read_json()
            .map_err(|e| format!("vLLM {route} response: {e}"))
    }

    fn tokenize(&self, messages: &[Value], generation: bool) -> Result<Vec<i32>, String> {
        let result = self.post(
            "/tokenize",
            json!({
                "model": self.model, "messages": messages,
                "add_generation_prompt": generation,
                "continue_final_message": !generation,
                "add_special_tokens": false,
                "chat_template_kwargs": {"enable_thinking": false}
            }),
        )?;
        token_ids(&result["tokens"]).ok_or("vLLM /tokenize returned invalid token IDs".into())
    }
}

fn token_ids(value: &Value) -> Option<Vec<i32>> {
    value
        .as_array()?
        .iter()
        .map(|v| {
            v.as_i64()
                .and_then(|id| i32::try_from(id).ok())
                .filter(|id| *id >= 0)
        })
        .collect()
}

fn scored(result: &Value, prompt: &[i32], slots: &[i32]) -> Result<Score, String> {
    if token_ids(&result["prompt_token_ids"]).as_deref() != Some(prompt) {
        return Err("vLLM generated from a different or truncated prompt".into());
    }
    let usage = &result["usage"];
    let input = usage["prompt_tokens"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("vLLM returned invalid prompt token usage")?;
    let output = usage["completion_tokens"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("vLLM returned invalid completion token usage")?;
    if input != prompt.len() || output != 1 {
        return Err("vLLM returned inconsistent token usage".into());
    }
    let entries = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        .as_array()
        .ok_or("vLLM returned no selected-token logprobs")?;
    let mut values = vec![None; slots.len()];
    for entry in entries {
        let Some(token) = entry["token"]
            .as_str()
            .and_then(|t| t.strip_prefix("token_id:"))
            .and_then(|s| s.parse::<i32>().ok())
        else {
            continue;
        };
        if let Some(index) = slots.iter().position(|id| *id == token) {
            values[index] = entry["logprob"].as_f64();
        }
    }
    let logprobs = values.into_iter().collect::<Option<Vec<_>>>().ok_or(
        "vLLM omitted a requested answer token; update vLLM or check logprob_token_ids support",
    )?;
    Ok(Score {
        probabilities: softmax(&logprobs)?,
        input_tokens: input,
        output_tokens: output,
    })
}

impl Engine for Vllm {
    fn score(&self, decision: &Decision, media: Option<&Media>) -> Result<Score, String> {
        if media.is_some() {
            return Err("vLLM multimodal input is not yet supported".into());
        }
        let messages = vec![
            json!({"role":"system", "content": SYSTEM}),
            json!({"role":"user", "content": decision.prompt}),
        ];
        let prompt = self.tokenize(&messages, true)?;
        if prompt.is_empty() {
            return Err("vLLM chat template produced no prompt tokens".into());
        }
        let mut slots = Vec::new();
        for letter in decision.letters() {
            let mut with_answer = messages.clone();
            with_answer.push(json!({"role":"assistant", "content": letter.to_string()}));
            let joined = self.tokenize(&with_answer, false)?;
            slots.push(check_slot(&prompt, &joined, letter, &slots)?);
        }
        let result = self.post(
            "/v1/chat/completions",
            json!({
                "model": self.model, "messages": messages,
                "add_generation_prompt": true, "add_special_tokens": false,
                "chat_template_kwargs": {"enable_thinking": false},
                "max_tokens": 1, "temperature": 1, "top_p": 1, "top_k": 0, "min_p": 0,
                "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0,
                "logprobs": true, "logprob_token_ids": slots,
                "return_tokens_as_token_ids": true, "return_token_ids": true
            }),
        )?;
        scored(&result, &prompt, &slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_scoring_response() {
        let reply = json!({"prompt_token_ids":[1,2],"usage":{"prompt_tokens":2,"completion_tokens":1},
        "choices":[{"logprobs":{"content":[{"top_logprobs":[
            {"token":"token_id:65","logprob":-1.0},
            {"token":"token_id:66","logprob":-2.0}
        ]}]}}]});
        let score = scored(&reply, &[1, 2], &[65, 66]).unwrap();
        assert!((score.probabilities[0] - 0.7310585786).abs() < 1e-8);
        assert_eq!(score.output_tokens, 1);
        let mut wrong = reply;
        wrong["prompt_token_ids"] = json!([1]);
        assert!(scored(&wrong, &[1, 2], &[65, 66]).is_err());
        wrong["prompt_token_ids"] = json!([1, 2]);
        wrong["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(scored(&wrong, &[1, 2], &[65, 66]).is_err());
    }

    #[test]
    fn rejects_non_root_server_url() {
        assert!(Vllm::new("http://127.0.0.1:8000/v1", "model").is_err());
        assert!(Vllm::new("http://user@127.0.0.1:8000", "model").is_err());
        assert!(Vllm::new("http://127.0.0.1:8000/", "model").is_ok());
    }

    #[test]
    fn vllm_wire_contract_and_boundary() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let url = format!("http://{}", server.server_addr());
        let worker = std::thread::spawn(move || {
            let mut base: Option<Vec<i32>> = None;
            for mut req in server.incoming_requests().take(4) {
                let mut body = String::new();
                std::io::Read::read_to_string(req.as_reader(), &mut body).unwrap();
                let data: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(data["model"], "test-model");
                assert_eq!(data["chat_template_kwargs"]["enable_thinking"], false);
                let response = if req.url() == "/tokenize" {
                    let mut tokens = vec![1, 2];
                    if data["continue_final_message"] == true {
                        let letter = data["messages"][2]["content"].as_str().unwrap();
                        tokens.push(letter.as_bytes()[0] as i32);
                    } else {
                        base = Some(tokens.clone());
                    }
                    json!({"tokens":tokens})
                } else {
                    assert_eq!(req.url(), "/v1/chat/completions");
                    assert_eq!(data["logprob_token_ids"], json!([65, 66]));
                    assert_eq!(data["max_tokens"], 1);
                    assert_eq!(data["return_tokens_as_token_ids"], true);
                    json!({"prompt_token_ids":base,"usage":{"prompt_tokens":2,"completion_tokens":1},
                    "choices":[{"logprobs":{"content":[{"top_logprobs":[
                        {"token":"token_id:65","logprob":-1.0}, {"token":"token_id:66","logprob":-2.0}
                    ]}]}}]})
                };
                req.respond(tiny_http::Response::from_string(response.to_string()))
                    .unwrap();
            }
        });
        let backend = Vllm::new(&url, "test-model").unwrap();
        let decision = Decision::prepare(
            &json!("hello"),
            &json!({"type":"noul","instructions":"Yes?"}),
        )
        .unwrap();
        assert!(
            (backend.score(&decision, None).unwrap().probabilities[0] - 0.7310585786).abs() < 1e-8
        );
        worker.join().unwrap();
    }
}
