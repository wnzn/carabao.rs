use serde_json::{Map, Value, json};

pub const SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Choice,
    Noul,
    Score,
}

pub struct Decision {
    kind: Kind,
    options: Vec<(String, String)>,
    legend: Map<String, Value>,
    pub prompt: String,
}

fn valid(value: &Value) -> bool {
    match value {
        Value::String(s) => !s.trim().is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        _ => false,
    }
}

pub fn validate_state(state: &Value) -> Result<(), String> {
    if valid(state) {
        Ok(())
    } else {
        Err("state must be a nonempty string, object, or array".into())
    }
}

fn describe(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

impl Decision {
    pub fn prepare(state: &Value, question: &Value) -> Result<Self, String> {
        let q = question.as_object().ok_or("question must be an object")?;
        let kind = match q
            .get("type")
            .and_then(Value::as_str)
            .ok_or("type must be choice, noul, or score")?
        {
            "choice" => Kind::Choice,
            "noul" => Kind::Noul,
            "score" => Kind::Score,
            _ => return Err("type must be choice, noul, or score".into()),
        };
        let instruction = q.get("instructions").ok_or("instructions required")?;
        if !valid(instruction) {
            return Err("instructions must be a nonempty string, object, or array".into());
        }
        let mut legend = Map::new();
        let mut options = Vec::new();
        match kind {
            Kind::Choice => {
                let entries = q
                    .get("criteria")
                    .and_then(Value::as_object)
                    .ok_or("choice criteria must be an object")?;
                if !(2..=16).contains(&entries.len()) {
                    return Err("choice requires 2-16 options".into());
                }
                for (id, value) in entries {
                    if id.trim().is_empty() || (!value.is_null() && !valid(value)) {
                        return Err(format!("invalid choice option {id:?}"));
                    }
                    options.push((
                        id.clone(),
                        if value.is_null() {
                            id.clone()
                        } else {
                            describe(value)
                        },
                    ));
                }
            }
            Kind::Noul => {
                options = vec![
                    ("true".into(), "Yes.".into()),
                    ("false".into(), "No.".into()),
                ];
                if let Some(criteria) = q.get("criteria").filter(|v| !v.is_null()) {
                    let rubrics = criteria
                        .as_object()
                        .ok_or("noul criteria must be an object")?;
                    for (id, description) in &mut options {
                        if let Some(rubric) = rubrics.get(id) {
                            if !valid(rubric) {
                                return Err(format!("invalid noul criteria {id:?}"));
                            }
                            description.push(' ');
                            description.push_str(&describe(rubric));
                        }
                    }
                }
            }
            Kind::Score => {
                let levels = q
                    .get("criteria")
                    .and_then(Value::as_array)
                    .ok_or("score criteria must be an array of 2-10 levels")?;
                if !(2..=10).contains(&levels.len()) {
                    return Err("score criteria must be an array of 2-10 levels".into());
                }
                for (i, value) in levels.iter().enumerate() {
                    if !valid(value) {
                        return Err(format!("invalid score level {i}"));
                    }
                    let id = i.to_string();
                    options.push((id.clone(), describe(value)));
                    legend.insert(id, value.clone());
                }
            }
        }
        // Serialize once. Spaces outside strings match SemIf's Python json.dumps separators.
        // Use a struct so field order stays stable independent of serde_json map features.
        #[derive(serde::Serialize)]
        struct Prompt<'a> {
            evidence: &'a Value,
            criterion: String,
            options: Vec<PromptOption<'a>>,
        }
        #[derive(serde::Serialize)]
        struct PromptOption<'a> {
            letter: char,
            description: &'a str,
        }
        let raw = serde_json::to_string(&Prompt {
            evidence: state,
            criterion: describe(instruction),
            options: options
                .iter()
                .enumerate()
                .map(|(i, (_, description))| PromptOption {
                    letter: (b'A' + i as u8) as char,
                    description,
                })
                .collect(),
        })
        .map_err(|e| e.to_string())?;
        Ok(Self {
            kind,
            options,
            legend,
            prompt: spaced_json(&raw),
        })
    }

    pub fn letters(&self) -> impl Iterator<Item = char> + '_ {
        (0..self.options.len()).map(|i| (b'A' + i as u8) as char)
    }

    pub fn answer(&self, probabilities: &[f64]) -> Result<Value, String> {
        if probabilities.len() != self.options.len()
            || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
        {
            return Err("backend returned invalid option probabilities".into());
        }
        if self.kind == Kind::Noul {
            return Ok(json!({"type":"noul", "noul":probabilities[0]}));
        }
        let mut probs = Map::new();
        let mut best = 0;
        let mut entropy = 0.0;
        let mut expected = 0.0;
        for (i, ((id, _), &p)) in self.options.iter().zip(probabilities).enumerate() {
            probs.insert(id.clone(), json!(p));
            if p > probabilities[best] {
                best = i;
            }
            if p > 0.0 {
                entropy -= p * p.ln();
            }
            expected += i as f64 * p;
        }
        let confidence = (1.0 - entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0);
        if self.kind == Kind::Choice {
            Ok(
                json!({"type":"choice", "choice": self.options[best].0, "probabilities": probs, "confidence": confidence}),
            )
        } else {
            Ok(
                json!({"type":"score", "score": expected, "legend": self.legend, "probabilities": probs, "confidence": confidence}),
            )
        }
    }
}

pub fn spaced_json(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + input.len() / 8);
    let (mut quoted, mut escaped) = (false, false);
    for c in input.chars() {
        if quoted {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                quoted = false;
            }
        } else if c == '"' {
            quoted = true;
        }
        out.push(c);
        if !quoted && (c == ':' || c == ',') {
            out.push(' ');
        }
    }
    out
}

pub fn softmax(logits: &[f64]) -> Result<Vec<f64>, String> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite() || *x <= -1e30) {
        return Err("invalid answer-token probability".into());
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|p| p / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_and_order() {
        let q: Value = serde_json::from_str(
            r#"{"type":"choice","instructions":"Choose","criteria":{"z":"Z","a":"A"}}"#,
        )
        .unwrap();
        let d = Decision::prepare(&json!("hello"), &q).unwrap();
        assert!(d.prompt.contains(r#""evidence": "hello", "criterion": "Choose", "options": [{"letter": "A", "description": "Z"}, {"letter": "B", "description": "A"}]"#));
        assert_eq!(d.answer(&[0.9, 0.1]).unwrap()["choice"], "z");
        assert!(d.answer(&[1.0]).is_err());
    }
    #[test]
    fn escaped_json() {
        assert_eq!(
            spaced_json(r#"{"a":"x,y:z\\\"w"}"#),
            r#"{"a": "x,y:z\\\"w"}"#
        );
    }
    #[test]
    fn normalization_and_validation() {
        let p = softmax(&[-1000., -1001.]).unwrap();
        assert!((p[0] - 0.7310585786).abs() < 1e-8);
        assert!(softmax(&[f64::NAN]).is_err());
        assert!(validate_state(&json!([])).is_err());
    }
}
