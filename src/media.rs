use crate::backend::{Media, Props};
use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, STANDARD_NO_PAD},
};
use serde_json::Value;

pub fn parse(state: &Value, props: Props) -> Result<(Value, Media), String> {
    let parts = state["content"]
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or("multimodal state requires a nonempty content array")?;
    let mut evidence = String::new();
    let mut data = Vec::new();
    let mut total = 0usize;
    for (index, part) in parts.iter().enumerate() {
        let kind = part["type"]
            .as_str()
            .ok_or_else(|| format!("content[{index}] must be a typed object"))?;
        if kind == "text" {
            let text = part["text"]
                .as_str()
                .filter(|s| !s.trim().is_empty())
                .ok_or("empty multimodal text")?;
            if text.contains(&props.marker) {
                return Err("text contains server media marker".into());
            }
            evidence.push_str(text);
            evidence.push('\n');
            continue;
        }
        if data.len() == 8 {
            return Err("at most 8 media parts are allowed".into());
        }
        let (source, modality, label) = match kind {
            "image_url" => (part["image_url"]["url"].as_str(), "vision", "Image"),
            "input_audio" => (single_source(&part["input_audio"])?, "audio", "Audio"),
            "input_video" => (single_source(&part["input_video"])?, "video", "Video"),
            _ => return Err(format!("unsupported content type {kind:?}")),
        };
        if props.modalities[modality] != true {
            return Err(format!("model does not support {modality} input"));
        }
        let source = source.ok_or("missing inline media data")?;
        let encoded = if let Some(url) = source.strip_prefix("data:") {
            let (metadata, payload) = url.split_once(',').ok_or("invalid media data URL")?;
            let prefix = if modality == "vision" {
                "image/"
            } else if modality == "audio" {
                "audio/"
            } else {
                "video/"
            };
            if !metadata.to_ascii_lowercase().starts_with(prefix) {
                return Err("media MIME type does not match input".into());
            }
            if !metadata.to_ascii_lowercase().ends_with(";base64") {
                return Err("media URL must be base64".into());
            }
            payload
        } else if modality == "vision" || source.contains("://") {
            return Err("use inline base64 data; remote and file URLs are not supported".into());
        } else {
            source
        };
        if encoded.len() > (32 << 20) * 4 / 3 + 8 {
            return Err("media part exceeds 32 MiB".into());
        }
        let bytes = STANDARD
            .decode(encoded)
            .or_else(|_| STANDARD_NO_PAD.decode(encoded))
            .map_err(|_| "invalid base64 media")?;
        if bytes.is_empty() || bytes.len() > (32 << 20) {
            return Err("media part must be 1 byte to 32 MiB".into());
        }
        total += bytes.len();
        if total > (48 << 20) {
            return Err("combined media exceeds 48 MiB".into());
        }
        data.push(STANDARD.encode(bytes));
        evidence.push_str(&format!("{label} {}: {}\n", data.len(), props.marker));
    }
    if data.is_empty() {
        return Err("multimodal state requires at least one media part".into());
    }
    Ok((
        Value::String(evidence),
        Media {
            data,
            marker: props.marker,
        },
    ))
}

fn single_source(value: &Value) -> Result<Option<&str>, String> {
    let data = value["data"].as_str().filter(|s| !s.is_empty());
    let url = value["url"].as_str().filter(|s| !s.is_empty());
    if data.is_some() && url.is_some() {
        return Err("specify only one media source".into());
    }
    Ok(data.or(url))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn inline_image_and_rejection() {
        let props = || Props {
            marker: "<media>".into(),
            modalities: json!({"vision":true}),
        };
        let state = json!({"content":[{"type":"text","text":"Describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}}]});
        let (text, media) = parse(&state, props()).unwrap();
        assert!(text.as_str().unwrap().contains("Image 1: <media>"));
        assert_eq!(media.data, ["YWJj"]);
        let audio = json!({"content":[{"type":"input_audio","input_audio":{"data":"YWI"}}]});
        let (_, audio_media) = parse(
            &audio,
            Props {
                marker: "<media>".into(),
                modalities: json!({"audio":true}),
            },
        )
        .unwrap();
        assert_eq!(audio_media.data, ["YWI="]);
        assert!(parse(&json!({"content":[{"type":"image_url","image_url":{"url":"https://example.org/a.png"}}]}), props()).is_err());
    }
}
