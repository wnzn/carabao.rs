mod backend;
mod decision;
mod media;
#[cfg(feature = "native")]
mod native;
mod strict_json;

use backend::{Engine, Remote};
use clap::{ArgGroup, Parser};
use decision::{Decision, validate_state};
use serde_json::{Value, json};
use std::{
    io::Read,
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

#[derive(Parser)]
#[command(version, about = "System One decisions on GGUF via llama.cpp", group(
    ArgGroup::new("backend").args(["remote", "llama_path"]).multiple(false)
))]
struct Cli {
    /// Adapter listen address (loopback by default)
    #[arg(long, default_value = "127.0.0.1:8090")]
    listen: String,
    /// Remote llama-server base URL
    #[arg(short = 'R', long)]
    remote: Option<String>,
    /// Existing llama-server executable (or directory containing it); starts a managed server
    #[arg(long)]
    llama_path: Option<PathBuf>,
    /// GGUF file (required for native or --llama-path modes)
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,
    /// Response model label
    #[arg(long, default_value = "carabao-local")]
    model_name: String,
    /// Native context size or managed llama-server context size
    #[arg(long, default_value_t = 4096)]
    ctx_size: u32,
    /// GPU layers to offload (0 = CPU)
    #[arg(long, default_value_t = 0)]
    gpu_layers: u32,
    /// Native inference threads (0 = llama.cpp default)
    #[arg(long, default_value_t = 0)]
    threads: i32,
    /// Managed llama-server address
    #[arg(long, default_value = "127.0.0.1:8080")]
    llama_listen: String,
    /// Managed llama-server startup timeout, in seconds
    #[arg(long, default_value_t = 300)]
    llama_startup_timeout: u64,
    /// Multimodal projector for managed llama-server
    #[arg(long)]
    mmproj: Option<PathBuf>,
    /// Initial remote top-N probability count
    #[arg(long, default_value_t = 256)]
    initial_top_probs: usize,
    /// Maximum remote top-N probability count
    #[arg(long, default_value_t = 262144)]
    max_top_probs: usize,
    /// Enable llama-server prompt cache (remote and managed modes)
    #[arg(long)]
    cache_prompt: bool,
    /// Allowed browser Origin; repeatable. CORS is disabled unless specified.
    #[arg(long = "cors-origin")]
    cors_origins: Vec<String>,
    /// Maximum simultaneous HTTP requests (excess requests receive 503)
    #[arg(long, default_value_t = 32)]
    max_inflight: usize,
}

struct App {
    engine: Arc<dyn Engine>,
    model: String,
    api_key: String,
    origins: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("carabao: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    if cli.initial_top_probs == 0 || cli.max_top_probs < cli.initial_top_probs {
        return Err("invalid top-probs range".into());
    }
    if cli.ctx_size < 8 || cli.threads < 0 {
        return Err("ctx-size must be >= 8 and threads >= 0".into());
    }
    if !(1..=1024).contains(&cli.max_inflight) {
        return Err("max-inflight must be 1..=1024".into());
    }
    if cli.llama_startup_timeout == 0 {
        return Err("llama-startup-timeout must be positive".into());
    }
    if cli.mmproj.is_some() && cli.llama_path.is_none() {
        return Err("--mmproj requires --llama-path".into());
    }
    for origin in &cli.cors_origins {
        let host = origin
            .strip_prefix("https://")
            .or_else(|| origin.strip_prefix("http://"));
        if host.is_none_or(|h| {
            h.is_empty()
                || !h
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b".:-[]".contains(&b))
        }) {
            return Err(format!(
                "invalid CORS origin {origin:?}; use a precise scheme://host[:port]"
            ));
        }
    }
    let mut child = Managed(None);
    let engine: Arc<dyn Engine> = if let Some(url) = &cli.remote {
        Arc::new(Remote::new(
            url,
            cli.initial_top_probs,
            cli.max_top_probs,
            cli.cache_prompt,
        )?)
    } else if let Some(llama_path) = &cli.llama_path {
        let model = cli
            .model
            .as_ref()
            .ok_or("--model is required with --llama-path")?;
        if !model.is_file() {
            return Err(format!("GGUF model not found: {}", model.display()));
        }
        let (host, port) = cli
            .llama_listen
            .rsplit_once(':')
            .ok_or("--llama-listen must be host:port")?;
        if host != "127.0.0.1" && host != "localhost" {
            return Err("managed llama-server must bind to loopback".into());
        }
        let listener = TcpListener::bind(&cli.llama_listen).map_err(|e| {
            format!(
                "managed llama-server address {} unavailable: {e}",
                cli.llama_listen
            )
        })?;
        drop(listener);
        let executable = if llama_path.is_dir() {
            llama_path.join("llama-server")
        } else {
            llama_path.clone()
        };
        let mut command = Command::new(&executable);
        command.args([
            "--model",
            model.to_str().ok_or("model path must be UTF-8")?,
            "--host",
            host,
            "--port",
            port,
            "--ctx-size",
            &cli.ctx_size.to_string(),
            "--gpu-layers",
            &cli.gpu_layers.to_string(),
            "--parallel",
            "1",
        ]);
        if let Some(mmproj) = &cli.mmproj {
            command.arg("--mmproj").arg(mmproj);
        }
        command.stdout(Stdio::null());
        child.0 = Some(
            command
                .spawn()
                .map_err(|e| format!("could not start {}: {e}", executable.display()))?,
        );
        let url = format!("http://{host}:{port}");
        let remote = Remote::new(
            &url,
            cli.initial_top_probs,
            cli.max_top_probs,
            cli.cache_prompt,
        )?;
        let mut ready = false;
        let deadline = Instant::now() + Duration::from_secs(cli.llama_startup_timeout);
        while Instant::now() < deadline {
            if let Some(status) = child
                .0
                .as_mut()
                .unwrap()
                .try_wait()
                .map_err(|e| e.to_string())?
            {
                return Err(format!("llama-server exited during startup: {status}"));
            }
            if remote.ready() {
                ready = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        if !ready {
            return Err(format!(
                "llama-server did not become ready in {} seconds",
                cli.llama_startup_timeout
            ));
        }
        Arc::new(remote)
    } else {
        let model = cli
            .model
            .as_ref()
            .ok_or("--model is required for local inference")?;
        #[cfg(feature = "native")]
        {
            Arc::new(native::Native::load(
                model,
                cli.ctx_size,
                cli.gpu_layers,
                cli.threads,
            )?)
        }
        #[cfg(not(feature = "native"))]
        {
            let _ = model;
            return Err("native support not compiled; use --remote or --llama-path".into());
        }
    };
    let app = Arc::new(App {
        engine,
        model: cli.model_name,
        api_key: std::env::var("CARABAO_API_KEY")
            .or_else(|_| std::env::var("SEMIF_API_KEY"))
            .unwrap_or_default(),
        origins: cli.cors_origins,
    });
    let server = Server::http(&cli.listen).map_err(|e| format!("listen {}: {e}", cli.listen))?;
    eprintln!("carabao listening on {}", cli.listen);
    let stopping = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&stopping);
    ctrlc::set_handler(move || signal_flag.store(true, Ordering::Relaxed))
        .map_err(|e| format!("could not install shutdown handler: {e}"))?;
    // The child is stopped on shutdown, including SIGINT/SIGTERM.
    let _managed = child;
    let (permit_tx, permit_rx) = mpsc::sync_channel(cli.max_inflight);
    for _ in 0..cli.max_inflight {
        permit_tx.send(()).map_err(|e| e.to_string())?;
    }
    while !stopping.load(Ordering::Relaxed) {
        let Some(request) = server
            .recv_timeout(Duration::from_millis(250))
            .map_err(|e| format!("HTTP accept: {e}"))?
        else {
            continue;
        };
        if permit_rx.try_recv().is_err() {
            respond(
                request,
                503,
                json!({"error":{"message":"too many concurrent requests"}}),
                None,
            );
            continue;
        }
        let permit = Permit(permit_tx.clone());
        let app = Arc::clone(&app);
        std::thread::spawn(move || {
            let _permit = permit;
            handle(request, &app);
        });
    }
    Ok(())
}

struct Managed(Option<Child>);
struct Permit(mpsc::SyncSender<()>);
impl Drop for Permit {
    fn drop(&mut self) {
        let _ = self.0.try_send(());
    }
}
impl Drop for Managed {
    fn drop(&mut self) {
        if let Some(child) = &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn header(name: &[u8], value: &str) -> Header {
    Header::from_bytes(name, value.as_bytes()).expect("valid HTTP header")
}

fn respond(request: Request, status: u16, value: Value, origin: Option<&str>) {
    let mut response = Response::from_string(value.to_string())
        .with_status_code(StatusCode(status))
        .with_header(header(b"Content-Type", "application/json"));
    if let Some(origin) = origin {
        response.add_header(header(b"Access-Control-Allow-Origin", origin));
        response.add_header(header(b"Vary", "Origin"));
    }
    let _ = request.respond(response);
}

fn handle(mut request: Request, app: &App) {
    let origin = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Origin"))
        .map(|h| h.value.as_str().to_owned());
    let allowed = origin
        .as_deref()
        .filter(|o| app.origins.iter().any(|allowed| allowed == *o));
    if origin.is_some() && allowed.is_none() {
        respond(
            request,
            403,
            json!({"error":{"message":"origin not allowed"}}),
            None,
        );
        return;
    }
    if let (true, Some(origin)) = (
        request.method() == &Method::Options && request.url() == "/v1/systemone",
        allowed,
    ) {
        let mut response = Response::empty(StatusCode(204));
        response.add_header(header(b"Access-Control-Allow-Origin", origin));
        response.add_header(header(b"Access-Control-Allow-Methods", "POST, OPTIONS"));
        response.add_header(header(
            b"Access-Control-Allow-Headers",
            "Authorization, Content-Type",
        ));
        response.add_header(header(b"Vary", "Origin"));
        let _ = request.respond(response);
        return;
    }
    if request.url() == "/health" {
        let status = if request.method() == &Method::Get {
            200
        } else {
            405
        };
        respond(request, status, json!({"status":"ok"}), allowed);
        return;
    }
    if request.url() != "/v1/systemone" {
        respond(
            request,
            404,
            json!({"error":{"message":"not found"}}),
            allowed,
        );
        return;
    }
    if request.method() != &Method::Post {
        respond(
            request,
            405,
            json!({"error":{"message":"POST required"}}),
            allowed,
        );
        return;
    }
    if !app.api_key.is_empty()
        && !request.headers().iter().any(|h| {
            h.field.equiv("Authorization") && h.value.as_str() == format!("Bearer {}", app.api_key)
        })
    {
        respond(
            request,
            401,
            json!({"error":{"message":"invalid API key"}}),
            allowed,
        );
        return;
    }
    let mut body = Vec::new();
    let result = request
        .as_reader()
        .take((64 << 20) + 1)
        .read_to_end(&mut body)
        .map_err(|e| (422, e.to_string()))
        .and_then(|_| {
            if body.len() > 64 << 20 {
                Err((413, "request exceeds 64 MiB".into()))
            } else {
                Ok(())
            }
        })
        .and_then(|_| {
            serde_json::from_slice::<strict_json::Strict>(&body)
                .map_err(|e| (422, format!("invalid JSON request: {e}")))
        })
        .and_then(|value| evaluate(&value.0, app));
    match result {
        Ok(reply) => respond(request, 200, reply, allowed),
        Err((status, message)) => respond(
            request,
            status,
            json!({"error":{"message":message}}),
            allowed,
        ),
    }
}

fn evaluate(request: &Value, app: &App) -> Result<Value, (u16, String)> {
    let fail = |s: &str| (422, s.to_owned());
    let state = request.get("state").ok_or_else(|| fail("state required"))?;
    validate_state(state).map_err(|s| (422, s))?;
    let model = request["model"]
        .as_str()
        .ok_or_else(|| fail("model required"))?;
    if model != "jev-latest" && model != app.model {
        return Err(fail("unknown model"));
    }
    let (state, media) = if state["type"] == "multimodal" {
        let props = app.engine.props().map_err(|s| (502, s))?;
        let (state, media) = media::parse(state, props).map_err(|s| (422, s))?;
        (state, Some(media))
    } else {
        (state.clone(), None)
    };
    let questions = request["questions"]
        .as_object()
        .filter(|q| !q.is_empty())
        .ok_or_else(|| fail("questions must be a nonempty object"))?;
    let mut prepared = Vec::with_capacity(questions.len());
    for (id, question) in questions {
        if id.trim().is_empty() {
            return Err(fail("question IDs cannot be empty"));
        }
        let decision = Decision::prepare(&state, question)
            .map_err(|s| (422, format!("questions[{id:?}]: {s}")))?;
        if let Some(m) = &media
            && decision.prompt.matches(&m.marker).count() != m.data.len()
        {
            return Err(fail("prompt contains unexpected media markers"));
        }
        prepared.push((id, decision));
    }
    prepared.sort_by(|a, b| a.0.cmp(b.0));
    let (mut input, mut output) = (0, 0);
    let mut answers = serde_json::Map::new();
    for (id, decision) in prepared {
        let score = app
            .engine
            .score(&decision, media.as_ref())
            .map_err(|s| (502, format!("questions[{id:?}]: {s}")))?;
        input += score.input_tokens;
        output += score.output_tokens;
        answers.insert(
            id.clone(),
            decision
                .answer(&score.probabilities)
                .map_err(|e| (502, e))?,
        );
    }
    Ok(
        json!({"model":app.model,"answers":answers,"usage":{"input_tokens":input,"output_tokens":output}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::{Media, Score};
    use std::{io::Write, net::TcpStream};
    struct Mock;
    impl Engine for Mock {
        fn score(&self, _: &Decision, _: Option<&Media>) -> Result<Score, String> {
            Ok(Score {
                probabilities: vec![0.8, 0.2],
                input_tokens: 12,
                output_tokens: 1,
            })
        }
    }
    fn app() -> App {
        App {
            engine: Arc::new(Mock),
            model: "test".into(),
            api_key: "secret".into(),
            origins: vec!["http://localhost:3000".into()],
        }
    }
    #[test]
    fn mixed_questions_and_errors() {
        let payload = json!({"model":"jev-latest", "state":"Refund requested", "questions": {
            "flag":{"type":"noul","instructions":"Is there a refund request?"},
            "team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments","technical":"Bugs"}},
            "severity":{"type":"score","instructions":"Urgency?","criteria":["Low","High"]}
        }});
        let result = evaluate(&payload, &app()).unwrap();
        assert_eq!(result["usage"]["input_tokens"], 36);
        assert_eq!(result["answers"]["team"]["choice"], "billing");
        assert_eq!(result["answers"]["flag"]["noul"], 0.8);
        assert_eq!(result["answers"]["severity"]["score"], 0.2);
        let mut invalid = payload;
        invalid["questions"]["team"]["criteria"] = json!({"only":"one"});
        assert_eq!(evaluate(&invalid, &app()).unwrap_err().0, 422);
    }

    #[test]
    fn http_auth_and_cors() {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_string();
        let worker = std::thread::spawn(move || {
            let app = app();
            for request in server.incoming_requests().take(4) {
                handle(request, &app);
            }
        });
        let send = |method: &str, origin: &str, auth: &str| {
            let mut stream = TcpStream::connect(&address).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let body = r#"{"model":"jev-latest","state":"hello","questions":{"q":{"type":"noul","instructions":"Is this text?"}}}"#;
            write!(stream, "{method} /v1/systemone HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n{origin}{auth}\r\n{body}", body.len()).unwrap();
            let mut reply = String::new();
            stream.read_to_string(&mut reply).unwrap();
            reply
        };
        let code = |reply: &str| {
            reply
                .split_whitespace()
                .nth(1)
                .unwrap()
                .parse::<u16>()
                .unwrap()
        };
        assert_eq!(code(&send("POST", "", "")), 401);
        assert_eq!(
            code(&send(
                "POST",
                "Origin: http://evil.test\r\n",
                "Authorization: Bearer secret\r\n"
            )),
            403
        );
        let preflight = send("OPTIONS", "Origin: http://localhost:3000\r\n", "");
        assert_eq!(code(&preflight), 204);
        assert!(
            preflight
                .to_ascii_lowercase()
                .contains("access-control-allow-origin: http://localhost:3000")
        );
        assert_eq!(
            code(&send("POST", "", "Authorization: Bearer secret\r\n")),
            200
        );
        worker.join().unwrap();
    }
}
