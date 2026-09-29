#!/usr/bin/env python3
"""Mock-upstream adapter overhead comparison (no model inference). stdlib only."""
import argparse
import http.server
import json
import os
import socket
import statistics
import subprocess
import threading
import time
import urllib.request
import urllib.error


class Mock(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_json({"media_marker": "<media>", "modalities": {}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            result = {"prompt": "<assistant>" + body["messages"][1]["content"] + "<answer>"}
        elif self.path == "/tokenize":
            result = {"tokens": list(body["content"].encode("utf-8"))}
        elif self.path == "/completion":
            result = {"completion_probabilities": [{"top_logprobs": [
                {"id": 65, "logprob": -0.1}, {"id": 66, "logprob": -1.1}
            ]}], "tokens_evaluated": len(body["prompt"])}
        else:
            self.send_error(404)
            return
        self.send_json(result)

    def send_json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        # Avoid Python's delayed-ACK artifact with tiny back-to-back HTTP bodies.
        # Both adapters get the same short-lived connection behavior.
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(data)


def server(handler):
    instance = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=instance.serve_forever, daemon=True).start()
    return instance


def wait_ready(url, process):
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(f"adapter exited: {process.returncode}: {process.stderr.read().decode()}")
        try:
            urllib.request.urlopen(url + "/health", timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("adapter did not start")


def measure(command, port, payload, count, endpoint):
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        wait_ready(url, process)
        times = []
        for i in range(count + 10):
            request = urllib.request.Request(url + endpoint, payload, {"Content-Type": "application/json"})
            start = time.perf_counter_ns()
            try:
                with urllib.request.urlopen(request, timeout=5) as reply:
                    result = json.load(reply)
                    assert result["answers"]["q"]["choice"] == "first"
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"adapter returned HTTP {error.code}: {error.read().decode()}") from error
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if i >= 10:
                times.append(elapsed)
        rss = 0
        with open(f"/proc/{process.pid}/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
        return statistics.median(times), sorted(times)[int(0.95 * (len(times) - 1))], rss / 1024
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", default="target/release/carabao")
    parser.add_argument("--go", default="../semif-go/semif-go")
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be positive")
    mock = server(Mock)
    try:
        payload = json.dumps({"model": "jev-latest", "state": "A support ticket", "questions": {
            "q": {"type": "choice", "instructions": "Which queue?", "criteria": {
                "first": "First team", "second": "Second team"}}
        }}).encode()
        commands = [
            ("carabao", [args.rust, "--listen", "127.0.0.1:{port}", "-rl", f"http://127.0.0.1:{mock.server_port}", "-lv", "warn"], "/decisions"),
            ("semif-go", [args.go, "-listen", "127.0.0.1:{port}", "-llama-url", f"http://127.0.0.1:{mock.server_port}"], "/v1/systemone"),
        ]
        print("Adapter       binary MiB   RSS MiB   p50 ms   p95 ms")
        for name, cmd, endpoint in commands:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            result = measure([arg.replace("{port}", str(port)) for arg in cmd], port, payload, args.requests, endpoint)
            print(f"{name:<13} {os.path.getsize(cmd[0]) / 1048576:>10.2f} {result[2]:>9.1f} {result[0]:>8.2f} {result[1]:>8.2f}")
    finally:
        mock.shutdown()


if __name__ == "__main__":
    main()
