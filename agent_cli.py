import argparse
import os
import shlex
import subprocess
import sys
import textwrap
import json
import urllib.request
import urllib.error
from html.parser import HTMLParser
from urllib.parse import urlparse
from pathlib import Path
import math
import re

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


DEFAULT_WEATHER_URL = "http://YOUR_WEATHER_HOST:PORT/"
DEFAULT_SITE_URL = (
    "http://YOUR_SITE_HOST:PORT/"
)
DEFAULT_BOX_CLASS = "box"
DEFAULT_LLM_BASE_URL = "http://YOUR_LLM_HOST:PORT/v1"
DEFAULT_LLM_MODEL = "llama-3.3-70b" # model
WORKSPACE_ROOT = Path(__file__).resolve().parent


class BoxParser(HTMLParser):
    def __init__(self, class_name):
        super().__init__()
        self.class_name = class_name
        self.in_box = False
        self.current = []
        self.boxes = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "class" and value and self.class_name in value.split():
                self.in_box = True
                self.current = []
                break

    def handle_endtag(self, tag):
        if self.in_box:
            text = " ".join(" ".join(self.current).split())
            if text:
                self.boxes.append(text)
            self.in_box = False
            self.current = []

    def handle_data(self, data):
        if self.in_box:
            self.current.append(data.strip())


class WeatherParser(HTMLParser):
    def __init__(self, box_class="box", label_class="label", value_class="value"):
        super().__init__()
        self.box_class = box_class
        self.label_class = label_class
        self.value_class = value_class
        self.in_box = False
        self.box_depth = 0
        self.in_label = False
        self.in_value = False
        self.current_label = []
        self.current_value = []
        self.results = []

    def handle_starttag(self, tag, attrs):
        cls = None
        for key, value in attrs:
            if key == "class":
                cls = value
                break
        if cls:
            classes = cls.split()
            if self.box_class in classes:
                self.in_box = True
                self.box_depth = 1
                self.current_label = []
                self.current_value = []
            elif self.in_box and tag == "div":
                self.box_depth += 1
            if self.in_box and self.label_class in classes:
                self.in_label = True
            if self.in_box and self.value_class in classes:
                self.in_value = True

    def handle_endtag(self, tag):
        if self.in_label:
            self.in_label = False
        if self.in_value:
            self.in_value = False
        if self.in_box and tag == "div":
            self.box_depth -= 1
            if self.box_depth <= 0:
                label = " ".join(" ".join(self.current_label).split())
                value = " ".join(" ".join(self.current_value).split())
                if label or value:
                    self.results.append({"label": label, "value": value})
                self.in_box = False
                self.box_depth = 0

    def handle_data(self, data):
        if self.in_label:
            self.current_label.append(data.strip())
        if self.in_value:
            self.current_value.append(data.strip())


def load_config(path):
    if not os.path.exists(path):
        return {}
    if tomllib is None:
        raise RuntimeError("Python 3.11+ required for TOML config parsing.")
    with open(path, "rb") as f:
        return tomllib.load(f)


def wsl_run(command, auto_approve):
    if not auto_approve:
        if not sys.stdin.isatty():
            return 2, "", "Approval required but no TTY. Use --auto-approve or set approval.mode=auto."
        try:
            answer = input(f"Approve command?\n  {command}\n[y/N]: ").strip().lower()
        except EOFError:
            return 2, "", "Approval required but input was closed. Use --auto-approve or set approval.mode=auto."
        if answer not in ("y", "yes"):
            return 1, "", "Command cancelled."

    proc = subprocess.run(
        ["wsl.exe", "-e", "bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def resolve_path(path_str):
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (WORKSPACE_ROOT / p).resolve()
    else:
        p = p.resolve()
    if not str(p).startswith(str(WORKSPACE_ROOT)):
        raise ValueError(f"Path is outside workspace: {p}")
    return p


def confirm_destructive(action, target_path):
    if not sys.stdin.isatty():
        return False, "Destructive action requires confirmation, but no TTY."
    answer = input(f"{action} '{target_path}'? [y/N]: ").strip().lower()
    return answer in ("y", "yes"), ""


def tool_list_dir(path):
    p = resolve_path(path)
    if not p.exists():
        return {"ok": False, "error": "path not found"}
    if not p.is_dir():
        return {"ok": False, "error": "path is not a directory"}
    items = []
    for child in sorted(p.iterdir()):
        items.append({"name": child.name, "is_dir": child.is_dir()})
    return {"ok": True, "items": items, "path": str(p)}


def tool_read_file(path, max_bytes=20000):
    p = resolve_path(path)
    if not p.exists():
        return {"ok": False, "error": "file not found"}
    if not p.is_file():
        return {"ok": False, "error": "path is not a file"}
    data = p.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True
    else:
        truncated = False
    text = data.decode("utf-8", errors="replace")
    return {"ok": True, "path": str(p), "content": text, "truncated": truncated}


def tool_write_file(path, content, overwrite=False):
    p = resolve_path(path)
    if p.exists() and not overwrite:
        return {"ok": False, "error": "file exists; overwrite not allowed"}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(p), "bytes": len(content.encode("utf-8"))}


def tool_append_file(path, content):
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True, "path": str(p), "bytes": len(content.encode("utf-8"))}


def tool_replace_text(path, find_text, replace_text):
    p = resolve_path(path)
    if not p.exists():
        return {"ok": False, "error": "file not found"}
    data = p.read_text(encoding="utf-8", errors="replace")
    if find_text not in data:
        return {"ok": False, "error": "find_text not found"}
    new_data = data.replace(find_text, replace_text)
    p.write_text(new_data, encoding="utf-8")
    return {"ok": True, "path": str(p)}


def tool_delete_file(path):
    p = resolve_path(path)
    if not p.exists():
        return {"ok": False, "error": "path not found"}
    if p.is_dir():
        return {"ok": False, "error": "refusing to delete directory"}
    p.unlink()
    return {"ok": True, "path": str(p)}


def tool_weather(url, class_name, auto_approve):
    rc, out, err = wsl_run(f"curl -fsSL --max-time 10 {shlex.quote(url)}", auto_approve)
    if rc != 0:
        return {"ok": False, "error": err.strip() or "fetch failed"}

    # Prefer structured label/value parsing, fallback to raw box text.
    wparser = WeatherParser(box_class=class_name)
    wparser.feed(out)
    if wparser.results:
        return {"ok": True, "items": wparser.results}

    parser = BoxParser(class_name)
    parser.feed(out)
    if not parser.boxes:
        return {"ok": False, "error": f"No elements found with class '{class_name}'."}
    return {"ok": True, "boxes": parser.boxes}


def tool_site_check(url, auto_approve):
    cmd = f"curl -s --max-time 10 -o /dev/null -w '%{{http_code}}' {shlex.quote(url)}"
    rc, out, err = wsl_run(cmd, auto_approve)
    if rc != 0:
        return {"ok": False, "error": err.strip() or "site check failed"}
    status = out.strip()
    return {"ok": status == "200", "status": status, "url": url}


def tool_ping(target, count, auto_approve):
    rc, out, err = wsl_run(f"ping -c {count} {shlex.quote(target)}", auto_approve)
    return {"ok": rc == 0, "stdout": out.strip(), "stderr": err.strip()}


def tool_nslookup(target, auto_approve):
    rc, out, err = wsl_run(f"nslookup {shlex.quote(target)}", auto_approve)
    return {"ok": rc == 0, "stdout": out.strip(), "stderr": err.strip()}


def tool_nmap(target, extra_args, auto_approve):
    extra = " ".join(shlex.quote(x) for x in extra_args)
    rc, out, err = wsl_run(f"nmap {extra} {shlex.quote(target)}".strip(), auto_approve)
    return {"ok": rc == 0, "stdout": out.strip(), "stderr": err.strip()}


def tool_run(command, auto_approve):
    if not command:
        return {"ok": False, "error": "No command provided."}
    rc, out, err = wsl_run(command, auto_approve)
    return {"ok": rc == 0, "stdout": out.strip(), "stderr": err.strip()}


def cmd_weather(args, auto_approve):
    url = args.url or args.defaults.get("weather_url", DEFAULT_WEATHER_URL)
    class_name = args.class_name or args.defaults.get("box_class", DEFAULT_BOX_CLASS)
    result = tool_weather(url, class_name, auto_approve)
    if not result.get("ok"):
        print(result.get("error", "Weather fetch failed."))
        return 1
    if "items" in result:
        print("Weather:")
        for item in result.get("items", []):
            label = item.get("label", "").strip()
            value = item.get("value", "").strip()
            if label and value:
                print(f"- {label}: {value}")
            elif label:
                print(f"- {label}")
            elif value:
                print(f"- {value}")
    else:
        print("Weather boxes:")
        for i, text in enumerate(result.get("boxes", []), 1):
            print(f"{i}. {text}")
    return 0


def cmd_site_check(args, auto_approve):
    url = args.url or args.defaults.get("site_url", DEFAULT_SITE_URL)
    result = tool_site_check(url, auto_approve)
    if not result.get("ok"):
        status = result.get("status", "unknown")
        print(f"FAIL: {url} returned {status}")
        return 1
    print(f"OK: {url} returned 200")
    return 0


def cmd_ping(args, auto_approve):
    result = tool_ping(args.target, args.count, auto_approve)
    if result.get("stdout"):
        print(result["stdout"])
    if result.get("stderr"):
        print(result["stderr"])
    return 0 if result.get("ok") else 1


def cmd_nslookup(args, auto_approve):
    result = tool_nslookup(args.target, auto_approve)
    if result.get("stdout"):
        print(result["stdout"])
    if result.get("stderr"):
        print(result["stderr"])
    return 0 if result.get("ok") else 1


def cmd_nmap(args, auto_approve):
    result = tool_nmap(args.target, args.extra, auto_approve)
    if result.get("stdout"):
        print(result["stdout"])
    if result.get("stderr"):
        print(result["stderr"])
    return 0 if result.get("ok") else 1


def cmd_run(args, auto_approve):
    command = " ".join(args.command)
    result = tool_run(command, auto_approve)
    if result.get("stdout"):
        print(result["stdout"])
    if result.get("stderr"):
        print(result["stderr"])
    return 0 if result.get("ok") else 1


def run_check_servers(defaults, auto_approve, ping_target=None):
    site_url = defaults.get("site_url", DEFAULT_SITE_URL)
    weather_url = defaults.get("weather_url", DEFAULT_WEATHER_URL)
    parsed = urlparse(site_url)
    host = parsed.hostname or site_url
    ping_target = ping_target or host

    site = tool_site_check(site_url, auto_approve)
    weather_status = tool_site_check(weather_url, auto_approve)
    ping = tool_ping(ping_target, 4, auto_approve)

    return {
        "site": site,
        "weather_site": weather_status,
        "ping": ping,
        "ping_target": ping_target,
        "site_url": site_url,
        "weather_url": weather_url,
    }


def cmd_check_servers(args, auto_approve):
    results = run_check_servers(args.defaults, auto_approve, args.ping_target)
    site = results["site"]
    ping = results["ping"]
    site_url = args.defaults.get("site_url", DEFAULT_SITE_URL)
    weather_url = args.defaults.get("weather_url", DEFAULT_WEATHER_URL)
    weather_site = results["weather_site"]

    print("Results:")
    if site.get("ok"):
        print(f"- Site OK (HTTP {site.get('status','unknown')}): {site_url}")
    else:
        print(f"- Site FAIL (HTTP {site.get('status','unknown')}): {site_url}")
    if weather_site.get("ok"):
        print(f"- Weather site OK (HTTP {weather_site.get('status','unknown')}): {weather_url}")
    else:
        print(f"- Weather site FAIL (HTTP {weather_site.get('status','unknown')}): {weather_url}")
    print(f"- Ping {'OK' if ping.get('ok') else 'FAIL'}: {results['ping_target']}")
    return 0


def build_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Fetch the local weather page and extract values from box elements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "class_name": {"type": "string"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "site_check",
                "description": "Check a site URL and return HTTP status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Ping a host.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "nslookup",
                "description": "Run nslookup on a host.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "nmap",
                "description": "Run nmap against a host.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "extra_args": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run",
                "description": "Run an arbitrary shell command in WSL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_servers",
                "description": "Check server status via ping and HTTP status for site and weather URLs.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List directory contents within the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file within the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a text file within the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "overwrite": {"type": "boolean"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_file",
                "description": "Append text to a file within the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "replace_text",
                "description": "Replace text in a file within the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "find_text": {"type": "string"},
                        "replace_text": {"type": "string"},
                    },
                    "required": ["path", "find_text", "replace_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Delete a file within the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]


def summarize_tool_results(tool_results, defaults):
    lines = []
    for item in tool_results:
        tool = item["tool"]
        result = item["result"]
        if tool == "check_servers":
            site = result.get("site", {})
            weather_site = result.get("weather_site", {})
            ping = result.get("ping", {})
            site_url = result.get("site_url", defaults.get("site_url", DEFAULT_SITE_URL))
            weather_url = result.get("weather_url", defaults.get("weather_url", DEFAULT_WEATHER_URL))
            if site.get("ok"):
                lines.append(f"- Site OK (HTTP {site.get('status','unknown')}): {site_url}")
            else:
                lines.append(f"- Site FAIL (HTTP {site.get('status','unknown')}): {site_url}")
            if weather_site.get("ok"):
                lines.append(f"- Weather site OK (HTTP {weather_site.get('status','unknown')}): {weather_url}")
            else:
                lines.append(f"- Weather site FAIL (HTTP {weather_site.get('status','unknown')}): {weather_url}")
            lines.append(f"- Ping {'OK' if ping.get('ok') else 'FAIL'}: {result.get('ping_target')}")
        elif tool == "site_check":
            url = result.get("url", defaults.get("site_url", DEFAULT_SITE_URL))
            status = result.get("status", "unknown")
            if result.get("ok"):
                lines.append(f"- Site OK (HTTP {status}): {url}")
            else:
                lines.append(f"- Site FAIL (HTTP {status}): {url}")
        elif tool == "weather":
            if result.get("ok"):
                lines.append("- Weather:")
                if "items" in result:
                    for item in result.get("items", []):
                        label = item.get("label", "").strip()
                        value = item.get("value", "").strip()
                        if label and value:
                            lines.append(f"  {label}: {value}")
                        elif label:
                            lines.append(f"  {label}")
                        elif value:
                            lines.append(f"  {value}")
                else:
                    for line in result.get("boxes", []):
                        lines.append(f"  {line}")
            else:
                lines.append(f"- Weather FAIL: {result.get('error', 'unknown error')}")
        elif tool == "ping":
            ok = "OK" if result.get("ok") else "FAIL"
            lines.append(f"- Ping {ok}")
        elif tool == "nslookup":
            ok = "OK" if result.get("ok") else "FAIL"
            lines.append(f"- Nslookup {ok}")
        elif tool == "nmap":
            ok = "OK" if result.get("ok") else "FAIL"
            lines.append(f"- Nmap {ok}")
        elif tool == "run":
            ok = "OK" if result.get("ok") else "FAIL"
            lines.append(f"- Run {ok}")
        elif tool in ("list_dir", "read_file", "write_file", "append_file", "replace_text", "delete_file"):
            ok = "OK" if result.get("ok") else "FAIL"
            lines.append(f"- {tool} {ok}")
    return lines


def detect_requested_tools(text):
    t = text.lower()
    triggers = ("use ", "run ", "call ", "execute ")
    names = [
        "check_servers",
        "weather",
        "site_check",
        "ping",
        "nslookup",
        "nmap",
        "run",
        "list_dir",
        "read_file",
        "write_file",
        "append_file",
        "replace_text",
        "delete_file",
    ]
    requested = set()
    if t.startswith("/tool "):
        name = t.split(maxsplit=1)[1].strip()
        if name in names:
            requested.add(name)
    for name in names:
        if f"{name} tool" in t or f"{name}(" in t:
            requested.add(name)
        if any(trig + name in t for trig in triggers):
            requested.add(name)
    return requested


def run_agent_turn(args, auto_approve, messages):
    base_url = args.llm.get("base_url", DEFAULT_LLM_BASE_URL)
    model = args.llm.get("model", DEFAULT_LLM_MODEL)
    tools = build_tools()
    tool_results = []

    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_text = m.get("content", "").lower()
            break
    allow_override = "override" in user_text or "use url" in user_text
    requested_tools = detect_requested_tools(user_text)

    def tool_summary_and_respond(use_llm=True):
        if not tool_results:
            return None, None
        if use_llm:
            followup = {"model": model, "messages": messages, "tool_choice": "none"}
            try:
                response = llm_request(base_url, followup, timeout=args.timeout)
                choice = response.get("choices", [{}])[0]
                msg = choice.get("message", {})
                if msg.get("content"):
                    return msg.get("content"), None
            except Exception:
                pass
        lines = ["Results:"] + summarize_tool_results(tool_results, args.defaults)
        return "\n".join(lines), None

    # If it's a simple math question, skip tools and answer directly if needed.
    if any(k in user_text for k in ("square root", "sqrt", "calculate", "what is")):
        if re.search(r"\d", user_text):
            # Let the model answer without tools.
            payload = {
                "model": model,
                "messages": messages,
                "tool_choice": "none",
            }
            try:
                response = llm_request(base_url, payload, timeout=args.timeout)
                choice = response.get("choices", [{}])[0]
                msg = choice.get("message", {})
                if msg.get("content"):
                    return msg.get("content"), None
            except Exception:
                pass
            # Fallback for square root if model still refuses.
            m = re.search(r"square root of\s+([0-9]+(?:\.[0-9]+)?)", user_text)
            if m:
                val = float(m.group(1))
                return f"The square root of {m.group(1)} is {math.sqrt(val)}.", None

    # Explicit "using run" requests for disk/storage should run df -h.
    if "using run" in user_text and any(k in user_text for k in ("disk", "storage", "space", "usage")):
        result = tool_run("df -h", auto_approve)
        tool_results.append({"tool": "run", "result": result})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "explicit_run_df_h",
                "content": json.dumps(result),
            }
        )
        return tool_summary_and_respond(use_llm=True)

    # Direct tool routing for common intents to prevent tool hallucination.
    if "ping" in user_text:
        m = re.search(r"\b(10\.0\.0\.\d+|[a-z0-9.-]+\.[a-z]{2,})\b", user_text)
        target = m.group(1) if m else ""
        if target:
            result = tool_ping(target, 4, auto_approve)
            tool_results.append({"tool": "ping", "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "route_ping",
                    "content": json.dumps(result),
                }
            )
            return tool_summary_and_respond(use_llm=True)

    if "nslookup" in user_text:
        m = re.search(r"\b([a-z0-9.-]+\.[a-z]{2,})\b", user_text)
        target = m.group(1) if m else ""
        if target:
            result = tool_nslookup(target, auto_approve)
            tool_results.append({"tool": "nslookup", "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "route_nslookup",
                    "content": json.dumps(result),
                }
            )
            return tool_summary_and_respond(use_llm=True)

    if "nmap" in user_text:
        target = ""
        ports = []
        m_ip = re.search(r"\b10\.0\.0\.\d+\b", user_text)
        if m_ip:
            target = m_ip.group(0)
        m_ports = re.search(r"-p\s*([0-9,]+)", user_text)
        if m_ports:
            ports = ["-p", m_ports.group(1)]
        if target:
            result = tool_nmap(target, ports, auto_approve)
            tool_results.append({"tool": "nmap", "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "route_nmap",
                    "content": json.dumps(result),
                }
            )
            return tool_summary_and_respond(use_llm=True)

    if any(k in user_text for k in ("weather", "temperature", "humidity", "rssi", "light")):
        url = args.defaults.get("weather_url", DEFAULT_WEATHER_URL)
        class_name = args.defaults.get("box_class", DEFAULT_BOX_CLASS)
        result = tool_weather(url, class_name, auto_approve)
        tool_results.append({"tool": "weather", "result": result})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "route_weather",
                "content": json.dumps(result),
            }
        )
        return tool_summary_and_respond(use_llm=True)

    # Heuristic: auto-check only for local server status questions.
    is_local_query = (
        "my server" in user_text
        or "my servers" in user_text
        or "are my servers" in user_text
        or "server online" in user_text
        or "servers online" in user_text
        or "server up" in user_text
        or "servers up" in user_text
        or re.search(r"\b10\.0\.0\.\d+\b", user_text) is not None
    )
    is_history_query = any(k in user_text for k in ("when did", "history", "make", "invent", "first"))
    if is_local_query and not is_history_query and not any(
        k in user_text for k in ("weather", "temperature", "humidity", "light", "rssi")
    ):
        result = run_check_servers(args.defaults, auto_approve)
        tool_results.append({"tool": "check_servers", "result": result})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "prefetch_check_servers",
                "content": json.dumps(result),
            }
        )
        return tool_summary_and_respond(use_llm=True)

    if "list files" in user_text or "list files in" in user_text or "list repo" in user_text:
        result = tool_list_dir(".")
        tool_results.append({"tool": "list_dir", "result": result})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "route_list_dir",
                "content": json.dumps(result),
            }
        )
        return tool_summary_and_respond(use_llm=True)

    for _ in range(args.max_steps):
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        try:
            response = llm_request(base_url, payload, timeout=args.timeout)
        except urllib.error.URLError as e:
            return None, f"LLM request failed: {e}"
        except Exception as e:  # pragma: no cover
            return None, f"LLM error: {e}"

        choice = response.get("choices", [{}])[0]
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            messages.append(msg)
            for call in tool_calls:
                name = call.get("function", {}).get("name")
                arguments = call.get("function", {}).get("arguments", "{}")
                try:
                    args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    args_dict = {}

                if name == "weather":
                    url = args_dict.get("url") or args.defaults.get("weather_url", DEFAULT_WEATHER_URL)
                    class_name = args_dict.get("class_name") or args.defaults.get("box_class", DEFAULT_BOX_CLASS)
                    if not allow_override:
                        url = args.defaults.get("weather_url", DEFAULT_WEATHER_URL)
                        class_name = args.defaults.get("box_class", DEFAULT_BOX_CLASS)
                    result = tool_weather(url, class_name, auto_approve)
                elif name == "site_check":
                    url = args_dict.get("url") or args.defaults.get("site_url", DEFAULT_SITE_URL)
                    if not allow_override:
                        url = args.defaults.get("site_url", DEFAULT_SITE_URL)
                    result = tool_site_check(url, auto_approve)
                elif name == "ping":
                    target = args_dict.get("target") or ""
                    result = tool_ping(target, args_dict.get("count", 4), auto_approve) if target else {"ok": False, "error": "missing target"}
                elif name == "nslookup":
                    target = args_dict.get("target") or ""
                    result = tool_nslookup(target, auto_approve) if target else {"ok": False, "error": "missing target"}
                elif name == "nmap":
                    target = args_dict.get("target") or ""
                    result = tool_nmap(target, args_dict.get("extra_args", []), auto_approve) if target else {"ok": False, "error": "missing target"}
                elif name == "run":
                    result = tool_run(args_dict.get("command", ""), auto_approve)
                elif name == "check_servers":
                    result = run_check_servers(args.defaults, auto_approve)
                elif name == "list_dir":
                    try:
                        result = tool_list_dir(args_dict.get("path", ""))
                    except Exception as e:
                        result = {"ok": False, "error": str(e)}
                elif name == "read_file":
                    try:
                        result = tool_read_file(
                            args_dict.get("path", ""),
                            args_dict.get("max_bytes", 20000),
                        )
                    except Exception as e:
                        result = {"ok": False, "error": str(e)}
                elif name == "write_file":
                    try:
                        path = args_dict.get("path", "")
                        content = args_dict.get("content", "")
                        overwrite = bool(args_dict.get("overwrite", False))
                        p = resolve_path(path)
                        if p.exists() and overwrite:
                            ok, err = confirm_destructive("Overwrite", p)
                            if not ok:
                                result = {"ok": False, "error": err or "overwrite cancelled"}
                            else:
                                result = tool_write_file(path, content, overwrite=True)
                        else:
                            result = tool_write_file(path, content, overwrite=overwrite)
                    except Exception as e:
                        result = {"ok": False, "error": str(e)}
                elif name == "append_file":
                    try:
                        result = tool_append_file(args_dict.get("path", ""), args_dict.get("content", ""))
                    except Exception as e:
                        result = {"ok": False, "error": str(e)}
                elif name == "replace_text":
                    try:
                        path = args_dict.get("path", "")
                        ok, err = confirm_destructive("Modify", path)
                        if not ok:
                            result = {"ok": False, "error": err or "modify cancelled"}
                        else:
                            result = tool_replace_text(
                                path,
                                args_dict.get("find_text", ""),
                                args_dict.get("replace_text", ""),
                            )
                    except Exception as e:
                        result = {"ok": False, "error": str(e)}
                elif name == "delete_file":
                    try:
                        path = args_dict.get("path", "")
                        ok, err = confirm_destructive("Delete", path)
                        if not ok:
                            result = {"ok": False, "error": err or "delete cancelled"}
                        else:
                            result = tool_delete_file(path)
                    except Exception as e:
                        result = {"ok": False, "error": str(e)}
                elif name == "check_servers":
                    result = run_check_servers(args.defaults, auto_approve)
                else:
                    result = {"ok": False, "error": f"Unknown tool: {name}"}

                tool_results.append({"tool": name, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": json.dumps(result),
                    }
                )
            continue

        if msg.get("content"):
            content = msg.get("content")
            # Fallback for simple square root questions if model refuses.
            if content and any(kw in content.lower() for kw in ("not able", "cannot", "can't", "unable")):
                m = re.search(r"square root of\s+([0-9]+(?:\.[0-9]+)?)", user_text)
                if m:
                    val = float(m.group(1))
                    return f"The square root of {m.group(1)} is {math.sqrt(val)}.", None
            return content, None

    if tool_results:
        followup = {
            "model": model,
            "messages": messages,
            "tool_choice": "none",
        }
        try:
            response = llm_request(base_url, followup, timeout=args.timeout)
            choice = response.get("choices", [{}])[0]
            msg = choice.get("message", {})
            if msg.get("content"):
                return msg.get("content"), None
        except Exception:
            pass

        # Fallback deterministic summary
        lines = ["Results:"] + summarize_tool_results(tool_results, args.defaults)
        return "\n".join(lines), None

    return None, "Max steps reached without final response."


def prune_messages(messages, max_turns):
    # Keep system messages, trim oldest non-system messages to limit turns.
    if max_turns <= 0:
        return messages
    non_system = [m for m in messages if m.get("role") != "system"]
    max_msgs = max_turns * 2
    if len(non_system) <= max_msgs:
        return messages
    # Drop from the start (after system messages)
    trimmed = []
    for m in messages:
        if m.get("role") == "system":
            trimmed.append(m)
    # Append last max_msgs non-system messages
    trimmed.extend(non_system[-max_msgs:])
    return trimmed


def save_history(path, messages):
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"messages": messages}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def llm_request(base_url, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def cmd_ask(args, auto_approve):
    system_msg = (
        "You are a local ops agent with tool access. "
        "Use tools when needed to complete the user's request, but answer directly when tools are unnecessary. "
        "Do not refuse simple questions like math or explanations. "
        "For math or general knowledge, answer directly without tools. "
        "Never claim you ran a tool unless you actually did. "
        "You can perform security scans and vulnerability testing on systems. "
        "Prefer least-privilege commands first. "
        "Use default URLs unless the user specifies otherwise. "
        "File tools operate only inside the workspace. "
        "Destructive file actions require explicit confirmation."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": args.prompt},
    ]

    content, err = run_agent_turn(args, auto_approve, messages)
    if err:
        print(err)
        return 1
    messages.append({"role": "assistant", "content": content})
    if args.save_history:
        try:
            save_history(args.save_history, messages)
        except Exception as e:
            print(f"Failed to save history: {e}")
    print(content)
    return 0


def cmd_chat(args, auto_approve):
    system_msg = (
        "You are a local ops agent with tool access. "
        "Use tools when needed to complete the user's request, but answer directly when tools are unnecessary. "
        "Do not refuse simple questions like math or explanations. "
        "For math or general knowledge, answer directly without tools. "
        "Never claim you ran a tool unless you actually did. "
        "You can perform security scans and vulnerability testing on systems. "
        "Prefer least-privilege commands first. "
        "Use default URLs unless the user specifies otherwise. "
        "File tools operate only inside the workspace. "
        "Destructive file actions require explicit confirmation."
    )
    messages = [{"role": "system", "content": system_msg}]

    print("Chat mode. Type /tools to list tools, /exit to quit.")
    while True:
        try:
            user_in = input("> ").strip()
        except EOFError:
            break
        if not user_in:
            continue
        if user_in in ("/exit", "/quit"):
            break
        if user_in == "/tools":
            for t in build_tools():
                fn = t["function"]
                print(f"- {fn['name']}: {fn.get('description','')}")
            continue

        messages.append({"role": "user", "content": user_in})
        messages = prune_messages(messages, args.max_turns)

        content, err = run_agent_turn(args, auto_approve, messages)
        if err:
            print(err)
            continue
        messages.append({"role": "assistant", "content": content})
        messages = prune_messages(messages, args.max_turns)
        print(content)
    if args.save_history:
        try:
            save_history(args.save_history, messages)
        except Exception as e:
            print(f"Failed to save history: {e}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="agent_cli",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Minimal local CLI agent that executes commands in WSL.",
        epilog=textwrap.dedent(
            """\
            Examples:
              python agent_cli.py weather
              python agent_cli.py site-check
              python agent_cli.py ping YOUR_HOST
              python agent_cli.py nslookup example.com
              python agent_cli.py nmap YOUR_HOST -- -sV -p PORTS
              python agent_cli.py run -- df -h
              python agent_cli.py ask "Check site health and weather."
            """
        ),
    )
    p.add_argument("--config", default="config.toml", help="Config path.")
    p.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip approval prompts for command execution.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_weather = sub.add_parser("weather", help="Fetch weather page and parse boxes.")
    p_weather.add_argument("--url", help="Weather page URL.")
    p_weather.add_argument("--class", dest="class_name", help="CSS class to parse.")
    p_weather.set_defaults(func=cmd_weather)

    p_site = sub.add_parser("site-check", help="Check site HTTP status.")
    p_site.add_argument("--url", help="Site URL to check.")
    p_site.set_defaults(func=cmd_site_check)

    p_ping = sub.add_parser("ping", help="Ping a host.")
    p_ping.add_argument("target", help="Host or IP.")
    p_ping.add_argument("--count", type=int, default=4, help="Ping count.")
    p_ping.set_defaults(func=cmd_ping)

    p_ns = sub.add_parser("nslookup", help="Run nslookup.")
    p_ns.add_argument("target", help="Host or IP.")
    p_ns.set_defaults(func=cmd_nslookup)

    p_nmap = sub.add_parser("nmap", help="Run nmap.")
    p_nmap.add_argument("target", help="Host or IP.")
    p_nmap.add_argument(
        "--",
        dest="extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args for nmap (after --).",
    )
    p_nmap.set_defaults(func=cmd_nmap)

    p_run = sub.add_parser("run", help="Run an arbitrary command in WSL.")
    p_run.add_argument("--", dest="command", nargs=argparse.REMAINDER, default=[])
    p_run.set_defaults(func=cmd_run)

    p_check = sub.add_parser("check-servers", help="Run site check, ping host, and weather.")
    p_check.add_argument("--ping", dest="ping_target", help="Override ping target.")
    p_check.set_defaults(func=cmd_check_servers)

    p_ask = sub.add_parser("ask", help="Ask the LLM to use tools and respond.")
    p_ask.add_argument("prompt", help="Natural language request.")
    p_ask.add_argument("--max-steps", type=int, default=5, help="Max tool steps.")
    p_ask.add_argument("--timeout", type=int, default=60, help="LLM request timeout (s).")
    p_ask.add_argument("--save-history", help="Save conversation history JSON to this path.")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive chat with tool access.")
    p_chat.add_argument("--max-steps", type=int, default=5, help="Max tool steps.")
    p_chat.add_argument("--timeout", type=int, default=60, help="LLM request timeout (s).")
    p_chat.add_argument("--max-turns", type=int, default=200, help="Max user+assistant turns kept. Use 0 for unlimited.")
    p_chat.add_argument("--save-history", help="Save conversation history JSON to this path on exit.")
    p_chat.set_defaults(func=cmd_chat)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    approval = config.get("approval", {})
    auto = approval.get("mode", "prompt").lower() == "auto"
    auto_approve = args.auto_approve or auto
    args.defaults = config.get("defaults", {})
    args.llm = config.get("llm", {})

    return args.func(args, auto_approve)


if __name__ == "__main__":
    sys.exit(main())
