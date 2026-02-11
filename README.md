# Agent-CLI
Powerful custom vLLM inference cli with tools and skills enabled via WSL command runner

Minimal local CLI agent that executes approved commands inside WSL and includes a few built-in tasks (weather fetch, site check, ping, nslookup, nmap).

## Prereqs
- Windows with WSL installed
- A Linux distro in WSL with:
  - `curl`
  - `ping` (iputils)
  - `nslookup` (dnsutils)
  - `nmap` (optional)
- Python 3.11+ on Windows

## Quick Start
1. Update `config.toml` if needed.
2. Run:

```powershell
python agent_cli.py weather
python agent_cli.py site-check
python agent_cli.py ask "Check my servers and temperature."
python agent_cli.py check-servers
```

## Commands
```powershell
python agent_cli.py weather
python agent_cli.py site-check
python agent_cli.py ping YOUR_HOST
python agent_cli.py nslookup example.com
python agent_cli.py nmap YOUR_HOST -- -sV -p 80,443
python agent_cli.py run -- df -h
python agent_cli.py ask "Check site health and weather."
python agent_cli.py check-servers
python agent_cli.py --auto-approve chat
```

File ops (workspace-only):
```powershell
python agent_cli.py --auto-approve ask "Create a file named notes.txt with hello"
python agent_cli.py --auto-approve ask "Read notes.txt"
```

## Approval Mode
In `config.toml`:
```toml
[approval]
mode = "prompt" # or "auto"
```
You can also pass `--auto-approve` to skip prompts for a single run.

## Notes
- All command execution happens inside WSL using `wsl.exe`.
- The weather parser is designed for my local weather station - it looks for elements with CSS class `box` by default. Set `box_class` in `config.toml`
- LLM integration uses the OpenAI-compatible endpoint at `llm.base_url` (vLLM).
- File tools operate only inside the repository workspace. Deletions/overwrites require confirmation.

