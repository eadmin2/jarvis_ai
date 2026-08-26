# J.A.R.V.I.S — Voice + HUD for Hermes Agent

A self-hosted, Iron-Man-style voice assistant and command center built on top of
[Hermes Agent](https://github.com/NousResearch/hermes-agent) (NousResearch's
open-source autonomous agent). Talk to a *real* agent — one with persistent
memory, terminal access, web search, file tools, and 80+ skills — through a
glowing arc-reactor HUD in any browser on your LAN, or a push-to-talk client.

**Everything runs on your own hardware.** The only cloud calls are your LLM
provider (via Hermes) and, by default, ElevenLabs for the voice — or skip that
entirely with the free, self-hosted Kokoro TTS engine, switchable live from the
HUD with no restart. Speech-to-text is fully local (Whisper on CPU).

## Demo

[![Watch the J.A.R.V.I.S demo](https://img.youtube.com/vi/YNI9pm3h6x8/hqdefault.jpg)](https://youtu.be/YNI9pm3h6x8)

▶ **[Watch the demo on YouTube](https://youtu.be/YNI9pm3h6x8)** — live transcription, agent tool calls, holographic media panels, and the cinematic boot, all in real time.

## What it does


Click the ring and speak. Your words transcribe **live on screen** while you
talk. The transcript goes to Hermes Agent, which actually *does things* — reads
and writes files, runs commands, searches the web, remembers you across
sessions — and the reply streams back as speech, sentence by sentence, while
the rest is still being generated. Typical round trip: 3–5 seconds.

The HUD around the ring is a real control center:

- **Live agent activity** — watch tool calls happen with command previews
- **STOP button** — halt a runaway agent turn mid-tool-call (Esc works too)
- **Approval cards** — dangerous commands pause for your ALLOW/DENY
- **Interrupt-aware barge-in** — cut it off mid-sentence; it knows exactly
  what you heard and what you didn't
- **Hands-free mode** — switch from push-to-talk to continuous listening and
  just speak; on-device voice activity detection opens and closes each turn,
  with an adjustable sensitivity slider and automatic mic muting while the
  agent is talking so it can't hear itself
- **Embedded dashboards** — Hermes' kanban board and session browser pop up
  in animated viewers, fully interactive
- **Holographic media panels** — say *"show me a video of how arc reactors
  work, on screen"* and a panel swoops in from Z-depth, traces its frame,
  materializes through a scanline, and plays the video. The agent drives it
  through a bundled Hermes plugin (`hud_display`); panels can fly into
  left/right thirds, and "clear the screen" sweeps them away
- **Usage tracking** — tokens/day, turns, ElevenLabs quota bar
- **Switchable voice engine** — flip between ElevenLabs and a free,
  self-hosted Kokoro TTS engine live from the HUD, no restart needed
- **Live model & thinking controls** — pick the Claude model from the HUD and
  switch it mid-conversation without losing context, and dial reasoning effort
  between FAST (speech latency) and DEEP (slow, for when you want it to actually
  think) per task
- **Multi-line chat with images** — Shift+Enter for a new line, and a paperclip
  button to send screenshots or photos straight to the model (camera and library
  on iPhone/iPad); images are downscaled in the browser before they're sent
- **Conversation history** — every turn is logged to `logs/conversation_history.jsonl`; open it any time from the VIEWS panel to scroll back through what you and Jarvis said, filterable per conversation
- **OmniFocus next actions** — pull the tasks you've tagged `next` into a HUD
  panel on demand, so what you're working on sits next to what the agent is doing.
  Reached through LMCP on your Mac, and cached so a sleeping laptop shows the last
  known list rather than an empty panel (the ClickUp source is still available too)
- **Machines panel** — live CPU/GPU stats for the host and remote workers
- **Cinematic boot** — press `B`: panels flicker in, ring spins up,
  "Systems online. Good morning."
- **Privacy filter** — secret-shaped strings are redacted before any text
  reaches cloud TTS
- **Optional GPU ears** — point it at any NVIDIA machine on your LAN running
  the included sidecar and transcription jumps to `large-v3-turbo` at ~0.2 s,
  with automatic fallback to local Whisper when that machine is off
- **Mobile-ready** — responsive layout + Add to Home Screen = full-screen
  Jarvis app on your phone

## Architecture

```
 Browser HUD (any LAN device)          Host machine (tested on macOS / Apple Silicon)
 ── https/wss :443 ──────────┐   ┌──────────────────────────────────────┐
   mic · speaker · panels    ├──►│ voice pipeline server (this repo)    │
                             │   │  STT: faster-whisper (local, free)   │   ┌─────────────────┐
 Push-to-talk client         │   │  TTS: ElevenLabs Flash (streaming)   ├──►│ Hermes Agent     │
 ── ws :8765 ────────────────┘   │  HUD + auth + dashboard TLS proxy    │   │  API :8642 (lo)  │
                                 └──────────────────────────────────────┘   │  memory · tools  │
                                                                             │  skills · cron   │
                                                                             └─────────────────┘
```

One brain, many faces: voice and typed chat share a single persistent Hermes
session, so each knows what you said to the other — and memory survives every
restart.

## Requirements

- A machine for the server (tested: Mac mini, Apple Silicon). Linux should
  work with minor changes (launchd → systemd).
- [Hermes Agent](https://hermes-agent.nousresearch.com/docs/) installed and
  configured with an LLM provider
- Python 3.11+
- An [ElevenLabs](https://elevenlabs.io) API key (free tier works; ~0.5
  credits/char on Flash) — or run [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) yourself instead, a free
  self-hosted TTS engine (see `voice.kokoro` in `config/server.example.yaml`);
  both can be configured at once and switched live from the HUD
- Any modern browser on the LAN

## Install

Full walkthrough in [docs/SETUP.md](docs/SETUP.md). Short version:

```bash
# 1. Enable the Hermes Agent API server
cat >> ~/.hermes/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')
JARVIS_HUD_TOKEN=$(python3 -c 'import secrets;print("jarvis-"+secrets.token_hex(3))')
ELEVENLABS_API_KEY=your-key-here
EOF
hermes gateway   # or set up its LaunchAgent / service

# 2. This repo
git clone https://github.com/YOURNAME/jarvis-hermes-hud
cd jarvis-hermes-hud/server
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn requests pyyaml numpy anthropic \
    RealtimeSTT faster-whisper silero-vad websockets psutil
cp config/server.example.yaml config/server.yaml   # edit: your ElevenLabs voice_id etc.
scripts/make-certs.sh                              # self-signed TLS (browser mic needs it)
scripts/make-boot-audio.sh YourName                # one-time boot greeting synthesis

# 3. Run
.venv/bin/python server.py
# open https://YOUR_HOST/hud/ → accept cert → enter your JARVIS_HUD_TOKEN → talk
```

For auto-start on boot, see [launchd/](launchd/) (macOS) — the plists document
two non-obvious macOS traps (external-drive TCC and log paths) that cost us an
evening.

## Usage

| Action | How |
|---|---|
| Talk | Click the ring (or Space) · speak · click again to send |
| Hands-free | PUSH TO TALK / CONTINUOUS LISTENING toggle in panel 01 (remembered between visits) |
| Sync tasks | ⟳ SYNC OMNIFOCUS in the ACTIVE TASKS panel (tasks tagged `next`) |
| New line in chat | Shift+Enter (Enter sends) |
| Attach an image | 📎 beside the chat input · max 4 per message |
| Change model | MODELS LOADOUT panel → Model dropdown (applies to the current session) |
| Think harder | MODELS LOADOUT panel → Thinking dropdown → DEEP (much slower) |
| Stop the agent | red ■ STOP button or Esc |
| Barge in | click the ring while it's speaking |
| Typed chat | input bar at the bottom (same conversation as voice) |
| Cinematic boot | press `B` |
| Kanban / dashboards | VIEWS panel → animated pop-up viewers |
| Phone | open the HUD → Add to Home Screen |

## Repo layout

```
server/          FastAPI voice pipeline + HUD host (the core of this project)
server/hud/      single-file HUD (vanilla JS, no build step)
server/scripts/  start/stop/health/smoke + cert & boot-audio generators
client/          optional Windows/Linux push-to-talk Python client (wake word capable)
worker/          optional GPU sidecars: big-model STT server + stats agent for the Machines panel
hermes-plugin/   Hermes tool plugin: lets the agent summon/dismiss HUD media panels
launchd/         macOS auto-start templates with hard-won TCC + FD-limit notes
docs/            SETUP, ARCHITECTURE (protocols/endpoints), TROUBLESHOOTING
```

## Security model

- The Hermes API key never reaches the browser: the HUD talks through a
  strict allowlist proxy on the voice server.
- All HUD endpoints + dashboard proxy + browser WebSockets are gated by a
  token (cookie, entered once per device).
- Hermes' API binds to loopback only; the dashboard binds to loopback only.
- Secret-shaped strings are redacted before text leaves for cloud TTS.
- LAN-only by design — do not port-forward this to the internet.

## Credits & license

Built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous
Research. HUD aesthetics inspired by
[jarvis-dashboard](https://github.com/AndrewKochulab/jarvis-dashboard).
STT by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) /
[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT). Voice by
[ElevenLabs](https://elevenlabs.io).

MIT — see [LICENSE](LICENSE). Use it, fork it, build your own Jarvis.
