# Full Setup Guide

Tested on macOS (Apple Silicon) with Hermes Agent v0.16. Allow ~1 hour.

## 1. Hermes Agent (the brain)

Install Hermes Agent and give it an LLM provider — follow the
[official quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart).
Verify `hermes` works in your terminal before continuing.

Enable its API server in `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=<long random secret>      # required — full toolset incl. terminal!
```

Start the gateway (`hermes gateway`) and verify:

```bash
KEY=$(grep '^API_SERVER_KEY=' ~/.hermes/.env | cut -d= -f2)
curl -H "Authorization: Bearer $KEY" http://127.0.0.1:8642/health
# {"status": "ok", ...}
```

For the founder-local Codex route, authenticate Hermes with its native
`openai-codex` flow and set both the conversational and API-session model to
`gpt-5.6-terra` in `config.yaml`:

```yaml
model:
  provider: openai-codex
  default: gpt-5.6-terra
gateway:
  api_server:
    model_name: gpt-5.6-terra
security:
  redact_secrets: true
```

This uses the Hermes Codex/ChatGPT OAuth integration. It does not launch a
local `codex` CLI subprocess, and it does not require storing an OpenAI API key
for conversational inference.

Recommended: add voice-behavior rules to your global `~/.hermes/SOUL.md`
(short spoken sentences, no markdown aloud, never speak secrets, announce risky
actions and wait for approval). The agent — not the voice server — should own
its personality.

## 2. Voice provider

The example config uses OpenAI TTS. Add its key to `~/.hermes/.env`:

```bash
OPENAI_API_KEY=...
```

To use ElevenLabs instead, set `voice.provider: elevenlabs`, select its model
and voice id, and add `ELEVENLABS_API_KEY`. For its remote quota bar, give the
key the **User → Read** permission. Local character usage works without it.

`voice.preset` selects a founder-local voice profile from `voice.presets`.
Presets may change only the model, voice, instructions and speed. Put names and
acronyms under `voice.pronunciations`. An optional `voice.fallback` provider is
used only when the primary fails before sending any audio; it never splices a
second voice into an utterance already playing.

## 3. The voice pipeline server (this repo)

```bash
cd server
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config/server.example.yaml config/server.yaml
```

Note: `faster-whisper` and `silero-vad` are REQUIRED — recent RealtimeSTT
releases treat them as optional extras and fail at runtime without them
(silently for VAD, loudly for the engine). The first start takes 60–90 s
(torch import + model download); subsequent starts are faster.

Edit `config/server.yaml`: keep `brain.provider: hermes`, select the voice
provider, and adjust `machines:` (or delete it). The first run downloads the
Whisper model (~460 MB for `small.en`).

Gbrain is an optional personal-knowledge retrieval layer, not the conversational
agent. Install and initialize its local CLI, verify `gbrain query "test"`, then
set `gbrain.enabled: true`. JARVIS runs the CLI on loopback-only founder turns,
frames the result as untrusted reference data, and continues with Hermes if
Gbrain is absent, times out, or returns an error. No Gbrain HTTP server is
required.

When Hermes is connected to Gbrain over MCP, install the deterministic router
to retrieve knowledge before the model call whenever the founder says
"Gbrain", "FrankBrain", "Frank Knowledge", or expresses an explicit knowledge
lookup intent:

```bash
cp -R hermes-plugin/gbrain_router ~/.hermes/plugins/gbrain_router
hermes plugins enable gbrain-router
# restart the Hermes gateway
```

The plugin calls the current MCP names `mcp__gbrain__query` and
`mcp__gbrain__search`, bounds and labels returned text as untrusted context,
and asks for page-slug citations. Write language never causes an automatic
mutation: Hermes must make an explicit `mcp__gbrain__put_page` tool call after
deriving a clear slug and page body.

### TLS certificates (required for browser microphone)

Browsers only expose the mic to secure origins:

```bash
scripts/make-certs.sh            # auto-detects your LAN IP for the SAN
```

Trust `certs/cert.pem` on each device (macOS: Keychain; Windows:
`certutil -user -addstore Root cert.pem`; iPhone: download
`https://host:8765/hud/jarvis.cer`, install profile, then enable in
Settings → General → About → Certificate Trust Settings).

Optional but nice: rename your host's mDNS name (`sudo scutil --set
LocalHostName jarvis` on macOS) so the HUD lives at `https://jarvis.local/hud/`.
Port 443 needs the wildcard bind already set in the example config.

### HUD access token

```bash
echo "JARVIS_HUD_TOKEN=jarvis-$(python3 -c 'import secrets;print(secrets.token_hex(3))')" >> ~/.hermes/.env
```

The HUD asks for this once per device. Omit the variable entirely to disable
auth (not recommended).

### Boot greeting (optional; the helper currently uses ElevenLabs)

```bash
scripts/make-boot-audio.sh YourFirstName
```

### Run it

```bash
.venv/bin/python server.py
```

Wait ~40 s for the STT model to warm, then open `https://YOUR_HOST/hud/`.
Run `scripts/jarvis-health.sh` to check all five services.

In the Voice Link panel, **Auto-send after silence** is opt-in and preserves
manual click/Space control. Remote GPU STT can return confidence; low-confidence
turns pause for review before they reach the brain. Approval prompts are spoken
without reading sensitive command details and time out to deny.

## 4. Auto-start on boot (macOS)

Copy the two plists from `launchd/`, edit the `/PATH/TO` and `YOUR_USER`
placeholders, then:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.jarvis.voice.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.jarvis.dashboard.plist
```

**Read the comments in the plists** — there are two macOS traps (external-drive
TCC permissions and log-path location) that produce silent hangs or `EX_CONFIG`
errors if ignored. Day-to-day management: `scripts/jarvis-{start,stop,restart,health}.sh`.

### Founder-local FrankBrain refresh

FrankBrain uses a loop-safe two-lane sync; Hermes/Jarvis continues to query the
canonical knowledge backend directly. Canonical pages flow into the generated
`wiki/frankbrain/` visualization projection. LLM Wiki-authored pages created by
Save to Wiki or auto-ingest flow back into the dedicated federated `frankbrain`
source. Projection files and app aggregates never flow back.

Register the write-back source once, then install the stable local scripts and
LaunchAgent definition. It refreshes every fifteen minutes and at login:

```bash
python3 server/scripts/install_frankbrain_refresh.py --install
gbrain sources add frankbrain \
  --path "$HOME/Library/Application Support/FrankBrain/writeback" \
  --name "FrankBrain authored pages" --federated --force
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.frankbrain.refresh.plist
```

The sync is lock-protected and updates changed Markdown files atomically. It
does not propagate deletions into Gbrain; deletion remains an explicit governed
operation. Generated projection pages should be treated as read-only because a
later canonical refresh replaces them. The job opens no network listener and
writes its log to `~/Library/Logs/frankbrain-refresh.log`.

## 5. Optional extras

**Push-to-talk client** (Windows/Linux box with a mic):

```bash
cd client
python -m venv .venv && .venv/Scripts/pip install sounddevice websockets numpy openwakeword
python client.py --push-to-talk --server ws://YOUR_HOST:8765/ws --list-devices
```

Without `--push-to-talk`, wake-word mode stops an utterance after local silence
detection instead of always recording for the maximum duration. Tune
`--speech-rms-threshold`, `--silence-seconds`, and `--min-record-seconds` for
the founder's microphone and room.

**GPU worker stats** (shows in the Machines panel): copy
`worker/worker_stats.py` to the worker machine, `pip install psutil`, run it,
and point a `machines:` entry's `stats_url` at `http://worker-ip:8767/stats`.

**GPU speech recognition** — the single biggest quality upgrade if you own any
NVIDIA machine on the LAN. On that machine:

```
cd worker
python -m venv .venv
.venv/Scripts/pip install faster-whisper fastapi uvicorn nvidia-cublas-cu12 nvidia-cudnn-cu12
copy run-stt.example.bat run-stt.bat    # fill in your JARVIS_HUD_TOKEN value
run-stt.bat
```

Then uncomment the `stt.remote:` block in the server's `server.yaml` (point
`url` at the GPU machine) and restart the voice server. Result: `large-v3-turbo`
accuracy at ~0.2 s per utterance, with automatic fallback to the local model
whenever the GPU machine is off. For auto-start at Windows logon, drop a
`JarvisSTT.vbs` in `shell:startup` (template in the .bat comments).

**Phone app feel**: open the HUD in Safari/Chrome on your phone →
Share → Add to Home Screen.

**Let the agent put things on your screen** (the showstopper): install the
bundled Hermes plugin so "show me a video of X on screen" makes holographic
media panels materialize on every open HUD:

```bash
cp -R hermes-plugin/hud_display ~/.hermes/plugins/hud_display
# edit ~/.hermes/plugins/hud_display/schemas.py: replace YOUR_HOST
hermes plugins enable hud_display     # repeat with -p <profile> if you use profiles
# restart your hermes gateway
```

The plugin needs `JARVIS_HUD_TOKEN` in `~/.hermes/.env` (same token as the
HUD). Directory name must stay a valid Python module name — hyphens silently
break plugin discovery. The agent then has `hud_display` / `hud_dismiss`
tools; YouTube links play as embedded video, `position` left/right lets it
fly in multiple panels from different vectors.

## 6. Verify everything

```bash
scripts/jarvis-health.sh   # all five rows OK
scripts/jarvis-smoke.sh    # synthesized voice turn through the full stack (macOS)
```

Then the real test: click the ring and ask "what's in your memory file?" —
a real agent answers with real file contents.
