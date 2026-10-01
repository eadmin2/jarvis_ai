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

Recommended: add voice-behavior rules to your global `~/.hermes/SOUL.md`
(short spoken sentences, no markdown aloud, never speak secrets, announce risky
actions and wait for approval). The agent — not the voice server — should own
its personality.

## 2. ElevenLabs (the voice)

Create an API key at elevenlabs.io and pick a voice from their library, noting
its `voice_id`. Add to `~/.hermes/.env`:

```bash
ELEVENLABS_API_KEY=...
```

For the HUD's quota bar, give the key the **User → Read** permission.

### Alternative: 60db

Add `SIXTYDB_API_KEY=your-key` to `~/.hermes/.env`, then replace the `voice`
section in `server/config/server.yaml`:

```yaml
voice:
  provider: sixtydb
  voice_id: your-workspace-voice-id
  output_format: pcm_16000
  speed: 1.0
  # model: 60db-fast-v01  # optional; omit for the workspace default
```

Get your voice ID from the authenticated [60db voice catalog](https://docs.60db.ai/api-reference/tts/get-voices).
The server requests mono PCM16 at 16 kHz for browser playback. It validates WAV
metadata and JSON/NDJSON audio envelopes, rejects redirects and synthesis errors,
and caps responses at 32 MiB. Speed accepts 0.5 to 2.0; sentences are limited to
5000 characters. Text still passes through the existing privacy filter.

Barge-in stops playback and signals the HTTP worker to stop consuming audio.
A blocked HTTP read can take up to 30 seconds to release; connect timeout is
10 seconds. WAV/JSON responses buffer before validation; NDJSON PCM streams.
The boot greeting generator and ElevenLabs quota bar remain ElevenLabs-specific.
With only a 60db key, skip `make-boot-audio.sh`; live replies do not need it.

Run `cd server && python -m unittest test_sixtydb test_server_tts` to check the protocol adapter.
Live 60db synthesis and microphone playback were not available during this
contribution; verify voice availability, greeting, interruption, and playback
with your workspace credentials before deployment.

## 3. The voice pipeline server (this repo)

```bash
cd server
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn requests pyyaml numpy anthropic \
    RealtimeSTT faster-whisper silero-vad websockets psutil
cp config/server.example.yaml config/server.yaml
```

Note: `faster-whisper` and `silero-vad` are REQUIRED — recent RealtimeSTT
releases treat them as optional extras and fail at runtime without them
(silently for VAD, loudly for the engine). The first start takes 60–90 s
(torch import + model download); subsequent starts are faster.

Edit `config/server.yaml`: set `voice.voice_id`, and adjust the `machines:`
list (or delete it). The first run downloads the Whisper model (~460 MB for
`small.en`).

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

### Boot greeting (one-time, ~110 ElevenLabs characters)

```bash
scripts/make-boot-audio.sh YourFirstName
```

### Run it

```bash
.venv/bin/python server.py
```

Wait ~40 s for the STT model to warm, then open `https://YOUR_HOST/hud/`.
Run `scripts/jarvis-health.sh` to check all five services.

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

## 5. Optional extras

**Push-to-talk client** (Windows/Linux box with a mic):

```bash
cd client
python -m venv .venv && .venv/Scripts/pip install sounddevice websockets numpy openwakeword
python client.py --push-to-talk --server ws://YOUR_HOST:8765/ws --list-devices
```

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
