# Architecture & Protocols

## How a voice turn flows

1. Browser captures mic at 16 kHz mono (AudioContext created at 16 kHz —
   avoids resampling artifacts that wreck Whisper accuracy) and streams int16
   PCM over WebSocket `/ws`.
2. While you speak, the server runs incremental Whisper passes every ~1.2 s
   and emits `partial_transcript` events (live captions).
3. On stop: final transcription — the server first tries the optional **GPU
   STT worker** (`stt.remote`, big model, ~0.2 s) and falls back to local
   Whisper if it's unreachable — then the transcript goes to Hermes via its **Sessions
   API** (`POST /api/sessions/{id}/chat/stream`, SSE). Session ids persist in
   `logs/hermes_sessions.json` per conversation name, so memory survives
   restarts. Typed chat (`/api/chat`) uses the *same* session.
4. SSE events parsed: `run.started` (run id → STOP support), `assistant.delta`
   (text), `tool.started` (name + preview → HUD activity), `assistant.completed`
   (incl. `interrupted` flag), `run.completed` (token usage), `*approval*`
   (→ HUD approval cards).
5. Text is sentence-split, markdown/think-block-stripped, **secret-redacted**,
   then each sentence streams through the configured OpenAI or ElevenLabs TTS
   adapter as 16 kHz PCM while generation continues. OpenAI's 24 kHz PCM is
   converted by a stateful resampler whose phase persists across HTTP chunks.
6. TTS audio is framed by `speech_segment`/`speech_segment_end`. The HUD sends
   `speech_played` only after every source in that segment ends naturally, so
   interruption memory reflects what reached the founder's speakers.
7. One cancellation token owns the brain SSE response and TTS HTTP response.
   Barge-in closes both resources before the next local turn is started.

Hermes and Gbrain implement one normalized brain contract: run start, text,
tool use, approval, memory update, completion, interruption and error. Gbrain's
local endpoints and upstream SSE event names are configurable in `server.yaml`.

Why the Sessions API and not `/v1/responses` or `/v1/runs`: on Hermes v0.16,
sessions are the only surface that combines named persistent memory, run ids,
tool events, and approval events in one stream.

## WebSocket protocol (voice clients)

Client → server (JSON + binary):

```
{"type":"start","sample_rate":16000,"format":"pcm_s16le","channels":1,"conversation":"jarvis-main"?}
<binary int16 PCM chunks>
{"type":"stop"}
{"type":"stop_run"}
{"type":"approval_decision","run_id":...,"approval_id":...,"decision":"allow"|"deny"}
{"type":"speech_played","segment_id":"turn:segment"}
{"type":"transcript_review","review_id":"turn:review","text":"corrected text"}
```

`start` during an active turn = barge-in: the server cancels the turn, stops
the selected brain run, records the last HUD-acknowledged sentence, and prefixes
the next turn with an interruption note so memory matches what was actually heard.

Server → client:

```
status · partial_transcript · transcript · run_started{run_id}
agent_status{state: transcribing|thinking|tool_use(+tool,preview)|speaking|interrupted|stopped|error}
approval_request{data,run_id} · error · done{timing}
speech_segment{segment_id,text} · speech_segment_end{segment_id}
<binary TTS PCM — frames split at arbitrary byte boundaries; buffer odd bytes>
```

## HTTP endpoints (voice server)

| Endpoint | Purpose |
|---|---|
| `/hud/` | the HUD (static, single file) |
| `/api/hermes/{path}` | **allowlist** proxy to Hermes API, injects the bearer key (GET: health, capabilities, skills, toolsets, jobs, sessions; POST: v1/responses only) |
| `/api/chat` | typed chat turn on the shared voice session |
| `/api/machines` | host psutil stats + remote workers from config |
| `/api/usage` | provider-neutral local token/character tally plus optional ElevenLabs quota |
| `/api/summon` | broadcasts a holographic media panel (`{media, src, title, position}` or `{action:"dismiss"}`) to every connected HUD over its WebSocket — this is what the bundled `hud_display` Hermes plugin calls |
| port 9443 (separate app) | TLS reverse proxy of the Hermes dashboard with WebSocket bridge and frame-header stripping, so the HTTPS HUD can iframe it |

## Auth model

`JARVIS_HUD_TOKEN` (env) gates `/api/*`, the dashboard proxy, and
browser-originated WebSockets (Origin allowlist + cookie). Native clients (the
PTT client, test scripts) send no Origin header and are exempt — the
threat model is a malicious *website* doing cross-origin requests against your
LAN, not your own processes. The Hermes API key never reaches any browser.

## Latency profile (Apple Silicon, small.en on CPU)

STT finalize ~0.7 s · Hermes first token 1–3 s (more with tools) · first
audible audio ~3 s · total simple turn ~4 s. The biggest lever is the LLM
behind Hermes; the second is the Whisper model size.

## Gotchas encoded in this repo (learned the hard way)

1. **SSE charset**: Hermes' event stream has no charset header; Python
   `requests` then decodes as Latin-1 → mojibake. The client forces UTF-8.
2. **macOS port 443**: non-root binds only work on the wildcard address
   (`0.0.0.0`), not a specific IP.
3. **launchd + external drives**: see launchd/*.plist comments (TCC hang on
   `getcwd`, EX_CONFIG from external log paths, venv-python invocation).
4. **Browser mic**: requires a secure context — hence the self-signed TLS and
   the per-device cert trust.
5. **TTS PCM frames** arrive at arbitrary byte boundaries; decode only
   complete int16 pairs and carry the leftover byte.
6. **Near-silent audio** makes faster-whisper raise ("No clip timestamps
   found") — both STT paths treat any transcription exception as an empty
   transcript rather than failing the turn.
7. **Windows venv + NVIDIA pip wheels**: locate the cuBLAS/cuDNN DLLs via
   `sys.prefix` (`site.getsitepackages()` misses the venv) and
   `os.add_dll_directory` them before importing ctranslate2.
8. **Startup**: listeners open immediately; the local Whisper fallback warms
   in the background, exactly once and race-locked (the startup hook fires
   once per uvicorn listener — concurrent recorder inits crash lifespans).
9. **FD limits**: launchd grants 256 file descriptors by default and the
   server idles at ~244 — set NumberOfFiles=8192 in the plist (shipped) and
   close streaming responses in `finally` (done) or it dies within hours.
10. **Orphaned STT children steal ports**: the recorder's child process has a
    cmdline without "server.py", survives naive pkills, and can inherit the
    listen socket — the next spawn then fails its first bind and uvicorn's
    SystemExit cancels ALL listeners. Stop by PORT ownership
    (`lsof -ti tcp:PORT -sTCP:LISTEN | xargs kill -9`) — shipped in
    `scripts/jarvis-stop.sh`.
11. **Agent tool-choice**: SOUL.md prose cannot redirect the model away from
    attractive built-in tools (it kept "showing" videos in its own invisible
    browser); a first-class plugin tool with an explicit schema wins
    instantly. Hence `hermes-plugin/hud_display`.
