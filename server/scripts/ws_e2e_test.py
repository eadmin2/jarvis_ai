#!/usr/bin/env python3
"""End-to-end voice pipeline test: stream a WAV through the WS, print events."""
import asyncio
import json
import sys
import wave

import websockets

URL = "ws://127.0.0.1:8765/ws"


async def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: ws_e2e_test.py INPUT.wav [conversation] [--realtime]")
    wav_path = sys.argv[1]
    conversation = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "jarvis-debug"
    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "need 16k mono"
        pcm = w.readframes(w.getnframes())
    audio_bytes = 0
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "sample_rate": 16000,
                                  "format": "pcm_s16le", "channels": 1,
                                  "conversation": conversation}))
        step = 2560  # 80 ms
        realtime = "--realtime" in sys.argv  # pace audio like a live mic (tests partials)
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step])
            await asyncio.sleep(0.08 if realtime else 0.005)
        await ws.send(json.dumps({"type": "stop"}))
        if "--stop-after" in sys.argv:
            delay = float(sys.argv[sys.argv.index("--stop-after") + 1])
            async def stopper():
                await asyncio.sleep(delay)
                print(f"(sending stop_run after {delay}s)")
                await ws.send(json.dumps({"type": "stop_run"}))
            asyncio.ensure_future(stopper())
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=300)
            if isinstance(msg, bytes):
                audio_bytes += len(msg)
                continue
            ev = json.loads(msg)
            t = ev.get("type")
            if t == "transcript":
                print("TRANSCRIPT:", ev.get("text"))
            elif t == "partial_transcript":
                print("PARTIAL:", ev.get("text"))
            elif t == "run_started":
                print("RUN:", ev.get("run_id"))
            elif t == "approval_request":
                print("APPROVAL_REQ:", str(ev.get("data"))[:160])
            elif t == "status":
                print("STATUS:", ev.get("message"))
            elif t == "agent_status":
                print("AGENT:", ev.get("state"), ev.get("tool", ""), (ev.get("preview") or "")[:60])
                if ev.get("state") == "stopped":
                    print("STOPPED_OK")
                    break
            elif t == "error":
                print("ERROR:", ev.get("message"))
                break
            elif t == "done":
                tm = ev.get("timing", {})
                print("RESPONSE:", (tm.get("response_text") or "")[:400])
                print("AUDIO_BYTES:", audio_bytes)
                print("PROVIDER:", tm.get("llm_provider"), tm.get("llm_model"))
                print("TOOLS:", tm.get("tools_used"))
                print("LATENCY eos->first_audio:", tm.get("end_of_speech_to_first_audio_seconds"),
                      "total:", tm.get("total_turn_seconds"))
                break

if __name__ == "__main__":
    asyncio.run(main())
