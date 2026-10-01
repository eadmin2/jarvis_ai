import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
import wave
from unittest.mock import patch

import sixtydb

PCM = b"\x01\x00" * 320
VOICE = {"voice_id": "workspace-voice", "output_format": "pcm_16000"}


def wav(rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(PCM)
    return output.getvalue()


class Response:
    def __init__(self, body, kind="application/json", status=200):
        self.body, self.status_code = body, status
        self.headers = {"Content-Type": kind}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), 17):
            yield self.body[start:start + 17]


class SixtyDBTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SIXTYDB_API_KEY": "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def synthesize(self, response, voice=None):
        with patch("sixtydb.requests.post", return_value=response) as post:
            audio = b"".join(sixtydb.tts_chunks("Hello", voice or VOICE))
        self.assertTrue(response.closed)
        return audio, post

    def test_binary_wav_and_bearer_request(self):
        audio, post = self.synthesize(Response(wav(), "audio/wav"), {**VOICE, "model": "60db-fast-v01"})
        self.assertEqual(audio, PCM)
        call = post.call_args
        self.assertEqual(call.args, ("https://api.60db.ai/tts-synthesize",))
        self.assertFalse(call.kwargs["allow_redirects"])
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(call.kwargs["json"]["audio_config"], {"audio_encoding": "LINEAR16", "sample_rate_hertz": 16000})
        self.assertEqual(call.kwargs["json"]["model_id"], "60db-fast-v01")

    def test_json_and_split_ndjson_with_double_encoded_audio(self):
        encoded = lambda value: base64.b64encode(value).decode()
        envelope = {"result": {"audioContent": encoded(PCM)}}
        nested = {"audio_base64": encoded(json.dumps(envelope).encode())}
        for body, kind in [(json.dumps(nested).encode(), "application/json"),
                           (json.dumps({"sample_rate": 16000}).encode() + b"\n" + json.dumps(nested).encode(), "application/x-ndjson")]:
            self.assertEqual(self.synthesize(Response(body, kind))[0], PCM)

    def test_invalid_audio_and_status_close_response(self):
        cases = [Response(wav(24000), "audio/wav"), Response(wav()[:-2], "audio/wav"),
                 Response(b'{"success":false}'), Response(b'{"encoding":"mp3"}'),
                 Response(b'{"audio_base64":"bad!"}'), Response(b'{"audio_base64":"AA=="}'),
                 Response(b'[]'), Response(b'{}'), Response(b'', status=302),
                 Response(b'', status=401), Response(b'ID3bad', "application/octet-stream")]
        for response in cases:
            with self.subTest(body=response.body, status=response.status_code):
                with patch("sixtydb.requests.post", return_value=response), self.assertRaises((RuntimeError, ValueError)):
                    list(sixtydb.tts_chunks("Hello", VOICE))
                self.assertTrue(response.closed)

    def test_invalid_input_never_calls_http(self):
        for text, voice in [("", VOICE), ("x" * 5001, VOICE), ("Hi", {}),
                            ("Hi", {**VOICE, "speed": float("nan")}),
                            ("Hi", {**VOICE, "output_format": "mp3"})]:
            with patch("sixtydb.requests.post") as post, self.assertRaises((RuntimeError, ValueError)):
                list(sixtydb.tts_chunks(text, voice))
            post.assert_not_called()

    def test_cancelled_stream_and_generator_close_release_http(self):
        data = json.dumps({"audio_base64": base64.b64encode(PCM).decode()}).encode() + b"\n"
        for close in (False, True):
            stopped = threading.Event()
            response = Response(data * 2, "application/x-ndjson")
            with patch("sixtydb.requests.post", return_value=response):
                stream = sixtydb.tts_chunks("Hello", VOICE, stopped)
                self.assertEqual(next(stream), PCM)
                if close:
                    stream.close()
                else:
                    stopped.set()
                    self.assertEqual(list(stream), [])
            self.assertTrue(response.closed)

    def test_real_http_wav_contract(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                captured.append((self.headers.get("Authorization"), body))
                audio = wav()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(audio)))
                self.end_headers()
                self.wfile.write(audio)

            def log_message(self, *_):
                pass
        http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=http.serve_forever)
        thread.start()
        post = sixtydb.requests.post
        try:
            with patch("sixtydb.requests.post", side_effect=lambda url, **options: post(
                f"http://127.0.0.1:{http.server_port}/tts", **options
            )):
                self.assertEqual(b"".join(sixtydb.tts_chunks("Hello", VOICE)), PCM)
            self.assertEqual(captured[0][0], "Bearer test-key")
            self.assertEqual(captured[0][1]["voice_id"], "workspace-voice")
        finally:
            http.shutdown()
            http.server_close()
            thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_size_limit_and_nested_envelope(self):
        response = Response(wav(), "audio/wav")
        with patch("sixtydb.MAX_BYTES", 16), patch("sixtydb.requests.post", return_value=response), self.assertRaises(RuntimeError):
            list(sixtydb.tts_chunks("Hi", VOICE))
        self.assertTrue(response.closed)
        value = {"audio_base64": base64.b64encode(PCM).decode()}
        for _ in range(6):
            value = {"audio_base64": base64.b64encode(json.dumps(value).encode()).decode()}
        with self.assertRaises(RuntimeError):
            sixtydb._decode(value)


if __name__ == "__main__":
    unittest.main()
