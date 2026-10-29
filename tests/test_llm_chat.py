"""Whether a model server answers, reached as the model's client reaches it (paco.llm.chat): any
answer will do; a connection that fails is said with its cause."""

import http.server
import socket
import threading

from pydantic import SecretStr

from paco.llm.chat import unanswered


def test_a_server_answering_anything_answers() -> None:
    # A company's gateway refusing the model list (400 Bad Request) still answers.
    class Refusing(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(400)
            self.end_headers()

    server = http.server.HTTPServer(("127.0.0.1", 0), Refusing)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        said = unanswered(f"http://127.0.0.1:{server.server_port}/v1", SecretStr("EMPTY"), 10)
    finally:
        server.shutdown()
        server.server_close()

    assert said is None


def test_nothing_listening_is_said_with_its_cause() -> None:
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]

    said = unanswered(f"http://127.0.0.1:{port}/v1", SecretStr("EMPTY"), 10)

    assert said is not None and said.startswith("ConnectError")
