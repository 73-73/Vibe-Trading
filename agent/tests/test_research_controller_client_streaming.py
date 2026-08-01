"""Real-socket bounded response tests for the DSA loopback client (P2-19)."""

from __future__ import annotations

import gzip
import socket
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import httpx
import pytest

from src.research_controller.client.dsa_client import (
    DsaLoopClient,
    DsaProtocolError,
    DsaUnavailableError,
)

MAX_RESPONSE_BYTES = 10_000_000


@contextmanager
def _raw_http_response(parts: list[tuple[bytes, float]]) -> Iterator[str]:
    """Serve one handcrafted HTTP/1.1 response over a real loopback socket."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    stopped = threading.Event()

    def _serve() -> None:
        try:
            conn, _address = listener.accept()
            with conn:
                request = bytearray()
                while b"\r\n\r\n" not in request:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    request.extend(chunk)
                for payload, delay in parts:
                    if delay:
                        time.sleep(delay)
                    try:
                        conn.sendall(payload)
                    except (BrokenPipeError, ConnectionResetError):
                        return
        finally:
            stopped.set()
            listener.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        stopped.wait(timeout=2)
        listener.close()
        thread.join(timeout=2)


def _response_headers(*headers: str) -> bytes:
    return ("HTTP/1.1 200 OK\r\n" + "\r\n".join(headers) + "\r\n\r\n").encode("ascii")


def _request(parts: list[tuple[bytes, float]], *, timeout: float = 1.0) -> dict:
    with _raw_http_response(parts) as base_url:
        return DsaLoopClient(base_url=base_url, timeout=timeout).request("probe", "GET", "/probe")


def test_declared_content_length_and_exact_boundary_are_accepted() -> None:
    body = b'{"x":"' + (b"a" * (MAX_RESPONSE_BYTES - 8)) + b'"}'
    assert len(body) == MAX_RESPONSE_BYTES
    result = _request(
        [
            (
                _response_headers(
                    f"Content-Length: {len(body)}",
                    "Content-Type: application/json",
                    "Connection: close",
                )
                + body,
                0,
            )
        ],
        timeout=5.0,
    )
    assert result["status"] == "ok"
    assert len(result["data"]["x"]) == MAX_RESPONSE_BYTES - 8


def test_chunked_json_is_accumulated_with_a_hard_limit() -> None:
    body = b'{"status":"ok","data":{"value":1}}'
    wire_body = b"".join(
        f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"
        for chunk in (body[:7], body[7:19], body[19:])
    ) + b"0\r\n\r\n"
    result = _request(
        [
            (
                _response_headers(
                    "Transfer-Encoding: chunked",
                    "Content-Type: application/json",
                    "Connection: close",
                )
                + wire_body,
                0,
            )
        ]
    )
    assert result["data"]["value"] == 1


def test_real_http_content_length_must_be_legal() -> None:
    with pytest.raises(DsaProtocolError):
        _request([(_response_headers("Content-Length: abc", "Connection: close") + b"{}", 0)])


@pytest.mark.parametrize(
    "headers",
    [
        [("Content-Length", "2"), ("Content-Length", "2")],
        [("Content-Length", "2, 2")],
    ],
)
def test_normalized_response_still_requires_one_content_length(headers: list[tuple[str, str]]) -> None:
    # httpx/h11 may normalize identical duplicate wire headers before exposing
    # the Response.  This direct framing test covers every Response shape the
    # bounded reader itself can observe.
    response = httpx.Response(200, headers=headers, content=b"{}")
    with pytest.raises(DsaProtocolError):
        DsaLoopClient._read_buffered_response_body(response)


def test_oversized_declared_length_is_rejected_before_body_read() -> None:
    with pytest.raises(DsaProtocolError, match="too_large"):
        _request(
            [
                (
                    _response_headers(
                        f"Content-Length: {MAX_RESPONSE_BYTES + 1}",
                        "Connection: close",
                    ),
                    0,
                )
            ]
        )


def test_lying_content_length_and_incomplete_chunk_are_protocol_errors() -> None:
    with pytest.raises(DsaProtocolError):
        _request(
            [
                (
                    _response_headers("Content-Length: 20", "Connection: close") + b"{}",
                    0,
                )
            ]
        )

    with pytest.raises(DsaProtocolError):
        _request(
            [
                (
                    _response_headers("Transfer-Encoding: chunked", "Connection: close")
                    + b"20\r\n{}",
                    0,
                )
            ]
        )


def test_gzip_json_is_bounded_after_decompression() -> None:
    body = gzip.compress(b'{"status":"ok","data":{"value":2}}')
    result = _request(
        [
            (
                _response_headers(
                    f"Content-Length: {len(body)}",
                    "Content-Encoding: gzip",
                    "Connection: close",
                )
                + body,
                0,
            )
        ]
    )
    assert result["data"]["value"] == 2


def test_gzip_bomb_and_unknown_encoding_are_rejected() -> None:
    bomb = gzip.compress(b'{"x":"' + b"a" * MAX_RESPONSE_BYTES + b'"}')
    with pytest.raises(DsaProtocolError, match="too_large"):
        _request(
            [
                (
                    _response_headers(
                        f"Content-Length: {len(bomb)}",
                        "Content-Encoding: gzip",
                        "Connection: close",
                    )
                    + bomb,
                    0,
                )
            ]
        )

    with pytest.raises(DsaProtocolError, match="unsupported_content_encoding"):
        _request(
            [
                (
                    _response_headers(
                        "Content-Length: 2",
                        "Content-Encoding: br",
                        "Connection: close",
                    )
                    + b"{}",
                    0,
                )
            ]
        )


def test_slow_body_is_transport_unavailable_not_unbounded_wait() -> None:
    body = b'{"status":"ok"}'
    headers = _response_headers(f"Content-Length: {len(body)}", "Connection: close")
    with pytest.raises(DsaUnavailableError):
        _request([(headers, 0), (body, 0.2)], timeout=0.05)
