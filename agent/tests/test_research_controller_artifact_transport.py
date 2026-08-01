from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from src.research_controller.client.dsa_client import DsaLoopClient, DsaProtocolError


def _client(handler) -> DsaLoopClient:
    transport = httpx.MockTransport(handler)
    return DsaLoopClient(
        base_url="http://127.0.0.1:8011",
        client_factory=lambda **kwargs: httpx.Client(
            base_url=kwargs["base_url"], timeout=kwargs["timeout"], transport=transport
        ),
    )


def test_fixed_length_upload_and_hash_checked_download(tmp_path: Path) -> None:
    content = b"factor-values"
    sha = hashlib.sha256(content).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/metadata"):
            body = json.dumps(
                {"data": {"artifact_id": "artifact_test", "size_bytes": len(content), "sha256": sha}}
            ).encode()
            return httpx.Response(
                200,
                stream=httpx.ByteStream(body),
                headers={"Content-Length": str(len(body))},
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                stream=httpx.ByteStream(content),
                headers={"Content-Length": str(len(content))},
            )
        uploaded = request.read()
        assert request.headers["content-length"] == str(len(content))
        assert "transfer-encoding" not in request.headers
        assert uploaded == content
        body = json.dumps({"data": {"received_size_bytes": len(uploaded)}}).encode()
        return httpx.Response(
            200,
            stream=httpx.ByteStream(body),
            headers={"Content-Length": str(len(body))},
        )

    client = _client(handler)
    source = tmp_path / "source.csv"
    source.write_bytes(content)
    assert client.upload_artifact_content("upload_test", source)["status"] == "ok"
    destination = tmp_path / "download.csv"
    result = client.download_artifact("artifact_test", destination)
    assert result["status"] == "ok"
    assert destination.read_bytes() == content


def test_download_declared_size_mismatch_never_publishes_partial(tmp_path: Path) -> None:
    content = b"abc"
    sha = hashlib.sha256(content).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/metadata"):
            body = json.dumps(
                {"data": {"artifact_id": "artifact_test", "size_bytes": 4, "sha256": sha}}
            ).encode()
            return httpx.Response(
                200,
                stream=httpx.ByteStream(body),
                headers={"Content-Length": str(len(body))},
            )
        return httpx.Response(
            200,
            stream=httpx.ByteStream(content),
            headers={"Content-Length": "3"},
        )

    destination = tmp_path / "must_not_exist.csv"
    with pytest.raises(DsaProtocolError, match="content_length_mismatch"):
        _client(handler).download_artifact("artifact_test", destination)
    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []
