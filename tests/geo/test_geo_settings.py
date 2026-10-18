"""The geophysicist's own settings: where its server listens, beside PACo's guided server."""

import pytest

from paco.geo.settings import GeoSettings


def test_the_chat_finds_the_server_where_it_listens_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("PACO_GEO_HOST", "PACO_GEO_PORT", "PACO_GEO_MCP_URL"):
        monkeypatch.delenv(name, raising=False)

    settings = GeoSettings()

    assert settings.mcp_url == f"http://{settings.host}:{settings.port}/mcp"
    # Beside paco-server (8000), a local vLLM (8001) and the tunnel to Cloud Run (8002).
    assert settings.port not in (8000, 8001, 8002)
