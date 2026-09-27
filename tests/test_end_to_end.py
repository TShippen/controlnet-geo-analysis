"""End-to-end run of every analysis through the MCP client with real checkpoints.

Uses the model directory named in the project's .env file and skips when the
checkpoints are not installed there.
"""

from collections.abc import AsyncIterator, Callable, Iterable
from pathlib import Path

import pytest
import torch
from mcp import Client
from mcp.types import ImageContent, TextContent

from conftest import write_test_image
from controlnet_mcp.analysis import AnalysisService
from controlnet_mcp.cache import AnalysisCache
from controlnet_mcp.checkpoints import REQUIRED_CHECKPOINTS, CheckpointSpec
from controlnet_mcp.config import Settings
from controlnet_mcp.model_manager import ModelManager
from controlnet_mcp.processors import ANALYSIS_KINDS
from controlnet_mcp.server import build_server

pytestmark = [pytest.mark.anyio, pytest.mark.slow, pytest.mark.integration]


@pytest.fixture
def manager(installed_checkpoints: Callable[[Iterable[CheckpointSpec]], Path]) -> ModelManager:
    model_dir = installed_checkpoints(REQUIRED_CHECKPOINTS)
    return ModelManager(model_dir, torch.device("cpu"), max_loaded=1)


@pytest.fixture
def settings(tmp_path: Path, manager: ModelManager) -> Settings:
    references = tmp_path / "references"
    references.mkdir()
    write_test_image(references / "gradient.png", size=(96, 64), color=(120, 80, 200))
    return Settings(
        reference_image_dir=references,
        model_dir=manager.model_dir,
        output_dir=tmp_path / "outputs",
        default_detect_resolution=64,
        max_loaded_models=1,
        device="cpu",
    )


@pytest.fixture
async def client(settings: Settings, manager: ModelManager) -> AsyncIterator[Client]:
    service = AnalysisService(settings, manager, AnalysisCache(settings.output_dir))
    async with Client(build_server(service)) as connected:
        yield connected


async def test_all_analyses_end_to_end(client: Client, manager: ModelManager) -> None:
    for kind in ANALYSIS_KINDS:
        result = await client.call_tool("analyze_image", arguments_for(kind))

        assert result.is_error is not True, f"{kind}: {result.content}"
        text, image = result.content
        assert isinstance(text, TextContent)
        assert isinstance(image, ImageContent)
        assert len(manager.loaded_detectors) <= 1
        assert "%" in text.text or "straight edges" in text.text, kind
        assert "Ask for it" not in text.text, kind

    for kind in ANALYSIS_KINDS:
        cached = await client.call_tool("analyze_image", arguments_for(kind))

        assert isinstance(cached.content[0], TextContent)
        assert "cache" in cached.content[0].text, kind


def arguments_for(kind: str) -> dict[str, object]:
    arguments: dict[str, object] = {"filename": "gradient.png", "analysis": kind, "resolution": 64}
    if kind == "segments":
        arguments["box"] = [0.3, 0.25, 0.8, 0.85]
    return arguments
