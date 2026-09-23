"""In-process tests of the MCP tools: discovery, image content, and error behavior."""

import base64
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import torch
from mcp import Client
from mcp.types import ImageContent, TextContent

from conftest import write_test_image
from controlnet_mcp.analysis import AnalysisService
from controlnet_mcp.cache import AnalysisCache
from controlnet_mcp.checkpoints import PREPARE_COMMAND
from controlnet_mcp.config import Settings
from controlnet_mcp.model_manager import ModelManager
from controlnet_mcp.server import SERVER_INSTRUCTIONS, build_server

MODEL_NAMES = ("Zoe", "MLSD", "SAM", "BAE", "BEiT", "Canny", "ControlNet")

pytestmark = pytest.mark.anyio


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    (tmp_path / "references").mkdir()
    (tmp_path / "models").mkdir()
    write_test_image(tmp_path / "references" / "chair.png", size=(64, 32))
    write_test_image(tmp_path / "references" / "table.jpg", size=(16, 16))
    return Settings(
        reference_image_dir=tmp_path / "references",
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "outputs",
        default_detect_resolution=64,
        max_loaded_models=1,
        device="cpu",
    )


@pytest.fixture
def service(settings: Settings) -> AnalysisService:
    manager = ModelManager(settings.model_dir, torch.device("cpu"), max_loaded=1)
    return AnalysisService(settings, manager, AnalysisCache(settings.output_dir))


@pytest.fixture
async def client(settings: Settings, service: AnalysisService) -> AsyncIterator[Client]:
    async with Client(build_server(settings, service)) as connected:
        yield connected


async def test_lists_three_tools_with_schemas(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {"list_reference_images", "get_reference_image", "analyze_image"}
    analysis_schema = tools["analyze_image"].input_schema["properties"]["analysis"]
    assert set(analysis_schema["enum"]) == {
        "depth",
        "normals",
        "lineart",
        "lines",
        "segments",
        "canny",
    }
    assert tools["analyze_image"].input_schema["required"] == ["filename", "analysis"]


async def test_analyze_schema_has_box_and_point(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    properties = tools["analyze_image"].input_schema["properties"]
    assert "box" in properties
    assert "point" in properties


async def test_agent_facing_text_has_no_model_names(client: Client) -> None:
    texts = [SERVER_INSTRUCTIONS]
    for tool in (await client.list_tools()).tools:
        texts.append(tool.description or "")
        for field in tool.input_schema["properties"].values():
            texts.append(field.get("description", ""))

    for text in texts:
        for name in MODEL_NAMES:
            assert name not in text, f"{name!r} appears in agent-facing text: {text[:80]}"


async def test_segments_without_prompt_is_error(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image", {"filename": "chair.png", "analysis": "segments"}
    )

    assert result.is_error is True
    assert "box" in result.content[0].text
    assert "point" in result.content[0].text


async def test_prompt_on_canny_is_error(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image", {"filename": "chair.png", "analysis": "canny", "point": [0.5, 0.5]}
    )

    assert result.is_error is True
    assert "whole image" in result.content[0].text


async def test_malformed_box_is_error(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image",
        {"filename": "chair.png", "analysis": "segments", "box": [0.9, 0.1, 0.2, 0.5]},
    )

    assert result.is_error is True
    assert "top-left" in result.content[0].text


async def test_list_reference_images_returns_metadata(client: Client) -> None:
    result = await client.call_tool("list_reference_images", {})

    assert result.is_error is not True
    entries = result.structured_content["result"]
    assert [entry["filename"] for entry in entries] == ["chair.png", "table.jpg"]
    assert entries[0] == {"filename": "chair.png", "width": 64, "height": 32, "format": "PNG"}


async def test_get_reference_image_returns_image_content(client: Client) -> None:
    result = await client.call_tool("get_reference_image", {"filename": "chair.png"})

    assert result.is_error is not True
    (block,) = result.content
    assert isinstance(block, ImageContent)
    assert block.mime_type == "image/png"
    assert base64.b64decode(block.data)[:8] == b"\x89PNG\r\n\x1a\n"


async def test_get_reference_image_jpeg_mime(client: Client) -> None:
    result = await client.call_tool("get_reference_image", {"filename": "table.jpg"})

    (block,) = result.content
    assert isinstance(block, ImageContent)
    assert block.mime_type == "image/jpeg"


async def test_get_reference_image_traversal_is_error(client: Client) -> None:
    result = await client.call_tool("get_reference_image", {"filename": "../chair.png"})

    assert result.is_error is True
    assert isinstance(result.content[0], TextContent)
    assert "reference image directory" in result.content[0].text


async def test_analyze_canny_returns_text_and_image(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image", {"filename": "chair.png", "analysis": "canny", "resolution": 64}
    )

    assert result.is_error is not True
    text, image = result.content
    assert isinstance(text, TextContent)
    assert isinstance(image, ImageContent)
    assert "canny" in text.text
    assert image.mime_type == "image/png"
    assert result.structured_content is None


async def test_analyze_unknown_kind_is_error(client: Client) -> None:
    result = await client.call_tool("analyze_image", {"filename": "chair.png", "analysis": "pose"})

    assert result.is_error is True


async def test_analyze_resolution_out_of_range_is_error(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image", {"filename": "chair.png", "analysis": "canny", "resolution": 32}
    )

    assert result.is_error is True
    assert "64" in result.content[0].text


async def test_analyze_missing_checkpoint_is_error(client: Client) -> None:
    result = await client.call_tool("analyze_image", {"filename": "chair.png", "analysis": "depth"})

    assert result.is_error is True
    assert PREPARE_COMMAND in result.content[0].text
    assert "ZoeD_M12_N.pt" in result.content[0].text


async def test_analyze_repeated_call_serves_cache(client: Client) -> None:
    arguments = {"filename": "chair.png", "analysis": "canny"}
    await client.call_tool("analyze_image", arguments)

    second = await client.call_tool("analyze_image", arguments)

    assert isinstance(second.content[0], TextContent)
    assert "cache" in second.content[0].text
