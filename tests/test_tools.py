"""In-process tests of the MCP tools: discovery, image content, and error behavior."""

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import torch
from mcp import Client
from mcp.types import ImageContent, TextContent
from PIL import Image

from conftest import (
    SAMPLE_MEASUREMENT,
    fake_test_line_detection,
    model_names_in_test_text,
    sampled_test_spec,
    write_test_image,
)
from controlnet_mcp.analysis import AnalysisService
from controlnet_mcp.cache import AnalysisCache
from controlnet_mcp.checkpoints import PREPARE_COMMAND
from controlnet_mcp.config import MeasurementSetting, Settings
from controlnet_mcp.model_manager import ModelManager
from controlnet_mcp.processors import (
    PROCESSORS,
    SAMPLED_KINDS,
    AnalysisOptions,
    AnalysisOutput,
    ProcessorSpec,
)
from controlnet_mcp.regions import CropRegion
from controlnet_mcp.segmentation import RegionPrompt
from controlnet_mcp.server import SERVER_INSTRUCTIONS, build_server

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
    async with Client(build_server(service)) as connected:
        yield connected


@asynccontextmanager
async def client_measuring(settings: Settings, mode: MeasurementSetting) -> AsyncIterator[Client]:
    """A client whose server and service both emit measurements at ``mode``."""
    tuned = settings.model_copy(update={"result_measurements": mode})
    manager = ModelManager(tuned.model_dir, torch.device("cpu"), max_loaded=1)
    service = AnalysisService(tuned, manager, AnalysisCache(tuned.output_dir))
    async with Client(build_server(service)) as connected:
        yield connected


async def tool_description(client: Client, name: str) -> str:
    """The description the server advertises for one tool."""
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    return tools[name].description or ""


def fake_segments_spec() -> ProcessorSpec:
    """Stands in for the segments processor, whose checkpoint the fast suite does not install.

    Its description carries the same "Ask for it" phrasing as the real ones, so
    a result text can be checked for having dropped the description.
    """

    def build(model_dir: Path, device: torch.device) -> object:
        return object()

    def run(
        detector: object,
        image: Image.Image,
        resolution: int,
        prompt: RegionPrompt | None,
        region: CropRegion,
        options: AnalysisOptions,
    ) -> AnalysisOutput:
        return AnalysisOutput(Image.new("RGB", (resolution, resolution)), SAMPLE_MEASUREMENT)

    return ProcessorSpec(
        kind="segments",
        description="Region outline: the part you pointed at. Ask for it to isolate a component.",
        use_when="Use it to split the object into components.",
        checkpoints=(),
        build=build,
        run=run,
        accepts_prompt=True,
    )


async def test_lists_tools_with_schemas(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {
        "list_reference_images",
        "get_reference_image",
        "analyze_image",
        "sample_analysis",
        "compare_images",
    }
    analysis_schema = tools["analyze_image"].input_schema["properties"]["analysis"]
    assert set(analysis_schema["enum"]) == {
        "depth",
        "normals",
        "lineart",
        "lines",
        "perspective",
        "segments",
        "canny",
    }
    assert tools["analyze_image"].input_schema["required"] == ["filename", "analysis"]


async def test_perspective_text_names_its_assumptions(client: Client) -> None:
    description = await tool_description(client, "analyze_image")

    assert "are assumptions" in description
    assert "It gives no position, distance, or size." in description


async def test_perspective_text_states_its_limits(client: Client) -> None:
    """The description says how loosely edges are grouped and joined, and what a far point costs."""
    description = await tool_description(client, "analyze_image")

    assert "within 3 degrees of that group's vanishing point" in description
    assert "within 2 pixels of the other's line" in description
    assert "placed less precisely than a near one" in description
    assert "Edges shorter than 20 pixels at the working resolution" in description
    assert "At most 200 edges are detected" in description
    assert "0 degrees running to the image right and 90 degrees straight up" in description


async def test_line_length_on_perspective_is_error(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image",
        {"filename": "chair.png", "analysis": "perspective", "line_length": "long"},
    )

    assert result.is_error is True
    assert "line_length" in result.content[0].text


async def test_analyze_schema_has_box_and_point(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    properties = tools["analyze_image"].input_schema["properties"]
    assert "fractions" in properties["box"]["description"]
    assert "fractions" in properties["point"]["description"]
    assert "segments" in tools["analyze_image"].description


def schema_descriptions(schema: object) -> list[str]:
    """Every description in a JSON schema, at any depth, nested models included."""
    if isinstance(schema, list):
        return [text for item in schema for text in schema_descriptions(item)]
    if not isinstance(schema, dict):
        return []
    found = schema_descriptions(list(schema.values()))
    description = schema.get("description")
    if isinstance(description, str):
        found.append(description)
    return found


async def test_agent_facing_text_has_no_model_names(client: Client) -> None:
    texts = [SERVER_INSTRUCTIONS]
    for spec in PROCESSORS.values():
        texts += [spec.description, spec.use_when, spec.values_description or ""]
    for tool in (await client.list_tools()).tools:
        texts.append(tool.description or "")
        texts += schema_descriptions(tool.input_schema)
        texts += schema_descriptions(tool.output_schema)

    for text in texts:
        assert model_names_in_test_text(text) == [], f"in agent-facing text: {text[:80]}"


async def test_schema_descriptions_reach_the_fields_of_a_returned_model(client: Client) -> None:
    """The model-name check reads output fields, which sit in nested definitions."""
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    descriptions = schema_descriptions(tools["sample_analysis"].output_schema)

    sample = tools["sample_analysis"].output_schema["$defs"]["Sample"]
    assert sample["properties"]["on_boundary"]["description"] in descriptions


async def test_analysis_result_text_has_no_model_names(settings: Settings) -> None:
    """The analysis is asked for by the name "canny", in lower case, and the text repeats it."""
    for mode in ("off", "brief", "full"):
        async with client_measuring(settings, mode) as client:
            result = await client.call_tool(
                "analyze_image", {"filename": "chair.png", "analysis": "canny", "resolution": 64}
            )

        text = result.content[0]
        assert isinstance(text, TextContent)
        assert model_names_in_test_text(text.text) == [], mode


async def test_comparison_result_text_has_no_model_names(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both outcomes: paired in a shared frame, and an alignment withheld for lack of features."""
    fake_test_line_detection(monkeypatch)

    for mode in ("off", "brief", "full"):
        for align in ("none", "fit"):
            async with client_measuring(settings, mode) as client:
                result = await client.call_tool(
                    "compare_images",
                    {"first": "chair.png", "second": "chair.png", "align": align},
                )

            text = result.content[0]
            assert isinstance(text, TextContent)
            assert model_names_in_test_text(text.text) == [], f"{mode}, align {align}"


async def test_error_text_has_no_model_names(client: Client) -> None:
    """Errors over what was asked for. A missing checkpoint is apart: its error names the file."""
    results = [
        await client.call_tool(
            "analyze_image",
            {"filename": "chair.png", "analysis": "perspective", "line_length": "long"},
        ),
        await client.call_tool(
            "compare_images", {"first": "chair.png", "second": "table.jpg", "align": "none"}
        ),
        await client.call_tool(
            "sample_analysis",
            {"filename": "chair.png", "analysis": "canny", "points": [[0.5, 0.5]]},
        ),
    ]

    for result in results:
        assert result.is_error is True
        assert model_names_in_test_text(result.content[0].text) == []


async def test_analyze_schema_has_crop(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    crop = tools["analyze_image"].input_schema["properties"]["crop"]["description"]
    assert "fractions" in crop
    assert "segments" in crop


async def test_crop_appears_in_result_text(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image",
        {"filename": "chair.png", "analysis": "canny", "resolution": 64, "crop": [0, 0, 0.5, 1]},
    )

    assert result.is_error is not True
    text = result.content[0]
    assert isinstance(text, TextContent)
    assert "cropped to x 0.00 to 0.50, y 0.00 to 1.00" in text.text


async def test_crop_on_segments_is_error(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image",
        {
            "filename": "chair.png",
            "analysis": "segments",
            "point": [0.5, 0.5],
            "crop": [0, 0, 0.5, 1],
        },
    )

    assert result.is_error is True
    assert "box or point" in result.content[0].text


async def test_analyze_schema_has_line_length(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    line_length = tools["analyze_image"].input_schema["properties"]["line_length"]
    assert "lines only" in line_length["description"]


async def test_line_length_on_canny_is_error(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image", {"filename": "chair.png", "analysis": "canny", "line_length": "long"}
    )

    assert result.is_error is True
    assert "line_length" in result.content[0].text


async def test_analyze_schema_has_exclude_and_extent(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    properties = tools["analyze_image"].input_schema["properties"]
    assert "leave out" in properties["exclude"]["description"]
    assert "lone point" in properties["extent"]["description"]


async def test_extent_with_a_box_is_error(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image",
        {
            "filename": "chair.png",
            "analysis": "segments",
            "box": [0.1, 0.1, 0.9, 0.9],
            "extent": "largest",
        },
    )

    assert result.is_error is True
    assert "lone point" in result.content[0].text


async def test_exclude_on_canny_is_error(client: Client) -> None:
    result = await client.call_tool(
        "analyze_image",
        {
            "filename": "chair.png",
            "analysis": "canny",
            "point": [0.5, 0.5],
            "exclude": [[0.1, 0.1]],
        },
    )

    assert result.is_error is True
    assert "whole image" in result.content[0].text


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


async def test_off_text_matches_previous_format(settings: Settings) -> None:
    async with client_measuring(settings, "off") as client:
        result = await client.call_tool(
            "analyze_image", {"filename": "chair.png", "analysis": "canny", "resolution": 64}
        )

    text = result.content[0]
    assert isinstance(text, TextContent)
    assert text.text == (
        f"canny analysis of chair.png (128x64). {PROCESSORS['canny'].description} "
        "Detection resolution 64."
    )


async def test_brief_text_omits_instructions_and_carries_measurement(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PROCESSORS, "segments", fake_segments_spec())

    async with client_measuring(settings, "brief") as client:
        result = await client.call_tool(
            "analyze_image",
            {
                "filename": "chair.png",
                "analysis": "segments",
                "box": [0.2, 0.2, 0.8, 0.8],
                "resolution": 64,
            },
        )

    text = result.content[0]
    assert isinstance(text, TextContent)
    assert "at resolution 64" in text.text
    assert "Region covers" in text.text
    assert "Ask for it" not in text.text


async def test_off_keeps_the_description_and_drops_the_measurement(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off carries no numbers for any analysis, including the prompted one."""
    monkeypatch.setitem(PROCESSORS, "segments", fake_segments_spec())

    async with client_measuring(settings, "off") as client:
        result = await client.call_tool(
            "analyze_image",
            {
                "filename": "chair.png",
                "analysis": "segments",
                "box": [0.2, 0.2, 0.8, 0.8],
                "resolution": 64,
            },
        )

    text = result.content[0]
    assert isinstance(text, TextContent)
    assert "Ask for it" in text.text
    assert SAMPLE_MEASUREMENT.brief not in text.text


async def test_measurement_mode_follows_the_only_settings(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server has one source of configuration: the service it was built from."""
    monkeypatch.setitem(PROCESSORS, "segments", fake_segments_spec())
    measuring = settings.model_copy(update={"result_measurements": "full"})
    manager = ModelManager(measuring.model_dir, torch.device("cpu"), max_loaded=1)
    service = AnalysisService(measuring, manager, AnalysisCache(measuring.output_dir))

    async with Client(build_server(service)) as client:
        result = await client.call_tool(
            "analyze_image",
            {
                "filename": "chair.png",
                "analysis": "segments",
                "box": [0.2, 0.2, 0.8, 0.8],
                "resolution": 64,
            },
        )
        description = await tool_description(client, "analyze_image")

    text = result.content[0]
    assert isinstance(text, TextContent)
    assert SAMPLE_MEASUREMENT.full in text.text
    assert "measurements" in description


async def test_every_use_when_appears_in_the_tool_description(client: Client) -> None:
    description = await tool_description(client, "analyze_image")

    for kind, spec in PROCESSORS.items():
        assert spec.use_when in description, f"{kind} guidance missing from the description"


async def test_off_result_text_omits_use_when(settings: Settings) -> None:
    """Choosing guidance belongs to the tool description; results carry only the reading."""
    async with client_measuring(settings, "off") as client:
        result = await client.call_tool(
            "analyze_image", {"filename": "chair.png", "analysis": "canny", "resolution": 64}
        )

    text = result.content[0]
    assert isinstance(text, TextContent)
    assert PROCESSORS["canny"].description in text.text
    assert PROCESSORS["canny"].use_when not in text.text


async def test_description_mentions_measurements_only_when_on(settings: Settings) -> None:
    async with client_measuring(settings, "off") as off_client:
        off = await tool_description(off_client, "analyze_image")
    async with client_measuring(settings, "brief") as brief_client:
        brief = await tool_description(brief_client, "analyze_image")

    assert "measurements" not in off
    assert "measurements" in brief


async def test_sample_analysis_returns_structured_samples(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fake map holds level 50 on its left half, so x 0.25 reads 50."""
    monkeypatch.setitem(PROCESSORS, "depth", sampled_test_spec("depth"))

    result = await client.call_tool(
        "sample_analysis",
        {"filename": "chair.png", "analysis": "depth", "points": [[0.25, 0.5]]},
    )

    assert result.is_error is not True
    assert result.structured_content["samples"][0]["value"] == [50]


async def test_sample_analysis_response_carries_its_reading(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The response says what its values are, and not only the tool description."""
    monkeypatch.setitem(PROCESSORS, "depth", sampled_test_spec("depth"))

    result = await client.call_tool(
        "sample_analysis",
        {"filename": "chair.png", "analysis": "depth", "points": [[0.25, 0.5]]},
    )

    assert result.structured_content["reading"] == "Fake levels."
    assert result.structured_content["resolution"] == 64


async def test_real_sampled_analyses_say_what_their_values_are_not() -> None:
    assert "not distances" in PROCESSORS["depth"].values_description
    assert "not to the world" in PROCESSORS["normals"].values_description


async def test_sample_description_states_what_values_are_not(client: Client) -> None:
    description = await tool_description(client, "sample_analysis")

    assert "not distances" in description
    assert "relative to this camera" in description


async def test_sample_description_says_depth_steps_are_gradual(client: Client) -> None:
    description = await tool_description(client, "sample_analysis")

    assert "change gradually across a step" in description


async def test_sample_schema_says_unflagged_is_not_one_surface(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    sample = tools["sample_analysis"].output_schema["$defs"]["Sample"]
    assert "False does not show" in sample["properties"]["on_boundary"]["description"]


async def test_sample_description_warns_that_sky_reads_like_a_surface() -> None:
    """Each sampled analysis says so itself, since each gives sky a steady value."""
    for kind in SAMPLED_KINDS:
        assert "Sky and open background" in PROCESSORS[kind].values_description


async def test_sample_analysis_schema_lists_sampled_analyses(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    analysis_schema = tools["sample_analysis"].input_schema["properties"]["analysis"]
    assert analysis_schema["enum"] == ["depth", "normals"]


async def test_sample_on_canny_is_error(client: Client) -> None:
    result = await client.call_tool(
        "sample_analysis",
        {"filename": "chair.png", "analysis": "canny", "points": [[0.5, 0.5]]},
    )

    assert result.is_error is True


async def test_sample_with_malformed_point_is_error(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PROCESSORS, "depth", sampled_test_spec("depth"))

    result = await client.call_tool(
        "sample_analysis",
        {"filename": "chair.png", "analysis": "depth", "points": [[0.5, 1.5]]},
    )

    assert result.is_error is True
    assert "between 0 and 1" in result.content[0].text


async def test_compare_description_says_neither_image_is_correct(client: Client) -> None:
    description = await tool_description(client, "compare_images")

    assert "Neither image is treated as the correct one" in description
    assert "parallax" in description
    assert "not that they are the same physical edge" in description


async def test_compare_align_schema(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    schema = tools["compare_images"].input_schema
    assert schema["properties"]["align"]["enum"] == ["fit", "none"]
    assert schema["required"] == ["first", "second"]


async def test_compare_none_with_different_proportions_is_error(client: Client) -> None:
    """chair.png is 64x32 and table.jpg is 16x16."""
    result = await client.call_tool(
        "compare_images", {"first": "chair.png", "second": "table.jpg", "align": "none"}
    )

    assert result.is_error is True
    assert "proportions" in result.content[0].text


async def test_compare_description_gives_offset_units_and_the_edge_limit(client: Client) -> None:
    description = await tool_description(client, "compare_images")

    assert "as (right, down) in fractions of the first image's width and height" in description
    assert "At most 200 edges are detected in each image" in description


async def test_compare_off_still_states_the_outcome_and_the_image(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """chair.png against itself in a shared frame: paired, so the image is one frame."""
    fake_test_line_detection(monkeypatch)

    async with client_measuring(settings, "off") as client:
        result = await client.call_tool(
            "compare_images", {"first": "chair.png", "second": "chair.png", "align": "none"}
        )

    text = result.content[0]
    assert isinstance(text, TextContent)
    assert text.text == (
        "Comparison of chair.png with chair.png, align none, at resolution 64. Compared in one "
        "shared frame as asked, with no alignment fitted. The image shows the first image "
        "dimmed, its edges in cyan, the second image's edges in magenta, and a yellow line "
        "joining the two edges of each pair."
    )


async def test_compare_off_states_a_withheld_alignment(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Solid-color references have no features, so nothing is aligned."""
    fake_test_line_detection(monkeypatch)

    async with client_measuring(settings, "off") as client:
        result = await client.call_tool(
            "compare_images", {"first": "chair.png", "second": "chair.png"}
        )

    text = result.content[0]
    assert isinstance(text, TextContent)
    assert "Alignment withheld" in text.text
    assert "side by side" in text.text
    assert "Nothing in it is aligned." in text.text


async def test_compare_brief_says_a_withheld_image_is_two_panels(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_test_line_detection(monkeypatch)

    result = await client.call_tool("compare_images", {"first": "chair.png", "second": "chair.png"})

    text = result.content[0]
    assert isinstance(text, TextContent)
    assert "No edges were paired." in text.text
    assert "side by side" in text.text


async def test_compare_without_the_checkpoint_is_error(client: Client) -> None:
    result = await client.call_tool(
        "compare_images", {"first": "chair.png", "second": "chair.png"}
    )

    assert result.is_error is True
    assert PREPARE_COMMAND in result.content[0].text


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
