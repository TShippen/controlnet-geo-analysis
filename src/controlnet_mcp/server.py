"""MCP server exposing reference image access and geometric image analyses over stdio.

Tools are plain synchronous functions; the SDK runs them on a worker thread so
long inference calls do not block protocol handling. Logging goes to stderr
because stdout carries the protocol stream.
"""

import logging
import sys
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from controlnet_mcp import images
from controlnet_mcp.analysis import AnalysisResult, AnalysisService, ResolutionError
from controlnet_mcp.config import MeasurementSetting, apply_download_policy, load_settings
from controlnet_mcp.images import ReferenceImageError, ReferenceImageInfo
from controlnet_mcp.model_manager import MissingCheckpointError
from controlnet_mcp.processors import PROCESSORS, AnalysisKind, UnknownAnalysisError
from controlnet_mcp.segmentation import PromptError, RegionPrompt

logger = logging.getLogger(__name__)

SERVER_NAME = "ControlNet Geometry Analysis"

SERVER_INSTRUCTIONS = (
    "Visual evidence about reference images for rebuilding an object in 3D. "
    "Start with list_reference_images, look at the original with get_reference_image, then "
    "ask analyze_image for depth, normals, lineart, lines, canny, or segments. "
    "Each result is an image with a line of text. Combine several before committing to geometry."
)

_EXPECTED_ERRORS = (
    ReferenceImageError,
    UnknownAnalysisError,
    ResolutionError,
    PromptError,
    MissingCheckpointError,
)


def build_server(service: AnalysisService) -> MCPServer:
    """Create the MCP server with its three tools bound to ``service``.

    The measurement setting is read off the service, which selects the form of
    every measurement it reports, so the tool description and the result text
    describe the same service.

    Args:
        service: The analysis service backing ``analyze_image``, and the
            source of the configuration every tool reads.
    """
    measurement_mode = service.settings.result_measurements
    mcp = MCPServer(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    @mcp.tool(annotations=read_only)
    def list_reference_images() -> list[ReferenceImageInfo]:
        """List the reference images you can view and analyze.

        Returns each file's name, width, height, and format.
        """
        return images.list_reference_images(service.settings.reference_image_dir)

    @mcp.tool(annotations=read_only)
    def get_reference_image(
        filename: Annotated[str, Field(description="A name from list_reference_images.")],
    ) -> Image:
        """Show an original reference image.

        Look at it before analyzing, and use it to choose a box or point when
        you want a region segmented.
        """
        try:
            data, mime = images.read_reference_bytes(service.settings.reference_image_dir, filename)
        except ReferenceImageError as exc:
            raise ToolError(str(exc)) from exc
        return Image(data=data, format=mime.removeprefix("image/"))

    @mcp.tool(annotations=read_only, description=_analyze_image_description(measurement_mode))
    def analyze_image(
        filename: Annotated[str, Field(description="A name from list_reference_images.")],
        analysis: Annotated[AnalysisKind, Field(description="Which analysis to produce.")],
        resolution: Annotated[
            int | None,
            Field(
                description=(
                    "Working resolution for the short side of the image, 64 to 2048. "
                    "Leave unset for the default."
                )
            ),
        ] = None,
        box: Annotated[
            list[float] | None,
            Field(
                description=(
                    "Region to segment: [x0, y0, x1, y1] as fractions of width and height, "
                    "origin top-left, top-left corner first."
                )
            ),
        ] = None,
        point: Annotated[
            list[float] | None,
            Field(
                description=(
                    "Region to segment: [x, y] as fractions of width and height on the part "
                    "to isolate. May be combined with box."
                )
            ),
        ] = None,
    ) -> list[str | Image]:
        """Run one analysis and return its summary text and image.

        The agent-facing description comes from ``_analyze_image_description``,
        generated from the processor registry so the explanations exist in one
        place.
        """
        try:
            prompt = None
            if box is not None or point is not None:
                prompt = RegionPrompt.from_lists(box, point)
            result = service.analyze(filename, analysis, resolution, prompt)
        except _EXPECTED_ERRORS as exc:
            raise ToolError(str(exc)) from exc
        text = _result_text(result, filename, measurement_mode)
        return [text, Image(data=result.png, format="png")]

    return mcp


def _analyze_image_description(mode: MeasurementSetting) -> str:
    """The agent-facing description of ``analyze_image`` for one measurement setting.

    The per-analysis reading instructions live here rather than in each result,
    so a result that carries measurements does not repeat them.
    """
    description = (
        "Produce one visual analysis of a reference image. What each analysis shows and when to "
        "ask for it:\n"
        + "\n".join(f"- {kind}: {spec.description}" for kind, spec in PROCESSORS.items())
        + "\nAnalyses that take a box or point need one; use get_reference_image to choose it."
    )
    if mode == "off":
        return description
    return f"{description}\nEach result's text reports measurements taken from that output."


def _result_text(result: AnalysisResult, filename: str, mode: MeasurementSetting) -> str:
    """The text block returned beside the analysis image.

    With measurements off, the text repeats what the analysis shows and how to
    read it. Otherwise it names the output and reports what was measured from it.
    """
    if mode == "off":
        return (
            f"{result.kind} analysis of {filename} ({result.width}x{result.height}). "
            f"{result.description}"
        )
    cached = ", from cache" if result.from_cache else ""
    summary = (
        f"{result.kind} analysis of {filename} ({result.width}x{result.height}) "
        f"at resolution {result.resolution}{cached}."
    )
    if not result.measurement:
        return summary
    return f"{summary} {result.measurement}"


def main() -> None:
    """Load configuration from the environment and serve over stdio."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    apply_download_policy(settings)
    service = AnalysisService.from_settings(settings)
    server = build_server(service)
    logger.info("Starting %s over stdio", SERVER_NAME)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
