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
from controlnet_mcp.analysis import AnalysisService, ResolutionError
from controlnet_mcp.config import Settings, apply_download_policy, load_settings
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
    "Each result is an image with a short note. Combine several before committing to geometry."
)

ANALYZE_IMAGE_DESCRIPTION = (
    "Produce one visual analysis of a reference image. What each analysis shows and when to "
    "ask for it:\n"
    + "\n".join(f"- {kind}: {spec.description}" for kind, spec in PROCESSORS.items())
    + "\nAnalyses that take a box or point need one; use get_reference_image to choose it."
)

_EXPECTED_ERRORS = (
    ReferenceImageError,
    UnknownAnalysisError,
    ResolutionError,
    PromptError,
    MissingCheckpointError,
)


def build_server(settings: Settings, service: AnalysisService | None = None) -> MCPServer:
    """Create the MCP server with its three tools bound to ``settings``.

    Args:
        settings: Validated configuration.
        service: An analysis service to reuse; built from ``settings`` when omitted.
    """
    analysis_service = service if service is not None else AnalysisService.from_settings(settings)
    mcp = MCPServer(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    @mcp.tool(annotations=read_only)
    def list_reference_images() -> list[ReferenceImageInfo]:
        """List the reference images you can view and analyze.

        Returns each file's name, width, height, and format.
        """
        return images.list_reference_images(settings.reference_image_dir)

    @mcp.tool(annotations=read_only)
    def get_reference_image(
        filename: Annotated[str, Field(description="A name from list_reference_images.")],
    ) -> Image:
        """Show an original reference image.

        Look at it before analyzing, and use it to choose a box or point when
        you want a region segmented.
        """
        try:
            data, mime = images.read_reference_bytes(settings.reference_image_dir, filename)
        except ReferenceImageError as exc:
            raise ToolError(str(exc)) from exc
        return Image(data=data, format=mime.removeprefix("image/"))

    @mcp.tool(annotations=read_only, description=ANALYZE_IMAGE_DESCRIPTION)
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

        The agent-facing description is ``ANALYZE_IMAGE_DESCRIPTION``, generated
        from the processor registry so the explanations exist in one place.
        """
        try:
            prompt = None
            if box is not None or point is not None:
                prompt = RegionPrompt.from_lists(box, point)
            result = analysis_service.analyze(filename, analysis, resolution, prompt)
        except _EXPECTED_ERRORS as exc:
            raise ToolError(str(exc)) from exc
        summary = (
            f"{result.kind} analysis of {filename} ({result.width}x{result.height}). "
            f"{result.description}"
        )
        if result.note:
            summary = f"{summary} {result.note}"
        return [summary, Image(data=result.png, format="png")]

    return mcp


def main() -> None:
    """Load configuration from the environment and serve over stdio."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    apply_download_policy(settings)
    server = build_server(settings)
    logger.info("Starting %s over stdio", SERVER_NAME)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
