"""MCP server exposing reference image access and geometric image analyses over stdio.

Tools are plain synchronous functions; the SDK runs them on a worker thread so
long inference calls do not block protocol handling. Logging goes to stderr
because stdout carries the protocol stream.
"""

import logging
import sys

from mcp.server import MCPServer
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from controlnet_mcp import images
from controlnet_mcp.analysis import AnalysisService, ResolutionError
from controlnet_mcp.config import Settings, apply_download_policy, load_settings
from controlnet_mcp.images import ReferenceImageError, ReferenceImageInfo
from controlnet_mcp.model_manager import MissingCheckpointError
from controlnet_mcp.processors import AnalysisKind, UnknownAnalysisError
from controlnet_mcp.segmentation import PromptError, RegionPrompt

logger = logging.getLogger(__name__)

SERVER_NAME = "ControlNet Geometry Analysis"

SERVER_INSTRUCTIONS = (
    "Visual evidence about reference images for rebuilding an object in 3D. "
    "Start with list_reference_images, look at the original with get_reference_image, then "
    "ask analyze_image for depth, normals, lineart, lines, canny, or segments. "
    "Each result is an image with a short note. Combine several before committing to geometry."
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
    def get_reference_image(filename: str) -> Image:
        """Show an original reference image.

        Look at it before analyzing, and use it to choose a box or point when
        you want a region segmented.

        Args:
            filename: A name from list_reference_images.
        """
        try:
            data, mime = images.read_reference_bytes(settings.reference_image_dir, filename)
        except ReferenceImageError as exc:
            raise ToolError(str(exc)) from exc
        return Image(data=data, format=mime.removeprefix("image/"))

    @mcp.tool(annotations=read_only)
    def analyze_image(
        filename: str,
        analysis: AnalysisKind,
        resolution: int | None = None,
        box: list[float] | None = None,
        point: list[float] | None = None,
    ) -> list[str | Image]:
        """Produce one visual analysis of a reference image.

        What each analysis shows and when to ask for it:
        - depth: brighter is closer. Which parts sit in front, and how deep to extrude.
        - normals: color is facing direction. Flat faces versus curved surfaces.
        - lineart: clean contours, texture removed. Silhouettes and part boundaries.
        - lines: straight edges only. Axes, planar edges, perspective direction.
        - canny: every raw edge pixel. A detail check when lineart dropped something.
        - segments: the region you point at, tinted and outlined, with its extent.
          Needs a box or a point; use get_reference_image to choose one.

        Args:
            filename: A name from list_reference_images.
            analysis: Which analysis to produce.
            resolution: Working resolution for the short side of the image, 64 to 2048.
                Leave unset for the default.
            box: For segments only. [x0, y0, x1, y1] as fractions of width and height,
                origin top-left, top-left corner first.
            point: For segments only. [x, y] as fractions of width and height on the
                part to isolate. May be combined with box.
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
