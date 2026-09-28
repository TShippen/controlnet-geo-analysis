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
from controlnet_mcp.analysis import (
    AnalysisResult,
    AnalysisService,
    ComparisonResult,
    ResolutionError,
)
from controlnet_mcp.comparison import AlignMode, ComparisonError
from controlnet_mcp.config import MeasurementSetting, apply_download_policy, load_settings
from controlnet_mcp.images import ReferenceImageError, ReferenceImageInfo
from controlnet_mcp.model_manager import MissingCheckpointError
from controlnet_mcp.processors import (
    DETECTED_EDGE_LIMIT,
    PROCESSORS,
    SAMPLED_KINDS,
    AnalysisKind,
    AnalysisOptions,
    LineLength,
    OptionError,
    SampledKind,
    UnknownAnalysisError,
)
from controlnet_mcp.regions import CropError, CropRegion
from controlnet_mcp.sampling import (
    DEFAULT_LINE_SAMPLES,
    MAX_LINE_SAMPLES,
    MAX_POINTS,
    MIN_LINE_SAMPLES,
    SampleReport,
    SamplingError,
    parse_line,
    parse_points,
)
from controlnet_mcp.segmentation import Extent, PromptError, RegionPrompt

logger = logging.getLogger(__name__)

SERVER_NAME = "ControlNet Geometry Analysis"

SERVER_INSTRUCTIONS = (
    "Visual evidence about reference images for rebuilding an object in 3D. "
    "Start with list_reference_images and look at the original with get_reference_image, then "
    "ask analyze_image for the one analysis your current step needs. A usual order: lines for "
    "axes on objects with straight parts, perspective for where those edges converge and what "
    "that says about the camera, segments to split the object into "
    "parts, lineart to trace profiles, normals to choose flat or curved surfaces for each part, "
    "depth to order parts front to back, and canny last for a missing detail. Once a depth or "
    "normals analysis shows where to look, read its values at those positions with "
    "sample_analysis. To check a render of your model against a reference, or one reference "
    "against another, pair their straight edges with compare_images. Every tool reads files "
    "from the reference image directory and none takes image data, so a render has to be saved "
    "into that directory before it can be compared; ask the user where the directory is when "
    "you do not know. No analysis gives absolute size; get one known dimension from the user "
    "or the image."
)

PAIRED_IMAGE_READING = (
    "The image shows the first image dimmed, its edges in cyan, the second image's edges in "
    "magenta, and a yellow line joining the two edges of each pair."
)
SIDE_BY_SIDE_READING = (
    "The image shows the two images side by side, each dimmed, the first with its edges in "
    "cyan and the second with its edges in magenta. Nothing in it is aligned."
)

COMPARISON_DESCRIPTION = (
    "Pair the straight edges of two reference images and report how far apart each pair lies. "
    "Neither image is treated as the correct one. Both must be files in the reference image "
    "directory, under the names list_reference_images gives; this tool takes no image data, so "
    "a render of your model has to be saved into that directory first. Each offset is how far "
    "the second image's edge lies from the first image's edge, as (right, down) in fractions "
    f"of the first image's width and height. {PAIRED_IMAGE_READING}\n"
    "- align fit: one flat transform is fitted from features the two images share, and the "
    "offsets are what remains after it. Whenever the viewpoints differ, the offsets mix real "
    "differences with the parallax of depth, and the result cannot tell them apart.\n"
    "- align none: you assert that the two images already share one frame, such as a render "
    "made from the same camera. Images of different proportions are refused.\n"
    "A pair means two edges lie close in direction and position, not that they are the same "
    "physical edge, and repeating structure can pair an edge with its neighbor. The result "
    "says when a second alignment is supported nearly as well as the first, as a second plane "
    "in the scene produces. Rows of identical parts leave few features to fit, and an "
    "alignment that is one repeat off is not always caught. When too few features "
    "match, the result says the views cannot be aligned and gives no pairs, and the image "
    "shows the two sets of edges side by side. At most "
    f"{DETECTED_EDGE_LIMIT} edges are detected in each image, so in a busy image some edges "
    "are missing, and an edge with no partner may be missing from the other image only "
    "because of that; the result says when an image reached the limit. This tool does not "
    "relate views of a 3D scene taken from different positions."
)

_EXPECTED_ERRORS = (
    ReferenceImageError,
    UnknownAnalysisError,
    ResolutionError,
    PromptError,
    CropError,
    OptionError,
    SamplingError,
    ComparisonError,
)

SERVER_FAULT_TEXT = (
    "The server failed while handling this call, and the cause is written to the server's log. "
    "Tell the user. Other images or analyses may still work."
)


def _unavailable(subject: str, exc: MissingCheckpointError) -> ToolError:
    """The agent-facing error for an analysis whose checkpoint is not installed.

    Logs the checkpoint files and preparation command carried by ``exc`` at
    ERROR, since the text returned to the agent below names neither.
    """
    logger.error("%s: required checkpoint files are not installed", subject, exc_info=exc)
    if subject == "Comparing images":
        next_step = "Tell the user that the server's files need to be prepared."
    else:
        next_step = (
            "Use another analysis, or tell the user that the server's files need to be prepared."
        )
    return ToolError(
        f"{subject} is not available on this server, because files the server needs are not "
        f"installed. No change to the call will fix this. {next_step}"
    )


def _server_fault(tool: str, exc: Exception) -> ToolError:
    """The agent-facing error for a failure the tool did not anticipate.

    Logs the exception and its traceback at ERROR, since ``SERVER_FAULT_TEXT``
    carries no detail of the cause.
    """
    logger.error("Unexpected failure in %s", tool, exc_info=exc)
    return ToolError(SERVER_FAULT_TEXT)


def build_server(service: AnalysisService) -> MCPServer:
    """Create the MCP server with its tools bound to ``service``.

    The measurement setting is read off the service, which selects the form of
    every measurement it reports, so the tool description and the result text
    describe the same service.

    Args:
        service: The analysis service backing ``analyze_image``,
            ``sample_analysis``, and ``compare_images``, and the source of the
            configuration every tool reads.
    """
    measurement_mode = service.settings.result_measurements
    mcp = MCPServer(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    @mcp.tool(annotations=read_only)
    def list_reference_images() -> list[ReferenceImageInfo]:
        """List the reference images you can view and analyze.

        Returns each file's name, width, height, and format.
        """
        try:
            return images.list_reference_images(service.settings.reference_image_dir)
        except Exception as exc:
            raise _server_fault("list_reference_images", exc) from exc

    @mcp.tool(annotations=read_only)
    def get_reference_image(
        filename: Annotated[str, Field(description="A name from list_reference_images.")],
    ) -> Image:
        """Show an original reference image.

        The image is returned as the file stores it, at the width and height
        list_reference_images gives, and is never scaled down. Look at it
        before analyzing, and use it to choose a box or point when you want a
        region segmented.
        """
        try:
            data, mime = images.read_reference_bytes(service.settings.reference_image_dir, filename)
            return Image(data=data, format=mime.removeprefix("image/"))
        except ReferenceImageError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            raise _server_fault("get_reference_image", exc) from exc

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
        exclude: Annotated[
            list[list[float]] | None,
            Field(
                description=(
                    "For segments: points [[x, y], ...] as fractions of width and height on "
                    "neighboring parts the region must leave out."
                )
            ),
        ] = None,
        extent: Annotated[
            Extent | None,
            Field(
                description=(
                    "For segments with a lone point only: largest keeps the whole object, "
                    "smallest the piece under the point, best (the default) the likeliest region."
                )
            ),
        ] = None,
        crop: Annotated[
            list[float] | None,
            Field(
                description=(
                    "Analyze only this part of the image, in more detail: [x0, y0, x1, y1] as "
                    "fractions of width and height, origin top-left, top-left corner first. "
                    "Positions in the result stay fractions of the full image. Not for segments."
                )
            ),
        ] = None,
        line_length: Annotated[
            LineLength | None,
            Field(
                description=(
                    "For lines only. long keeps straight edges at least 6% of the image's longer "
                    "side, dropping short fragments; all, the default, keeps every one."
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
            if any(value is not None for value in (box, point, exclude, extent)):
                prompt = RegionPrompt.from_lists(box, point, exclude, extent)
            region = CropRegion.from_list(crop) if crop is not None else None
            options = AnalysisOptions(line_length=line_length)
            result = service.analyze(filename, analysis, resolution, prompt, region, options)
            text = _result_text(result, filename, measurement_mode)
            return [text, Image(data=result.png, format="png")]
        except MissingCheckpointError as exc:
            raise _unavailable(f"The {analysis} analysis", exc) from exc
        except _EXPECTED_ERRORS as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            raise _server_fault("analyze_image", exc) from exc

    @mcp.tool(annotations=read_only, description=_sample_analysis_description())
    def sample_analysis(
        filename: Annotated[str, Field(description="A name from list_reference_images.")],
        analysis: Annotated[SampledKind, Field(description="Which analysis to read values from.")],
        points: Annotated[
            list[list[float]] | None,
            Field(
                description=(
                    f"Positions to read: [[x, y], ...] as fractions of the full image width and "
                    f"height, origin top-left, at most {MAX_POINTS}. Give points or line, not both."
                )
            ),
        ] = None,
        line: Annotated[
            list[float] | None,
            Field(
                description=(
                    "A line to read evenly along, both ends included: [x0, y0, x1, y1] as "
                    "fractions of the full image width and height, origin top-left."
                )
            ),
        ] = None,
        count: Annotated[
            int | None,
            Field(
                description=(
                    f"For line only: how many samples to take, {MIN_LINE_SAMPLES} to "
                    f"{MAX_LINE_SAMPLES}. Leave unset for {DEFAULT_LINE_SAMPLES}."
                )
            ),
        ] = None,
        resolution: Annotated[
            int | None,
            Field(
                description=(
                    "Working resolution of the analysis that is read, 64 to 2048. Leave unset "
                    "for the default. Use the value you gave analyze_image to read the same map."
                )
            ),
        ] = None,
        crop: Annotated[
            list[float] | None,
            Field(
                description=(
                    "Read the analysis of this part of the image: [x0, y0, x1, y1] as fractions "
                    "of width and height, origin top-left, top-left corner first. Use the crop "
                    "you gave analyze_image to read the same map. Positions stay fractions of "
                    "the full image and must lie inside the crop."
                )
            ),
        ] = None,
    ) -> SampleReport:
        """Read values off one analysis at chosen positions.

        The agent-facing description comes from ``_sample_analysis_description``,
        generated from the processor registry so each analysis explains its
        own values in one place.
        """
        try:
            positions = parse_points(points) if points is not None else None
            ends = parse_line(line) if line is not None else None
            region = CropRegion.from_list(crop) if crop is not None else None
            return service.sample(filename, analysis, positions, ends, count, resolution, region)
        except MissingCheckpointError as exc:
            raise _unavailable(f"The {analysis} analysis", exc) from exc
        except _EXPECTED_ERRORS as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            raise _server_fault("sample_analysis", exc) from exc

    @mcp.tool(annotations=read_only, description=COMPARISON_DESCRIPTION)
    def compare_images(
        first: Annotated[
            str,
            Field(
                description=(
                    "A name from list_reference_images. Positions and offsets in the result "
                    "are fractions of this image."
                )
            ),
        ],
        second: Annotated[
            str, Field(description="A name from list_reference_images, compared with the first.")
        ],
        align: Annotated[
            AlignMode,
            Field(
                description=(
                    "fit, the default, fits a transform from features the images share. none "
                    "takes the images to share one frame already."
                )
            ),
        ] = "fit",
        resolution: Annotated[
            int | None,
            Field(
                description=(
                    "Working resolution for the short side of each image, 64 to 2048. "
                    "Leave unset for the default."
                )
            ),
        ] = None,
        crop: Annotated[
            list[float] | None,
            Field(
                description=(
                    "Compare only this part of each image: [x0, y0, x1, y1] as fractions of "
                    "width and height, origin top-left, top-left corner first. The same "
                    "fractions are cut from both images."
                )
            ),
        ] = None,
    ) -> list[str | Image]:
        """Compare the straight edges of two images and return the summary text and image."""
        try:
            region = CropRegion.from_list(crop) if crop is not None else None
            result = service.compare(first, second, align, resolution, region)
            text = _comparison_text(result, first, second)
            return [text, Image(data=result.png, format="png")]
        except MissingCheckpointError as exc:
            raise _unavailable("Comparing images", exc) from exc
        except _EXPECTED_ERRORS as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            raise _server_fault("compare_images", exc) from exc

    return mcp


def _analyze_image_description(mode: MeasurementSetting) -> str:
    """The agent-facing description of ``analyze_image`` for one measurement setting.

    The per-analysis reading instructions live here rather than in each result,
    so a result that carries measurements does not repeat them. Guidance on
    when to use each analysis appears only here.
    """
    description = (
        "Produce one visual analysis of a reference image. Each analysis answers a different "
        "question, so pick the one that fits your current modeling step; running all of them "
        "rarely helps. What each shows and when to use it:\n"
        + "\n".join(
            f"- {kind}: {spec.description} {spec.use_when}" for kind, spec in PROCESSORS.items()
        )
        + "\nAnalyses that take a box or point need one; use get_reference_image to choose it."
        + "\nGive crop to see a small part in more detail. A cropped depth map is stretched "
        "again, so its gray levels do not match the full view, and depth and normals lose the "
        "surrounding context."
    )
    if mode == "off":
        return description
    return f"{description}\nEach result's text reports measurements taken from that output."


def _sample_analysis_description() -> str:
    """The agent-facing description of ``sample_analysis``.

    What the values of each analysis are, and are not, comes from the
    processor registry. The sentences on reading a sample hold for every
    sampled analysis and live here.
    """
    return (
        "Read the values of a depth or normals analysis at chosen positions, or evenly along "
        "a line. The values come from the same image analyze_image returns for the same "
        "filename, resolution, and crop, so look at that image first to choose positions. "
        "What the values are:\n"
        + "\n".join(f"- {kind}: {PROCESSORS[kind].values_description}" for kind in SAMPLED_KINDS)
        + "\nEach sample is the median of a small window and reports the spread inside that "
        "window. A large spread means the sample sits between surfaces and a nearby position "
        "would read differently, so sample again beside it. Along a line, each change brackets "
        "a boundary to within the sample spacing and does not locate it more finely. Deciding "
        "which surface a sample belongs to, and what a value means for the object, is up to "
        "you."
    )


def _comparison_text(result: ComparisonResult, first: str, second: str) -> str:
    """The text block returned beside the comparison image.

    It names the two images and the part of each that was compared when they
    were cropped, naming both images' crops when the two, each fitted to its
    own pixels, differ. That is followed by the measurement, or, with
    measurements off, by how the images were brought into one frame, or why
    they were not. Either way the text closes with how to read the image:
    the paired reading when edges were paired, since the image is then one
    frame, and the side-by-side reading otherwise, since the image is two
    panels. That reading holds in every measurement mode.
    """
    cropped = _comparison_crop_text(result.crop, result.second_crop)
    summary = (
        f"Comparison of {first} with {second}{cropped}, align {result.align}, "
        f"at resolution {result.resolution}."
    )
    reading = PAIRED_IMAGE_READING if result.paired else SIDE_BY_SIDE_READING
    if not result.measurement:
        return f"{summary} {result.outcome} {reading}"
    return f"{summary} {result.measurement} {reading}"


def _comparison_crop_text(crop: CropRegion | None, second_crop: CropRegion | None) -> str:
    """The crop clause of the comparison summary, empty when the whole images were compared.

    The crop is fitted to whole pixels in each image on its own, so the two
    fitted crops can differ even when cut from the same fractions. When they
    do, both are named, at whatever number of decimals first shows them as
    different; two crops that differ but print identically at two decimals
    would otherwise read as one repeated, meaningless number.
    """
    if crop is None:
        return ""
    if second_crop is None or second_crop == crop:
        return f", cropped to {_format_crop(crop, 2)}"
    decimals = 2
    while _format_crop(crop, decimals) == _format_crop(second_crop, decimals):
        decimals += 1
    return (
        f", cropped to {_format_crop(crop, decimals)} in the first image and "
        f"{_format_crop(second_crop, decimals)} in the second image"
    )


def _format_crop(region: CropRegion, decimals: int) -> str:
    """One crop's bounds to ``decimals`` places, with no leading comma or clause words."""
    return (
        f"x {region.x0:.{decimals}f} to {region.x1:.{decimals}f}, "
        f"y {region.y0:.{decimals}f} to {region.y1:.{decimals}f}"
    )


def _result_text(result: AnalysisResult, filename: str, mode: MeasurementSetting) -> str:
    """The text block returned beside the analysis image.

    With measurements off, the text repeats what the analysis shows and how to
    read it. Otherwise it names the output and reports what was measured from it.
    Either way it says which part of the image was analyzed when it was cropped.
    The size it gives is that of the returned image, which is rendered at the
    working resolution and so differs from the size of the reference file.
    """
    cropped = ""
    if result.crop is not None:
        region = result.crop
        cropped = (
            f", cropped to x {region.x0:.2f} to {region.x1:.2f}, "
            f"y {region.y0:.2f} to {region.y1:.2f}"
        )
    subject = (
        f"{result.kind} analysis of {filename}{cropped}, returned as a "
        f"{result.width}x{result.height} image"
    )
    if mode == "off":
        return f"{subject}. {result.description}"
    cached = ", from cache" if result.from_cache else ""
    summary = f"{subject} at resolution {result.resolution}{cached}."
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
