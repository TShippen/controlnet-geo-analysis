# ControlNet Geometry Analysis MCP

A local MCP server that gives an agent geometric evidence about reference images: depth,
surface normals, structural line art, straight-line structure, region segmentation, and Canny
edges. It is meant to sit beside a Rhino/Grasshopper MCP so the agent can reason about an
object's geometry before constructing it.

The server runs over stdio, reads images from one configured directory, loads model checkpoints
from local disk on demand, keeps a bounded number of models in memory, and caches every generated
analysis so repeated requests do not rerun inference.

## Tools

| Tool | Purpose |
| --- | --- |
| `list_reference_images` | Names, dimensions, and formats of the images in the reference directory. |
| `get_reference_image(filename)` | The original image as MCP image content. |
| `analyze_image(filename, analysis, resolution=None, box=None, point=None, exclude=None, extent=None, crop=None, line_length=None)` | One analysis image plus a one-line description. |
| `sample_analysis(filename, analysis, points=None, line=None, count=None, resolution=None, crop=None)` | The values of a depth or normals analysis at chosen positions, as structured output. |
| `compare_images(first, second, align="fit", resolution=None, crop=None)` | The straight edges of two images paired, with how far apart each pair lies, as text and an image. |

`analysis` is one of `depth`, `normals`, `lineart`, `lines`, `perspective`, `segments`, or
`canny`. The names are semantic on purpose: the backing model for any of them can change without
changing the agent-facing interface, and the text an agent sees never names a model. The current
backends are the controlnet-aux detectors Zoe, NormalBae, Lineart, MLSD, MobileSAM, and OpenCV
Canny. `perspective` runs the same detector as `lines`, and the two share one loaded model.

`segments` is prompted: the agent supplies a `box` (`[x0, y0, x1, y1]`) or a `point`
(`[x, y]`) in fractions of the image size, and gets back the chosen region tinted and outlined
on the image plus its area share and bounding box. The image is encoded once and each further
prompt on it takes a fraction of a second. A prompt on any other analysis is an error.
`exclude` adds points on neighboring parts the region must leave out. A lone point is ambiguous
between a whole object and its parts, so only that prompt produces several candidate masks, and
`extent` picks among them: `best` by predicted quality, or `largest` or `smallest` by area. Any
other prompt produces a single mask, and `extent` is rejected with it.

Every other analysis takes an optional `crop` (`[x0, y0, x1, y1]`, same coordinates as `box`) and
runs on that part of the image alone, so the detection resolution goes to the part instead of the
whole scene. Positions reported for a cropped result are still fractions of the full image, and
each crop is cached separately. A crop on `segments` is an error.

`lines` takes `line_length`: `all`, the default, keeps every detected segment, and `long` keeps
only those at least 6% of the image's longer side, which leaves the main edges for finding
axes and perspective. Any other analysis rejects it.

`perspective` groups the straight edges by the direction they run in the scene. The edges of a
group meet at one vanishing point, which may lie outside the image, or run parallel when that
point is too far away to tell from infinity. The image draws each group in its own color, and the
text gives each group's vanishing point or parallel direction with its edge count. In the full
form it also gives how tightly each group fits and which edges share one line. A group is taken
for the verticals of the scene when it runs within 10 degrees of the image vertical and either
runs parallel or has its vanishing point outside the image; a group whose vanishing point lies
inside the image recedes into the scene and is never taken for the verticals, however close its
direction runs to vertical. When a group is taken for the verticals, the analysis places the
horizon: through the vanishing points of two other converging groups when there are that many,
or, with only one, through that point and tilted by the verticals, perpendicular to their own
direction or, when the verticals converge, to the line from the image center to their vanishing
point; that second case is withheld for a crop, since the optical center of a crop is unknown.
When at least two groups converge and their vanishing points are consistent with perpendicular
directions, it estimates the field of view and the tilt of the camera. Each of
those comes with the assumption behind it. When the evidence does not support one, the text says
it is withheld and why: a cropped image has no known optical center, for example, so a crop never
gets a camera estimate. The grouping is loose by design: an edge joins a group when it points
within 3 degrees of that group's vanishing point, and edges count as sharing a line when they lie
within 2 pixels of each other's line, so a group can hold a few edges from another direction and
closely spaced parallel edges can be chained together. The tool description says so. The response
names the groups behind the horizon and the camera estimate and the group taken for the
verticals, so each can be checked against the image. It gives the distance of the farthest
vanishing point a camera estimate used, counts the edges that fit no group apart from the edges
too short to have a direction, and says when the image reached the 200 edges the detector returns
at most.

`sample_analysis` reads numbers off a `depth` or `normals` map. The agent gives either `points`
or a `line` with an optional `count` of samples, all in fractions of the full image. The values
are decoded from the same rendered map `analyze_image` returns for that filename, resolution, and
crop, so a cached render is reused and the numbers match the image the agent saw. Each sample is
the median of a small window around its position and comes with the spread inside that window. A
large spread flags the sample as sitting on a boundary between surfaces. Along a line, the result
also lists each place the value changes between two consecutive samples, which brackets a
boundary to within the sample spacing. A depth map changes gradually across a step between
surfaces, so the boundary flag catches only sharp steps there, and a line across the step is the
way to find it. Depth values are levels that order surfaces within one
render; they are not distances. Normal values are directions relative to the camera, so the same
face reads differently from another viewpoint. Sky and open background get values in both maps,
usually steady ones, so a steady reading does not mean a surface is there. The report carries
that reading with the numbers, along with the resolution, size, and crop of the map it read and,
for a line, the spacing between samples. Any other analysis is rejected.

`compare_images` detects the straight edges of two images, brings the second image into the
frame of the first, and pairs edges that lie close in direction and position. Each pair reports
the offset between its edges in fractions of the first image, and neither image is treated as the
correct one. An edge with no partner is unmatched, unless it lies on the line of a paired edge
with at least half of its length along a stretch the other image's edge of that pair covers, in
which case it is a piece of that matched edge instead. With `align` set to `fit`, the default,
one flat transform is fitted from features the two images share. The result states how many
features support it and how much of the image they cover, and flags the alignment as ambiguous
when a second transform is supported nearly as well. When too few features match, the alignment
is withheld with the reason, no pairs are reported, and the image shows the two sets of edges
side by side. With `align` set to `none` the caller asserts that the images already share one
frame, as a render from a matching camera does, and images of different proportions are refused.
The offsets left after a fitted transform mix real differences with the parallax of depth
whenever the viewpoints differ, so the tool does not relate views of a scene taken from different
positions. The response repeats how to read the offsets and what a pair is, and it says when an
image reached the 200 edges the detector returns at most, since an edge with no partner may then
be missing only from the other image's detection. How the images were aligned, or why they were
not, is stated even when `RESULT_MEASUREMENTS` is `off`. Comparisons are not cached.

Every analysis measures its own output. A short and a long form of that measurement are stored
with the cached image, and `RESULT_MEASUREMENTS` in `.env` decides which one the agent sees.
`off` sends no numbers from `analyze_image` or `compare_images`, segments included: the result
text repeats what the analysis shows and how to read it.
`brief`, the default, replaces that with the measured result and what it means. `full` extends it
with whatever else that analysis measured, such as a bounding box or a centroid. The setting is
read at startup and no tool argument exposes it, so both response styles can be compared over the
same cache. `sample_analysis` returns its numbers whatever the setting, since reading numbers is
all it does.

Filenames are confined to the reference directory. Absolute paths, parent references, symlinks
that leave the directory, and unsupported extensions are rejected. Supported formats are PNG,
JPEG, and WebP.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

The example `.env` points every directory at the `data/` folder in the repository, whose
subfolders exist in a fresh clone and whose contents git ignores, so it works without edits. Put
reference images in `data/references`. A directory variable may be relative or absolute: a
relative path is taken relative to the directory of the `.env` file, whatever directory the server
is launched from, and an absolute path (or `~/...`) keeps that data anywhere else on the machine.
A variable set in the MCP host's env block takes precedence over `.env`, so one machine can
override a single directory without editing the file. `REFERENCE_IMAGE_DIR` and `MODEL_DIR` must
exist; `OUTPUT_DIR` is created on first use. See `.env.example` for every variable. `HF_HOME` only
affects the preparation command; the running server never touches the Hugging Face cache.

Then install the checkpoints once. This is the only step that touches the network:

```bash
uv run python -m controlnet_mcp.prepare_models
```

The command downloads about 1.8 GB into `MODEL_DIR` and is safe to rerun; existing files are
skipped. `--check` reports what is missing without downloading. If a tool is called before its
checkpoint is installed, the tool returns an error naming the file and this command.

On Linux, the CPU build of torch is the default, so a plain `uv sync` and `uv run` stay small. A
CUDA machine sets `UV_NO_GROUP=cpu`, either in its shell or in the MCP host config's env block, and
`uv sync` and `uv run` then install the CUDA build from PyPI instead. The variable is the switch,
so running without it goes back to the CPU build. macOS and Windows use PyPI wheels either way, so
Apple MPS needs no configuration. `DEVICE=auto` picks CUDA, then MPS, then CPU.

```bash
export UV_NO_GROUP=cpu
uv sync
```

## Running

```bash
uv run controlnet-mcp
```

The server speaks MCP over stdin and stdout and logs to stderr. Register it with any stdio-capable
MCP host. A generic host entry looks like:

```json
{
  "mcpServers": {
    "geometry-analysis": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/controlnet-geo-analysis", "controlnet-mcp"]
    }
  }
}
```

To inspect the tools interactively:

```bash
npx @modelcontextprotocol/inspector uv run --directory /absolute/path/to/controlnet-geo-analysis controlnet-mcp
```

## How a request flows

1. The filename is resolved inside `REFERENCE_IMAGE_DIR` (`images.py`).
2. The image bytes are hashed and the cache under `OUTPUT_DIR` is checked (`cache.py`).
3. On a miss, the model manager loads the processor's checkpoint from `MODEL_DIR`, evicting the
   least recently used model when `MAX_LOADED_MODELS` is reached (`model_manager.py`).
4. The processor runs at the requested detection resolution; dimensions are rounded to multiples
   of 64 while keeping the aspect ratio (`processors.py`).
5. The PNG result is stored in the cache and returned as image content (`analysis.py`,
   `server.py`).

Canny needs no checkpoint and is never cached in memory. Cache keys include each processor's
render version, so bumping a version in the registry retires that analysis's old results
without touching the others. Prompted results are keyed by the prompt as well, and both forms of
the measurement travel inside the PNG's text chunks, so changing `RESULT_MEASUREMENTS` changes
what is reported and never invalidates the cache.

## Development

```bash
uv run pytest -m "not slow and not integration"   # fast suite, no models needed
uv run pytest -m "slow or integration"            # runs every real processor on CPU
uv run ruff check src tests
uv run mypy src
```

The slow tests skip themselves when checkpoints are absent from the `MODEL_DIR` named in `.env`.

## Out of scope for this version

Alternative depth and normal estimators, pose detection, vectorized output, camera position and
scale, and multi-view reconstruction. The camera estimate that `perspective` gives comes from the
vanishing points of a single image. The semantic tool interface is designed so those can be
swapped in later without changing how an agent calls the server.
