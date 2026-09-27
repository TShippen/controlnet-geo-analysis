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

`analysis` is one of `depth`, `normals`, `lineart`, `lines`, `segments`, or `canny`. The names
are semantic on purpose: the backing model for any of them can change without changing the
agent-facing interface, and the text an agent sees never names a model. The current backends
are the controlnet-aux detectors Zoe, NormalBae, Lineart, MLSD, MobileSAM, and OpenCV Canny.

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

`sample_analysis` reads numbers off a `depth` or `normals` map. The agent gives either `points`
or a `line` with an optional `count` of samples, all in fractions of the full image. The values
are decoded from the same rendered map `analyze_image` returns for that filename, resolution, and
crop, so a cached render is reused and the numbers match the image the agent saw. Each sample is
the median of a small window around its position and comes with the spread inside that window. A
large spread flags the sample as sitting on a boundary between surfaces. Along a line, the result
also lists each place the value changes between two consecutive samples, which brackets a
boundary to within the sample spacing. Depth values are levels that order surfaces within one
render; they are not distances. Normal values are directions relative to the camera, so the same
face reads differently from another viewpoint. Sky and open background get values in both maps,
usually steady ones, so a steady reading does not mean a surface is there. Any other analysis is
rejected.

Every analysis measures its own output. A short and a long form of that measurement are stored
with the cached image, and `RESULT_MEASUREMENTS` in `.env` decides which one the agent sees.
`off` sends no numbers for any analysis, segments included: the result text repeats what the
analysis shows and how to read it.
`brief`, the default, replaces that with one measured sentence, since the tool description
already says how to read each analysis. `full` extends the sentence with whatever else that
analysis measured, such as a bounding box or a centroid. The setting is read at startup and no
tool argument exposes it, so both response styles can be compared over the same cache.

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

Alternative depth and normal estimators, pose detection, vectorized output, camera calibration,
and multi-view reconstruction. The semantic tool interface is designed so those can be swapped
in later without changing how an agent calls the server.
