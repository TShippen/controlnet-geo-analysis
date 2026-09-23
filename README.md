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
| `analyze_image(filename, analysis, resolution=None, box=None, point=None)` | One analysis image plus a one-line description. |

`analysis` is one of `depth`, `normals`, `lineart`, `lines`, `segments`, or `canny`. The names
are semantic on purpose: the backing model for any of them can change without changing the
agent-facing interface, and the text an agent sees never names a model. The current backends
are the controlnet-aux detectors Zoe, NormalBae, Lineart, MLSD, MobileSAM, and OpenCV Canny.

`segments` is prompted: the agent supplies a `box` (`[x0, y0, x1, y1]`) or a `point`
(`[x, y]`) in fractions of the image size, and gets back the chosen region tinted and outlined
on the image plus its area share and bounding box. The image is encoded once and each further
prompt on it takes a fraction of a second. A prompt on any other analysis is an error.

Filenames are confined to the reference directory. Absolute paths, parent references, symlinks
that leave the directory, and unsupported extensions are rejected. Supported formats are PNG,
JPEG, and WebP.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Edit `.env` to point at three directories. `REFERENCE_IMAGE_DIR` and `MODEL_DIR` must exist;
`OUTPUT_DIR` is created on first use. See `.env.example` for every variable. `HF_HOME` only
affects the preparation command; the running server never touches the Hugging Face cache.

Then install the checkpoints once. This is the only step that touches the network:

```bash
uv run python -m controlnet_mcp.prepare_models
```

The command downloads about 1.8 GB into `MODEL_DIR` and is safe to rerun; existing files are
skipped. `--check` reports what is missing without downloading. If a tool is called before its
checkpoint is installed, the tool returns an error naming the file and this command.

On Linux, `pyproject.toml` pins torch to the CPU wheel index to keep the install small. To use a
CUDA build on Linux, remove the `[tool.uv.sources]` entries for torch and torchvision and run
`uv sync` again. macOS and Windows already use the default PyPI wheels, so Apple MPS works
without changes. `DEVICE=auto` picks CUDA, then MPS, then CPU.

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

Canny needs no checkpoint and is never cached in memory. Prompted results are keyed by the
prompt as well, and the note about the region travels inside the PNG's text chunk.

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
