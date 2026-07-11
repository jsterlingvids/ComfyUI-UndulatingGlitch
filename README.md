# ComfyUI Undulating Glitch

A small, dependency-free ComfyUI node pack for aligning and mixing **four image/video batches** with a travelling, undulating, block-glitched transition field.

The node pack was designed for four similarly framed performance videos: the dominant source cycles A → B → C → D → A while an irregular wave boundary crosses the frame. It can also display all four sources simultaneously as moving bands.

## Nodes

### Four-Way Quick Align

Uses input A as the reference canvas and provides independent alignment controls for inputs B, C, and D:

- X and Y offset in pixels
- scale
- rotation in degrees
- interpolation quality
- white, black, border, or reflected edge filling

The four aligned batches are returned separately so they can feed directly into **Four-Way Undulating Glitch Mixer**. Input A is passed through unchanged. Inputs B–D are resized to A's dimensions before their transforms are applied.

For isolated heads on a white background, start with `fill_mode: white` and `interpolation: bicubic`. Adjust scale first, then X/Y position, and use rotation only for small eye-line corrections.

### Four-Way Undulating Glitch Mixer

The fastest route. Connect four `IMAGE` batches—typically four outputs from VHS Load Video—and send the result to Video Combine.

Outputs:

- `images`: final four-way composite
- `field_preview`: coloured preview showing which source controls each area
- `transition_edges`: mask around active blend boundaries

### Undulating Glitch Field (4-Way)

Generates the effect separately as a continuous `MASK`. Use this when you want to inspect, modify, blur, distort, or reuse the field.

### Four-Way Field Composite

Applies a generated source field to four `IMAGE` batches.

## Installation

### Git clone

From your ComfyUI directory:

```bash
cd custom_nodes
git clone https://github.com/jsterlingvids/ComfyUI-UndulatingGlitch.git
```

Restart ComfyUI, then search for `Undulating Glitch`.

To update later:

```bash
cd ComfyUI/custom_nodes/ComfyUI-UndulatingGlitch
git pull
```

### Manual installation

1. Download the repository ZIP and extract the folder into:

   ```text
   ComfyUI/custom_nodes/ComfyUI-UndulatingGlitch
   ```

2. Restart ComfyUI.
3. Search for `Undulating Glitch` or open:

   ```text
   image/video/undulating glitch
   ```

No extra Python packages are required; it only uses PyTorch already included with ComfyUI.

## Suggested starting settings

For a continuous wave travelling left to right through all four sources:

```text
mode: sequential_wipe
cycle_frames: 144
transition_fraction: 1.00
angle: 0
wave_count: 2.0
wave_amplitude: 0.15
wave_speed: 4
noise_amount: 0.025
noise_scale: 2
block_size: 48
block_jitter: 0.020
tear_height: 20
tear_amount: 0.015
glitch_speed: 6
edge_softness: 14
blend_curve: smoothstep
```

At 24 fps, `cycle_frames: 144` makes one complete A → B → C → D → A cycle last six seconds. With `transition_fraction: 1.00`, each new boundary begins as soon as the previous one exits.

For harder digital blocks, increase `block_jitter`, lower `edge_softness`, and use `hard` blending. For a flowing liquid boundary, reduce `block_jitter` and `tear_amount`, then increase `wave_amplitude` and `edge_softness`.

## Input preparation

The four videos should ideally share the same:

- frame rate
- dimensions
- duration/frame count
- framing and subject scale

Use **Four-Way Quick Align** when composition or head placement differs slightly. The mixer's `frame_alignment` setting can then trim, loop, or stretch shorter batches.

## Publishing

The repository is already structured like a normal ComfyUI custom node. Before Registry publishing:

1. Create a Publisher and API key on the Comfy Registry.
2. Replace `YOUR_COMFY_PUBLISHER_ID` in `pyproject.toml`.
3. Install `comfy-cli`, then run:

   ```bash
   comfy node publish
   ```

A release should include a short demo GIF/video and a downloadable example workflow. The demo is the sales pitch; the node name is merely the paperwork.

## Status

Prototype `0.2.0`. The core effect, tensor handling, and quick alignment transforms are implemented and covered by basic tests, but the pack still benefits from real-world testing inside your specific ComfyUI installation and VHS workflow.
