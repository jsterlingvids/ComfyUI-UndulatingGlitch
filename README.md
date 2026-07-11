# ComfyUI Undulating Glitch

A small, dependency-free ComfyUI node pack for mixing **four image/video batches** with a travelling, undulating, block-glitched transition field.

The first version was designed for four similarly framed performance videos: the dominant source cycles A → B → C → D → A while an irregular wave boundary crosses the frame. It can also display all four sources simultaneously as moving bands.

## Nodes

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

For a single wave travelling left to right through all four sources:

```text
mode: sequential_wipe
cycle_frames: 96
transition_fraction: 0.78
angle: 0
wave_count: 1.5
wave_amplitude: 0.10
wave_speed: 1
noise_amount: 0.055
noise_scale: 2
block_size: 48
block_jitter: 0.055
tear_height: 20
tear_amount: 0.035
glitch_speed: 6
edge_softness: 10
blend_curve: smoothstep
```

At 24 fps, `cycle_frames: 96` makes one complete A → B → C → D → A cycle last four seconds.

For harder digital blocks, increase `block_jitter`, lower `edge_softness`, and use `hard` blending. For a flowing liquid boundary, reduce `block_jitter` and `tear_amount`, then increase `wave_amplitude` and `edge_softness`.

## Input preparation

The four videos should ideally share the same:

- frame rate
- dimensions
- duration/frame count
- framing and subject scale

`frame_alignment` can trim, loop, or stretch shorter batches. Inputs B–D are resized to input A's dimensions.

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

Prototype `0.1.0`. The core effect and tensor handling are implemented and covered by basic tests, but it still needs real-world testing inside your specific ComfyUI installation and VHS workflow.
