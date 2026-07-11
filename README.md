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

The mixer includes three movement modes:

- `sequential_wipe`: a clean undulating transition front
- `liquid_wipe`: a continuously reshaping, domain-warped liquid front
- `four_way_bands`: all four sources moving through the frame as bands

### Undulating Glitch Field (4-Way)

Generates the effect separately as a continuous `MASK`. Use this when you want to inspect, modify, blur, distort, or reuse the field.

### Four-Way Field Composite

Applies a generated source field to four `IMAGE` batches.

### Liquid Transition Warp

Adds a localized refractive glitch to the mixer's active transition boundary. Connect both `images` and `transition_edges` from **Four-Way Undulating Glitch Mixer**:

```text
Mixer images ──────────────┐
                           ├→ Liquid Transition Warp → Video Combine
Mixer transition_edges ────┘
```

The warp briefly expands and compresses pixels around the moving boundary, adds ripples travelling along the edge, and can split the red and blue channels for a stronger optical glitch. Areas away from the transition remain unchanged.

Suggested starting settings:

```text
warp_strength: 12
warp_width: 1.4
ripple_amount: 3
ripple_scale: 3
ripple_speed: 2
rgb_split: 1.5
interpolation: bicubic
color_fringe: off
color_fringe_strength: 0.08
```

Use a negative `warp_strength` to reverse the lens direction. Reduce `rgb_split` to zero for a purely liquid/refraction effect.

Enable `color_fringe` for a restrained optical grade on the two slopes of the active wave: opposing cyan/magenta bias, slight desaturation, and a small contrast lift. This is separate from `rgb_split`, which physically offsets the channels. Start with `color_fringe_strength: 0.08`; values above about `0.15` become intentionally stylized rather than subtle.

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

For a liquid-glitch wave travelling left to right through all four sources:

```text
mode: liquid_wipe
cycle_frames: 128
angle: 0
wave_count: 2.4
wave_amplitude: 0.18
wave_speed: 5
noise_amount: 0.035
noise_scale: 3
noise_speed: 2
block_size: 48
block_jitter: 0.018
tear_height: 20
tear_amount: 0.012
glitch_speed: 7
edge_softness: 18
wave_overlap: 0.08
blend_curve: smoothstep
```

At 24 fps, `cycle_frames: 128` makes one complete A → B → C → D → A cycle last about 5.3 seconds. `wave_overlap: 0.08` starts the next boundary just before the previous one finishes leaving the frame, keeping the motion nearly constant without a hard reset.

In `liquid_wipe`, `wave_amplitude` controls the overall depth of the advancing lobes, `wave_count` controls their density, and `wave_speed` controls how actively the large and small ripples crawl along the transition edge. Whole-number wave speeds produce a seamless complete effect cycle.

`transition_fraction` is used only by `sequential_wipe`. Liquid mode uses a continuous multi-front timing model instead: `wave_overlap: 0.00` places the outgoing and incoming fronts at opposite edges simultaneously, while values around `0.05–0.12` create a small, smooth overlap. Higher values put more than one transition visibly inside the frame.

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

Prototype `0.6.0`. The core effect, continuous liquid timing, transition-warp and color-fringe models, tensor handling, and quick alignment transforms are implemented and covered by tests, but the pack still benefits from real-world testing inside your specific ComfyUI installation and VHS workflow.
