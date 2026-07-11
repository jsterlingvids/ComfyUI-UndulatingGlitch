"""Four-way undulating glitch compositor nodes for ComfyUI."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


CATEGORY = "image/video/undulating glitch"


def _resize(x, h, w):
    if x.ndim != 4:
        raise ValueError(f"IMAGE must be [B,H,W,C], got {tuple(x.shape)}")
    if x.shape[1:3] == (h, w):
        return x
    return F.interpolate(
        x.movedim(-1, 1), (h, w), mode="bilinear", align_corners=False
    ).movedim(1, -1)


def _align(xs, mode):
    if len(xs) != 4:
        raise ValueError("Exactly four IMAGE inputs are required")
    lengths = [int(x.shape[0]) for x in xs]
    n = min(lengths) if mode == "trim_shortest" else max(lengths)
    h, w = map(int, xs[0].shape[1:3])
    dev, dtype = xs[0].device, xs[0].dtype
    out = []
    for x in xs:
        x = _resize(x.to(device=dev, dtype=dtype), h, w)
        m = int(x.shape[0])
        if mode == "loop_shorter":
            idx = torch.arange(n, device=dev) % m
        elif mode == "stretch_shorter":
            idx = (
                torch.linspace(0, m - 1, n, device=dev).round().long()
                if n > 1
                else torch.zeros(1, dtype=torch.long, device=dev)
            )
        else:
            idx = torch.arange(n, device=dev)
        out.append(x.index_select(0, idx))
    return out, n, h, w


def _transform_image(
    image,
    output_h,
    output_w,
    offset_x=0.0,
    offset_y=0.0,
    scale=1.0,
    rotation=0.0,
    interpolation="bicubic",
    fill_mode="white",
):
    """Resize and transform an IMAGE batch into input A's coordinate space.

    Offsets describe the visible image movement in output pixels. Scale values above
    one zoom in, and positive rotation values rotate clockwise on screen.
    """
    image = _resize(image, output_h, output_w)
    batch, _, _, channels = image.shape
    device, dtype = image.device, image.dtype

    safe_scale = max(1e-4, float(scale))
    angle = math.radians(float(rotation))
    c, s = math.cos(angle) / safe_scale, math.sin(angle) / safe_scale

    # affine_grid maps output coordinates back into the input. These values are
    # therefore the inverse of the user-facing translate/scale/rotate transform.
    tx = 2.0 * float(offset_x) / max(1, output_w)
    ty = 2.0 * float(offset_y) / max(1, output_h)
    theta = torch.tensor(
        [[c, s, -(c * tx + s * ty)], [-s, c, -(-s * tx + c * ty)]],
        device=device,
        dtype=dtype,
    ).unsqueeze(0).expand(batch, -1, -1)

    nchw = image.movedim(-1, 1)
    grid = F.affine_grid(theta, nchw.shape, align_corners=False)
    padding_mode = fill_mode if fill_mode in {"border", "reflection"} else "zeros"
    transformed = F.grid_sample(
        nchw,
        grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=False,
    )

    if fill_mode == "white":
        validity = F.grid_sample(
            torch.ones((batch, 1, output_h, output_w), device=device, dtype=dtype),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        transformed = transformed + (1.0 - validity)

    return transformed.movedim(1, -1).reshape(
        batch, output_h, output_w, channels
    ).clamp(0, 1)


def _prepare_mask(mask, frames, h, w, image):
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3:
        raise ValueError(f"MASK must be [B,H,W] or [H,W], got {tuple(mask.shape)}")
    mask = mask.to(device=image.device, dtype=image.dtype)
    if mask.shape[0] != frames:
        if mask.shape[0] == 1:
            mask = mask.expand(frames, -1, -1)
        else:
            idx = torch.linspace(
                0, mask.shape[0] - 1, frames, device=image.device
            ).round().long()
            mask = mask.index_select(0, idx)
    return F.interpolate(
        mask[:, None], (h, w), mode="bilinear", align_corners=False
    ).clamp(0, 1)


def _liquid_transition_warp(
    images,
    transition_edges,
    warp_strength=12.0,
    warp_width=1.4,
    ripple_amount=3.0,
    ripple_scale=3.0,
    ripple_speed=2.0,
    rgb_split=1.5,
    interpolation="bicubic",
    color_fringe=False,
    color_fringe_strength=0.08,
):
    """Warp pixels around an animated transition-edge mask.

    The edge-mask gradient supplies a local normal. Sampling toward that normal
    creates a brief lens-like expansion/compression, while a tangent component
    makes smaller ripples crawl along the wave front.
    """
    if images.ndim != 4:
        raise ValueError(f"IMAGE must be [B,H,W,C], got {tuple(images.shape)}")
    frames, h, w, channels = map(int, images.shape)
    device, dtype = images.device, images.dtype
    mask = _prepare_mask(transition_edges, frames, h, w, images)

    width = max(0.1, float(warp_width))
    shaped = mask.pow(1.0 / width)
    smooth = F.avg_pool2d(shaped, kernel_size=5, stride=1, padding=2)
    padded = F.pad(smooth, (1, 1, 1, 1), mode="replicate")
    grad_x = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])
    grad_y = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    normal_x, normal_y = grad_x / magnitude, grad_y / magnitude
    tangent_x, tangent_y = -normal_y, normal_x

    yy = torch.linspace(-0.5, 0.5, h, device=device, dtype=dtype)[None, None, :, None]
    xx = torch.linspace(-0.5, 0.5, w, device=device, dtype=dtype)[None, None, None, :]
    frame_phase = (
        torch.arange(frames, device=device, dtype=dtype)
        / max(1, frames)
    )[:, None, None, None]
    ripple = torch.sin(
        2
        * math.pi
        * (
            yy * float(ripple_scale)
            + xx * float(ripple_scale) * 0.37
            + frame_phase * float(ripple_speed)
        )
    )

    active = shaped * (0.35 + 0.65 * mask)
    displacement_x = active * (
        normal_x * float(warp_strength)
        + tangent_x * ripple * float(ripple_amount)
    )
    displacement_y = active * (
        normal_y * float(warp_strength)
        + tangent_y * ripple * float(ripple_amount)
    )

    base_y, base_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    base_grid = torch.stack((base_x, base_y), dim=-1)[None].expand(frames, -1, -1, -1)
    offset_grid = torch.stack(
        (
            2.0 * displacement_x[:, 0] / max(1, w),
            2.0 * displacement_y[:, 0] / max(1, h),
        ),
        dim=-1,
    )
    grid = base_grid + offset_grid
    nchw = images.movedim(-1, 1)
    warped = F.grid_sample(
        nchw,
        grid,
        mode=interpolation,
        padding_mode="border",
        align_corners=True,
    )

    split = float(rgb_split)
    if split > 0 and channels >= 3:
        split_grid = torch.stack(
            (
                2.0 * normal_x[:, 0] * active[:, 0] * split / max(1, w),
                2.0 * normal_y[:, 0] * active[:, 0] * split / max(1, h),
            ),
            dim=-1,
        )
        red = F.grid_sample(
            nchw[:, 0:1],
            grid + split_grid,
            mode=interpolation,
            padding_mode="border",
            align_corners=True,
        )
        blue = F.grid_sample(
            nchw[:, 2:3],
            grid - split_grid,
            mode=interpolation,
            padding_mode="border",
            align_corners=True,
        )
        warped = warped.clone()
        warped[:, 0:1], warped[:, 2:3] = red, blue

    fringe_strength = max(0.0, float(color_fringe_strength))
    if bool(color_fringe) and fringe_strength > 0 and channels >= 3:
        # Concentrate the grade on the two slopes of the transition rather than
        # washing the whole blend. The dominant normal component gives opposite
        # signs on the incoming and outgoing sides of the wave.
        frame_peak = magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        slope = (magnitude / frame_peak).sqrt().clamp(0, 1)
        fringe = (active * slope).clamp(0, 1)
        side = torch.where(
            normal_x.abs() >= normal_y.abs(), normal_x, normal_y
        ).clamp(-1, 1)

        rgb = warped[:, :3]
        luma = (
            rgb[:, 0:1] * 0.2126
            + rgb[:, 1:2] * 0.7152
            + rgb[:, 2:3] * 0.0722
        )
        desaturate = (fringe * fringe_strength).clamp(0, 1)
        rgb = rgb * (1 - desaturate) + luma * desaturate

        contrast = 1 + fringe * fringe_strength * 0.5
        rgb = (rgb - 0.5) * contrast + 0.5

        # Positive and negative sides receive opposite, near-luma-neutral gains:
        # restrained magenta on one fringe and cyan/green on the other.
        tint_vector = torch.tensor(
            [0.90, -0.55, 0.65], device=device, dtype=dtype
        )[None, :, None, None]
        tint = side * fringe * fringe_strength * 0.45
        rgb = rgb * (1 + tint * tint_vector)

        warped = warped.clone()
        warped[:, :3] = rgb.clamp(0, 1)

    return warped.movedim(1, -1).clamp(0, 1), active[:, 0]


def _rand(shape, seed, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(
        int(seed) & 0xFFFFFFFFFFFFFFFF
    )
    return torch.rand(
        shape, generator=g, device="cpu", dtype=torch.float32
    ).to(device=device, dtype=dtype)


def _coords(h, w, angle, device, dtype):
    y = torch.linspace(-0.5, 0.5, h, device=device, dtype=dtype)
    x = torch.linspace(-0.5, 0.5, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    t = math.radians(float(angle))
    c, s = math.cos(t), math.sin(t)
    along, perp = xx * c + yy * s, -xx * s + yy * c
    extent = max(1e-6, 0.5 * (abs(c) + abs(s)))
    return (along / (2 * extent) + 0.5)[None], perp[None]


def _liquid_wave(perp, phase, wave_count, wave_amplitude, wave_speed):
    """Return a loop-friendly, domain-warped liquid displacement field.

    Several differently sized waves share the same travelling phase. The larger
    lobe makes the front surge while the higher octaves crawl along its edge.
    Integer wave speeds remain seamless over a complete effect cycle.
    """
    cycles = max(0.05, float(wave_count))
    amplitude = float(wave_amplitude)
    motion = phase * float(wave_speed)

    # Bend the coordinate before evaluating the visible waves. This domain warp
    # stops the edge from looking like several clean sine curves added together.
    flow_coord = perp + amplitude * (
        0.46
        * torch.sin(2 * math.pi * (perp * cycles * 0.48 + motion))
        + 0.22
        * torch.sin(
            2 * math.pi * (perp * cycles * 1.13 - motion * 2) + 1.37
        )
    )

    large_lobes = torch.sin(
        2 * math.pi * (flow_coord * cycles - motion)
    )
    travelling_ripples = torch.sin(
        2 * math.pi * (flow_coord * cycles * 2.07 + motion * 2) + 1.11
    )
    fine_ripples = torch.sin(
        2 * math.pi * (flow_coord * cycles * 4.31 - motion * 3) + 2.43
    )
    surges = torch.sin(
        2 * math.pi * (perp * cycles * 0.63 - motion)
    ) * torch.sin(
        2 * math.pi * (flow_coord * cycles * 1.37 + motion * 2) + 0.71
    )

    return amplitude * (
        0.54 * large_lobes
        + 0.25 * travelling_ripples
        + 0.11 * fine_ripples
        + 0.10 * surges
    )


def _field(
    frames,
    h,
    w,
    start=0,
    mode="sequential_wipe",
    cycle_frames=96,
    transition_fraction=0.78,
    angle=0.0,
    wave_count=1.5,
    wave_amplitude=0.10,
    wave_speed=1.0,
    noise_amount=0.055,
    noise_scale=2.0,
    noise_speed=1.0,
    block_size=48,
    block_jitter=0.055,
    tear_height=20,
    tear_amount=0.035,
    glitch_speed=6.0,
    edge_softness=10.0,
    band_count=1,
    seed=42,
    wave_overlap=0.08,
    device=None,
    dtype=torch.float32,
    chunk_size=8,
):
    device = device or torch.device("cpu")
    f = torch.arange(start, start + frames, device=device, dtype=dtype)
    phase = torch.remainder(f / max(4, int(cycle_frames)), 1.0)
    p = phase[:, None, None]
    along, perp = _coords(h, w, angle, device, dtype)
    if mode == "liquid_wipe":
        wave = _liquid_wave(
            perp, p, wave_count, wave_amplitude, wave_speed
        )
    else:
        wave = wave_amplitude * torch.sin(
            2 * math.pi * (perp * wave_count - p * wave_speed)
        )

    noise = torch.zeros((frames, h, w), device=device, dtype=dtype)
    pars = _rand((4, 4), seed + 11, device, dtype)
    for i in range(4):
        a = pars[i, 0] * 2 * math.pi
        freq = (0.7 + pars[i, 1] * 2.3) * noise_scale
        temporal = (1 + torch.round(pars[i, 2] * 3)) * noise_speed
        noise += (0.58**i) * torch.sin(
            2
            * math.pi
            * (
                (along - 0.5) * torch.cos(a) * freq
                + perp * torch.sin(a) * freq
                + p * temporal
            )
            + pars[i, 3] * 2 * math.pi
        )
    noise = noise / sum(0.58**i for i in range(4)) * noise_amount

    gh = max(1, math.ceil(h / max(1, block_size)))
    gw = max(1, math.ceil(w / max(1, block_size)))
    blocks = _rand((1, 1, gh, gw), seed + 101, device, dtype) * 2 - 1
    bphase = _rand((1, 1, gh, gw), seed + 202, device, dtype) * 2 * math.pi
    blocks = (
        F.interpolate(
            blocks
            * torch.sin(
                2 * math.pi * phase[:, None, None, None] * glitch_speed
                + bphase
            ),
            (h, w),
            mode="nearest",
        )[:, 0]
        * block_jitter
    )

    stripes = max(1, math.ceil(h / max(1, tear_height)))
    tears = _rand((1, 1, stripes, 1), seed + 303, device, dtype) * 2 - 1
    tphase = (
        _rand((1, 1, stripes, 1), seed + 404, device, dtype)
        * 2
        * math.pi
    )
    tears = (
        F.interpolate(
            tears
            * torch.sin(
                2
                * math.pi
                * phase[:, None, None, None]
                * glitch_speed
                * 0.73
                + tphase
            ),
            (h, w),
            mode="nearest",
        )[:, 0]
        * tear_amount
    )

    spatial = along + wave + noise + blocks + tears
    if mode == "four_way_bands":
        coord = torch.remainder(
            spatial * max(1, int(band_count)) * 4 + p * 4, 4
        )
        field = coord / 4
    elif mode == "liquid_wipe":
        # A continuously advancing staircase of source indices. A new front is
        # emitted every source stage; travel_duration > 1 means the outgoing
        # front is still leaving the right edge as the next enters from the left.
        # The half-stage offset puts those two fronts at opposite edges at the
        # cycle boundary instead of beginning on a solid frame.
        stage = phase[:, None, None] * 4
        travel_duration = 1.0 + max(0.0, float(wave_overlap))
        source_position = stage + 0.5 - spatial * travel_duration
        base = torch.floor(source_position)
        frac = source_position - base
        softness = max(
            1e-5,
            (edge_softness / max(h, w)) * travel_duration,
        )
        blend = torch.sigmoid((frac - 0.5) / softness)
        field = torch.remainder(base + blend, 4) / 4
    else:
        stage = phase * 4
        current = torch.floor(stage)
        local = stage - current
        hold = 1 - max(0.01, min(1.0, transition_fraction))
        prog = (
            (local - hold) / max(0.01, transition_fraction)
        ).clamp(0, 1)
        prog = prog * prog * (3 - 2 * prog)
        margin = min(
            0.48,
            0.06
            + abs(wave_amplitude)
            + abs(noise_amount)
            + abs(block_jitter)
            + abs(tear_amount),
        )
        threshold = -margin + prog[:, None, None] * (1 + 2 * margin)
        softness = max(1e-5, edge_softness / max(h, w))
        blend = torch.sigmoid((threshold - spatial) / softness)
        field = torch.remainder(current[:, None, None] + blend, 4) / 4
    frac = torch.remainder(field * 4, 1)
    edge = 4 * frac * (1 - frac)
    return field.clamp(0, 1), edge.clamp(0, 1)


def _preview(field):
    pal = torch.tensor(
        [[0.95, 0.22, 0.20], [0.20, 0.78, 0.30], [0.20, 0.43, 0.95], [0.95, 0.78, 0.18]],
        device=field.device,
        dtype=field.dtype,
    )
    c = field * 4
    basef = torch.floor(c)
    base = torch.remainder(basef.long(), 4)
    nxt = torch.remainder(base + 1, 4)
    a = (c - basef)[..., None]
    return pal[base] * (1 - a) + pal[nxt] * a


def _composite(xs, field, curve):
    c = field * 4
    basef = torch.floor(c)
    base = torch.remainder(basef.long(), 4)
    nxt = torch.remainder(base + 1, 4)
    a = c - basef
    if curve == "smoothstep":
        a = a * a * (3 - 2 * a)
    elif curve == "hard":
        a = (a >= 0.5).to(a.dtype)
    out = torch.zeros_like(xs[0])
    for i, x in enumerate(xs):
        weight = (base == i).to(a.dtype) * (1 - a) + (nxt == i).to(a.dtype) * a
        out += x * weight[..., None]
    return out.clamp(0, 1), (4 * a * (1 - a)).clamp(0, 1)


EFFECT_INPUTS = {
    "mode": (["sequential_wipe", "liquid_wipe", "four_way_bands"],),
    "cycle_frames": ("INT", {"default": 96, "min": 4, "max": 10000, "step": 4}),
    "transition_fraction": ("FLOAT", {"default": 0.78, "min": 0.01, "max": 1.0, "step": 0.01}),
    "angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
    "wave_count": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
    "wave_amplitude": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.75, "step": 0.01}),
    "wave_speed": ("FLOAT", {"default": 1.0, "min": -12.0, "max": 12.0, "step": 0.1}),
    "noise_amount": ("FLOAT", {"default": 0.055, "min": 0.0, "max": 0.75, "step": 0.005}),
    "noise_scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1}),
    "noise_speed": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
    "block_size": ("INT", {"default": 48, "min": 1, "max": 1024}),
    "block_jitter": ("FLOAT", {"default": 0.055, "min": 0.0, "max": 0.75, "step": 0.005}),
    "tear_height": ("INT", {"default": 20, "min": 1, "max": 512}),
    "tear_amount": ("FLOAT", {"default": 0.035, "min": 0.0, "max": 0.75, "step": 0.005}),
    "glitch_speed": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.25}),
    "edge_softness": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 256.0, "step": 0.5}),
    "band_count": ("INT", {"default": 1, "min": 1, "max": 16}),
    "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
    "wave_overlap": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
}


def _transform_inputs(prefix):
    return {
        f"{prefix}_offset_x": ("FLOAT", {"default": 0.0, "min": -4096.0, "max": 4096.0, "step": 1.0}),
        f"{prefix}_offset_y": ("FLOAT", {"default": 0.0, "min": -4096.0, "max": 4096.0, "step": 1.0}),
        f"{prefix}_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01}),
        f"{prefix}_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
    }


class FourWayQuickAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
                "images_c": ("IMAGE",),
                "images_d": ("IMAGE",),
                **_transform_inputs("b"),
                **_transform_inputs("c"),
                **_transform_inputs("d"),
                "interpolation": (["bicubic", "bilinear", "nearest"],),
                "fill_mode": (["white", "black", "border", "reflection"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned_a", "aligned_b", "aligned_c", "aligned_d")
    FUNCTION = "align"
    CATEGORY = CATEGORY

    def align(
        self,
        images_a,
        images_b,
        images_c,
        images_d,
        b_offset_x,
        b_offset_y,
        b_scale,
        b_rotation,
        c_offset_x,
        c_offset_y,
        c_scale,
        c_rotation,
        d_offset_x,
        d_offset_y,
        d_scale,
        d_rotation,
        interpolation,
        fill_mode,
    ):
        if images_a.ndim != 4:
            raise ValueError(
                f"IMAGE must be [B,H,W,C], got {tuple(images_a.shape)}"
            )
        h, w = map(int, images_a.shape[1:3])
        device, dtype = images_a.device, images_a.dtype

        def transformed(image, prefix):
            values = {
                "b": (b_offset_x, b_offset_y, b_scale, b_rotation),
                "c": (c_offset_x, c_offset_y, c_scale, c_rotation),
                "d": (d_offset_x, d_offset_y, d_scale, d_rotation),
            }[prefix]
            return _transform_image(
                image.to(device=device, dtype=dtype),
                h,
                w,
                *values,
                interpolation=interpolation,
                fill_mode=fill_mode,
            )

        return images_a, transformed(images_b, "b"), transformed(images_c, "c"), transformed(images_d, "d")


class LiquidTransitionWarp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "transition_edges": ("MASK",),
                "warp_strength": ("FLOAT", {"default": 12.0, "min": -96.0, "max": 96.0, "step": 0.5}),
                "warp_width": ("FLOAT", {"default": 1.4, "min": 0.1, "max": 6.0, "step": 0.1}),
                "ripple_amount": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 48.0, "step": 0.25}),
                "ripple_scale": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "ripple_speed": ("FLOAT", {"default": 2.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "rgb_split": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 24.0, "step": 0.25}),
                "interpolation": (["bicubic", "bilinear", "nearest"],),
                "color_fringe": ("BOOLEAN", {"default": False}),
                "color_fringe_strength": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "warp_mask")
    FUNCTION = "warp"
    CATEGORY = CATEGORY

    def warp(
        self,
        images,
        transition_edges,
        warp_strength,
        warp_width,
        ripple_amount,
        ripple_scale,
        ripple_speed,
        rgb_split,
        interpolation,
        color_fringe,
        color_fringe_strength,
    ):
        return _liquid_transition_warp(
            images,
            transition_edges,
            warp_strength=warp_strength,
            warp_width=warp_width,
            ripple_amount=ripple_amount,
            ripple_scale=ripple_scale,
            ripple_speed=ripple_speed,
            rgb_split=rgb_split,
            interpolation=interpolation,
            color_fringe=color_fringe,
            color_fringe_strength=color_fringe_strength,
        )


class UndulatingGlitchField:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reference": ("IMAGE",), **EFFECT_INPUTS}}

    RETURN_TYPES = ("MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("source_field", "field_preview", "transition_edges")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, reference, **kw):
        n, h, w = map(int, reference.shape[:3])
        field, edge = _field(
            n,
            h,
            w,
            device=reference.device,
            dtype=reference.dtype,
            **kw,
        )
        return field, _preview(field), edge


class FourWayFieldComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
                "images_c": ("IMAGE",),
                "images_d": ("IMAGE",),
                "source_field": ("MASK",),
                "frame_alignment": (["trim_shortest", "loop_shorter", "stretch_shorter"],),
                "blend_curve": (["smoothstep", "linear", "hard"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "transition_edges")
    FUNCTION = "composite"
    CATEGORY = CATEGORY

    def composite(
        self,
        images_a,
        images_b,
        images_c,
        images_d,
        source_field,
        frame_alignment,
        blend_curve,
    ):
        xs, n, h, w = _align(
            [images_a, images_b, images_c, images_d], frame_alignment
        )
        field = source_field
        if field.ndim == 2:
            field = field[None]
        field = F.interpolate(
            field[:, None].to(xs[0]),
            (h, w),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        if frame_alignment == "loop_shorter":
            idx = torch.arange(n, device=field.device) % field.shape[0]
        elif frame_alignment == "stretch_shorter":
            idx = (
                torch.linspace(0, field.shape[0] - 1, n, device=field.device)
                .round()
                .long()
            )
        else:
            idx = torch.arange(n, device=field.device)
        return _composite(
            xs, field.index_select(0, idx).clamp(0, 1), blend_curve
        )


class FourWayUndulatingGlitchMixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
                "images_c": ("IMAGE",),
                "images_d": ("IMAGE",),
                "frame_alignment": (["trim_shortest", "loop_shorter", "stretch_shorter"],),
                "blend_curve": (["smoothstep", "linear", "hard"],),
                **EFFECT_INPUTS,
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("images", "field_preview", "transition_edges")
    FUNCTION = "mix"
    CATEGORY = CATEGORY

    def mix(
        self,
        images_a,
        images_b,
        images_c,
        images_d,
        frame_alignment,
        blend_curve,
        **kw,
    ):
        xs, n, h, w = _align(
            [images_a, images_b, images_c, images_d], frame_alignment
        )
        field, _ = _field(
            n,
            h,
            w,
            device=xs[0].device,
            dtype=xs[0].dtype,
            **kw,
        )
        out, edge = _composite(xs, field, blend_curve)
        return out, _preview(field), edge


NODE_CLASS_MAPPINGS = {
    "UG_FourWayQuickAlign": FourWayQuickAlign,
    "UG_LiquidTransitionWarp": LiquidTransitionWarp,
    "UG_UndulatingGlitchField": UndulatingGlitchField,
    "UG_FourWayFieldComposite": FourWayFieldComposite,
    "UG_FourWayUndulatingGlitchMixer": FourWayUndulatingGlitchMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UG_FourWayQuickAlign": "Four-Way Quick Align",
    "UG_LiquidTransitionWarp": "Liquid Transition Warp",
    "UG_UndulatingGlitchField": "Undulating Glitch Field (4-Way)",
    "UG_FourWayFieldComposite": "Four-Way Field Composite",
    "UG_FourWayUndulatingGlitchMixer": "Four-Way Undulating Glitch Mixer",
}
