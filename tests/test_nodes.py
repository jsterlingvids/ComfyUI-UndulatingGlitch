import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
spec = importlib.util.spec_from_file_location("undulating_glitch_nodes", MODULE_PATH)
nodes = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(nodes)


def solid(frames, height, width, rgb):
    image = torch.zeros((frames, height, width, 3), dtype=torch.float32)
    image[..., 0], image[..., 1], image[..., 2] = rgb
    return image


def effect_kwargs():
    return dict(
        mode="sequential_wipe",
        cycle_frames=12,
        transition_fraction=0.8,
        angle=0.0,
        wave_count=1.5,
        wave_amplitude=0.1,
        wave_speed=1.0,
        noise_amount=0.04,
        noise_scale=2.0,
        noise_speed=1.0,
        block_size=8,
        block_jitter=0.03,
        tear_height=4,
        tear_amount=0.02,
        glitch_speed=4.0,
        edge_softness=2.0,
        band_count=1,
        seed=42,
        wave_overlap=0.08,
    )


def align_kwargs():
    values = {}
    for prefix in ("b", "c", "d"):
        values.update(
            {
                f"{prefix}_offset_x": 0.0,
                f"{prefix}_offset_y": 0.0,
                f"{prefix}_scale": 1.0,
                f"{prefix}_rotation": 0.0,
            }
        )
    values.update(interpolation="bicubic", fill_mode="white")
    return values


def test_field_shapes_and_range():
    reference = solid(12, 32, 48, (1, 0, 0))
    field, preview, edge = nodes.UndulatingGlitchField().generate(
        reference, **effect_kwargs()
    )
    assert field.shape == (12, 32, 48)
    assert preview.shape == (12, 32, 48, 3)
    assert edge.shape == field.shape
    assert torch.isfinite(field).all()
    assert 0.0 <= float(field.min()) <= float(field.max()) <= 1.0


def test_liquid_wipe_is_animated_and_loop_compatible():
    kwargs = effect_kwargs()
    kwargs.update(
        mode="liquid_wipe",
        cycle_frames=16,
        transition_fraction=1.0,
        wave_count=2.4,
        wave_amplitude=0.18,
        wave_speed=4.0,
        noise_amount=0.0,
        block_jitter=0.0,
        tear_amount=0.0,
    )
    field, edge = nodes._field(17, 32, 48, **kwargs)

    assert field.shape == (17, 32, 48)
    assert torch.isfinite(field).all()
    assert not torch.allclose(field[0], field[1])
    assert torch.allclose(field[0], field[16], atol=2e-5)
    assert torch.allclose(edge[0], edge[16], atol=2e-5)


def test_liquid_wipe_keeps_a_transition_front_on_screen():
    kwargs = effect_kwargs()
    kwargs.update(
        mode="liquid_wipe",
        cycle_frames=64,
        transition_fraction=0.25,
        wave_count=2.4,
        wave_amplitude=0.18,
        wave_speed=5.0,
        edge_softness=1.0,
        wave_overlap=0.08,
    )
    field, edge = nodes._field(64, 64, 64, **kwargs)
    active_edge_per_frame = edge.amax(dim=(1, 2))

    assert field.shape == (64, 64, 64)
    assert torch.all(active_edge_per_frame > 0.5)

    # transition_fraction intentionally belongs to sequential_wipe only. Liquid
    # timing is controlled by wave_overlap and must not reintroduce solid holds.
    kwargs["transition_fraction"] = 1.0
    field_at_one, edge_at_one = nodes._field(64, 64, 64, **kwargs)
    assert torch.allclose(field, field_at_one)
    assert torch.allclose(edge, edge_at_one)


def test_liquid_wave_contains_more_edge_detail_than_single_sine():
    perp = torch.linspace(-0.5, 0.5, 128)[None, None, :]
    phase = torch.tensor([0.23])[:, None, None]
    liquid = nodes._liquid_wave(perp, phase, 2.4, 0.18, 4.0)
    simple = 0.18 * torch.sin(2 * torch.pi * (perp * 2.4 - phase * 4.0))

    liquid_curvature = torch.diff(liquid, n=2, dim=-1).abs().mean()
    simple_curvature = torch.diff(simple, n=2, dim=-1).abs().mean()
    assert liquid_curvature > simple_curvature


def test_all_in_one_mixer():
    inputs = [
        solid(16, 24, 32, (1, 0, 0)),
        solid(12, 24, 32, (0, 1, 0)),
        solid(10, 24, 32, (0, 0, 1)),
        solid(8, 24, 32, (1, 1, 0)),
    ]
    output, preview, edge = nodes.FourWayUndulatingGlitchMixer().mix(
        *inputs,
        frame_alignment="trim_shortest",
        blend_curve="smoothstep",
        **effect_kwargs(),
    )
    assert output.shape == (8, 24, 32, 3)
    assert preview.shape == output.shape
    assert edge.shape == (8, 24, 32)
    assert torch.isfinite(output).all()


def test_quick_align_identity_and_reference_dimensions():
    a = torch.rand((3, 24, 32, 3))
    b = torch.rand((3, 24, 32, 3))
    c = torch.rand((2, 12, 16, 3))
    d = torch.rand((1, 24, 32, 3))

    aligned_a, aligned_b, aligned_c, aligned_d = nodes.FourWayQuickAlign().align(
        a, b, c, d, **align_kwargs()
    )

    assert aligned_a is a
    assert torch.allclose(aligned_b, b, atol=2e-6)
    assert aligned_c.shape == (2, 24, 32, 3)
    assert aligned_d.shape == (1, 24, 32, 3)
    assert torch.isfinite(aligned_c).all()


def test_quick_align_positive_x_moves_image_right_and_fills_white():
    a = torch.ones((1, 7, 9, 3))
    marker = torch.ones_like(a)
    marker[:, 3, 2] = 0.0
    kwargs = align_kwargs()
    kwargs.update(b_offset_x=2.0, interpolation="nearest")

    _, aligned_b, _, _ = nodes.FourWayQuickAlign().align(
        a, marker, a, a, **kwargs
    )

    assert torch.equal(aligned_b[:, 3, 4], torch.zeros((1, 3)))
    assert torch.equal(aligned_b[:, 3, 2], torch.ones((1, 3)))
    assert torch.equal(aligned_b[:, :, :2], torch.ones((1, 7, 2, 3)))


def test_quick_align_scale_enlarges_marker():
    image = torch.ones((1, 9, 9, 3))
    image[:, 3:6, 3:6] = 0.0
    kwargs = align_kwargs()
    kwargs.update(b_scale=2.0, interpolation="nearest")

    _, aligned_b, _, _ = nodes.FourWayQuickAlign().align(
        image, image, image, image, **kwargs
    )

    source_dark = int((image[0, ..., 0] == 0).sum())
    aligned_dark = int((aligned_b[0, ..., 0] == 0).sum())
    assert aligned_dark > source_dark


def test_quick_align_registered_in_comfyui_mappings():
    assert nodes.NODE_CLASS_MAPPINGS["UG_FourWayQuickAlign"] is nodes.FourWayQuickAlign
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["UG_FourWayQuickAlign"] == "Four-Way Quick Align"


def transition_fixture(frames=4, height=32, width=48):
    x = torch.linspace(0, 1, width)[None, None, :, None]
    y = torch.linspace(0, 1, height)[None, :, None, None]
    image = (0.7 * x + 0.3 * y).expand(frames, height, width, 3).clone()
    edge_x = torch.linspace(-1, 1, width)[None, None, :]
    edge = torch.exp(-((edge_x / 0.16) ** 2)).expand(frames, height, width).clone()
    return image, edge


def test_liquid_transition_warp_zero_strength_is_identity():
    image, edge = transition_fixture()
    output, warp_mask = nodes._liquid_transition_warp(
        image,
        edge,
        warp_strength=0.0,
        ripple_amount=0.0,
        rgb_split=0.0,
        interpolation="nearest",
    )

    assert output.shape == image.shape
    assert warp_mask.shape == edge.shape
    assert torch.equal(output, image)


def test_liquid_transition_warp_is_localized_to_edge():
    image, edge = transition_fixture()
    output, warp_mask = nodes._liquid_transition_warp(
        image,
        edge,
        warp_strength=10.0,
        warp_width=1.4,
        ripple_amount=3.0,
        ripple_scale=3.0,
        ripple_speed=2.0,
        rgb_split=0.0,
        interpolation="bilinear",
    )

    difference = (output - image).abs().mean(dim=-1)
    assert float(difference[:, :, 18:30].max()) > 1e-3
    assert float(difference[:, :, :4].max()) < 1e-4
    assert float(difference[:, :, -4:].max()) < 1e-4
    assert float(warp_mask[:, :, 18:30].max()) > 0.5


def test_liquid_transition_warp_rgb_split_separates_channels():
    image, edge = transition_fixture()
    output, _ = nodes._liquid_transition_warp(
        image,
        edge,
        warp_strength=0.0,
        ripple_amount=0.0,
        rgb_split=4.0,
        interpolation="bilinear",
    )

    channel_difference = (output[..., 0] - output[..., 2]).abs()
    assert float(channel_difference[:, :, 18:30].max()) > 1e-3


def test_liquid_transition_warp_registered_in_comfyui_mappings():
    assert nodes.NODE_CLASS_MAPPINGS["UG_LiquidTransitionWarp"] is nodes.LiquidTransitionWarp
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["UG_LiquidTransitionWarp"] == "Liquid Transition Warp"
