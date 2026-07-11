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
    return dict(mode="sequential_wipe", cycle_frames=12, transition_fraction=0.8,
                angle=0.0, wave_count=1.5, wave_amplitude=0.1, wave_speed=1.0,
                noise_amount=0.04, noise_scale=2.0, noise_speed=1.0,
                block_size=8, block_jitter=0.03, tear_height=4,
                tear_amount=0.02, glitch_speed=4.0, edge_softness=2.0,
                band_count=1, seed=42)


def test_field_shapes_and_range():
    reference = solid(12, 32, 48, (1, 0, 0))
    field, preview, edge = nodes.UndulatingGlitchField().generate(reference, **effect_kwargs())
    assert field.shape == (12, 32, 48)
    assert preview.shape == (12, 32, 48, 3)
    assert edge.shape == field.shape
    assert torch.isfinite(field).all()
    assert 0.0 <= float(field.min()) <= float(field.max()) <= 1.0


def test_all_in_one_mixer():
    inputs = [solid(16,24,32,(1,0,0)), solid(12,24,32,(0,1,0)),
              solid(10,24,32,(0,0,1)), solid(8,24,32,(1,1,0))]
    output, preview, edge = nodes.FourWayUndulatingGlitchMixer().mix(
        *inputs, frame_alignment="trim_shortest", blend_curve="smoothstep", **effect_kwargs())
    assert output.shape == (8, 24, 32, 3)
    assert preview.shape == output.shape
    assert edge.shape == (8, 24, 32)
    assert torch.isfinite(output).all()
