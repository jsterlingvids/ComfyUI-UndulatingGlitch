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
    return F.interpolate(x.movedim(-1, 1), (h, w), mode="bilinear", align_corners=False).movedim(1, -1)


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
            idx = torch.linspace(0, m - 1, n, device=dev).round().long() if n > 1 else torch.zeros(1, dtype=torch.long, device=dev)
        else:
            idx = torch.arange(n, device=dev)
        out.append(x.index_select(0, idx))
    return out, n, h, w


def _rand(shape, seed, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return torch.rand(shape, generator=g, device="cpu", dtype=torch.float32).to(device=device, dtype=dtype)


def _coords(h, w, angle, device, dtype):
    y = torch.linspace(-0.5, 0.5, h, device=device, dtype=dtype)
    x = torch.linspace(-0.5, 0.5, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    t = math.radians(float(angle)); c, s = math.cos(t), math.sin(t)
    along, perp = xx * c + yy * s, -xx * s + yy * c
    extent = max(1e-6, 0.5 * (abs(c) + abs(s)))
    return (along / (2 * extent) + 0.5)[None], perp[None]


def _field(frames, h, w, start=0, mode="sequential_wipe", cycle_frames=96,
           transition_fraction=0.78, angle=0.0, wave_count=1.5, wave_amplitude=0.10,
           wave_speed=1.0, noise_amount=0.055, noise_scale=2.0, noise_speed=1.0,
           block_size=48, block_jitter=0.055, tear_height=20, tear_amount=0.035,
           glitch_speed=6.0, edge_softness=10.0, band_count=1, seed=42,
           device=None, dtype=torch.float32, chunk_size=8):
    device = device or torch.device("cpu")
    f = torch.arange(start, start + frames, device=device, dtype=dtype)
    phase = torch.remainder(f / max(4, int(cycle_frames)), 1.0)
    p = phase[:, None, None]
    along, perp = _coords(h, w, angle, device, dtype)
    wave = wave_amplitude * torch.sin(2 * math.pi * (perp * wave_count - p * wave_speed))

    noise = torch.zeros((frames, h, w), device=device, dtype=dtype)
    pars = _rand((4, 4), seed + 11, device, dtype)
    for i in range(4):
        a = pars[i, 0] * 2 * math.pi
        freq = (0.7 + pars[i, 1] * 2.3) * noise_scale
        temporal = (1 + torch.round(pars[i, 2] * 3)) * noise_speed
        noise += (0.58 ** i) * torch.sin(2 * math.pi * ((along - .5) * torch.cos(a) * freq + perp * torch.sin(a) * freq + p * temporal) + pars[i, 3] * 2 * math.pi)
    noise = noise / sum(0.58 ** i for i in range(4)) * noise_amount

    gh, gw = max(1, math.ceil(h / max(1, block_size))), max(1, math.ceil(w / max(1, block_size)))
    blocks = (_rand((1, 1, gh, gw), seed + 101, device, dtype) * 2 - 1)
    bphase = _rand((1, 1, gh, gw), seed + 202, device, dtype) * 2 * math.pi
    blocks = F.interpolate(blocks * torch.sin(2 * math.pi * phase[:, None, None, None] * glitch_speed + bphase), (h, w), mode="nearest")[:, 0] * block_jitter

    stripes = max(1, math.ceil(h / max(1, tear_height)))
    tears = (_rand((1, 1, stripes, 1), seed + 303, device, dtype) * 2 - 1)
    tphase = _rand((1, 1, stripes, 1), seed + 404, device, dtype) * 2 * math.pi
    tears = F.interpolate(tears * torch.sin(2 * math.pi * phase[:, None, None, None] * glitch_speed * .73 + tphase), (h, w), mode="nearest")[:, 0] * tear_amount

    spatial = along + wave + noise + blocks + tears
    if mode == "four_way_bands":
        coord = torch.remainder(spatial * max(1, int(band_count)) * 4 + p * 4, 4)
        field = coord / 4
    else:
        stage = phase * 4
        current = torch.floor(stage)
        local = stage - current
        hold = 1 - max(.01, min(1.0, transition_fraction))
        prog = ((local - hold) / max(.01, transition_fraction)).clamp(0, 1)
        prog = prog * prog * (3 - 2 * prog)
        margin = min(.48, .06 + abs(wave_amplitude) + abs(noise_amount) + abs(block_jitter) + abs(tear_amount))
        threshold = -margin + prog[:, None, None] * (1 + 2 * margin)
        softness = max(1e-5, edge_softness / max(h, w))
        blend = torch.sigmoid((threshold - spatial) / softness)
        field = torch.remainder(current[:, None, None] + blend, 4) / 4
    frac = torch.remainder(field * 4, 1)
    edge = 4 * frac * (1 - frac)
    return field.clamp(0, 1), edge.clamp(0, 1)


def _preview(field):
    pal = torch.tensor([[.95,.22,.20],[.20,.78,.30],[.20,.43,.95],[.95,.78,.18]], device=field.device, dtype=field.dtype)
    c = field * 4; basef = torch.floor(c); base = torch.remainder(basef.long(), 4); nxt = torch.remainder(base + 1, 4)
    a = (c - basef)[..., None]
    return pal[base] * (1 - a) + pal[nxt] * a


def _composite(xs, field, curve):
    c = field * 4; basef = torch.floor(c); base = torch.remainder(basef.long(), 4); nxt = torch.remainder(base + 1, 4)
    a = c - basef
    if curve == "smoothstep": a = a * a * (3 - 2 * a)
    elif curve == "hard": a = (a >= .5).to(a.dtype)
    out = torch.zeros_like(xs[0])
    for i, x in enumerate(xs):
        w = (base == i).to(a.dtype) * (1 - a) + (nxt == i).to(a.dtype) * a
        out += x * w[..., None]
    return out.clamp(0, 1), (4 * a * (1 - a)).clamp(0, 1)


EFFECT_INPUTS = {
    "mode": (["sequential_wipe", "four_way_bands"],),
    "cycle_frames": ("INT", {"default":96,"min":4,"max":10000,"step":4}),
    "transition_fraction": ("FLOAT", {"default":.78,"min":.01,"max":1.0,"step":.01}),
    "angle": ("FLOAT", {"default":0.0,"min":-180.0,"max":180.0,"step":1.0}),
    "wave_count": ("FLOAT", {"default":1.5,"min":0.0,"max":20.0,"step":.1}),
    "wave_amplitude": ("FLOAT", {"default":.10,"min":0.0,"max":.75,"step":.01}),
    "wave_speed": ("FLOAT", {"default":1.0,"min":-12.0,"max":12.0,"step":.1}),
    "noise_amount": ("FLOAT", {"default":.055,"min":0.0,"max":.75,"step":.005}),
    "noise_scale": ("FLOAT", {"default":2.0,"min":.1,"max":20.0,"step":.1}),
    "noise_speed": ("FLOAT", {"default":1.0,"min":-10.0,"max":10.0,"step":.1}),
    "block_size": ("INT", {"default":48,"min":1,"max":1024}),
    "block_jitter": ("FLOAT", {"default":.055,"min":0.0,"max":.75,"step":.005}),
    "tear_height": ("INT", {"default":20,"min":1,"max":512}),
    "tear_amount": ("FLOAT", {"default":.035,"min":0.0,"max":.75,"step":.005}),
    "glitch_speed": ("FLOAT", {"default":6.0,"min":0.0,"max":30.0,"step":.25}),
    "edge_softness": ("FLOAT", {"default":10.0,"min":0.0,"max":256.0,"step":.5}),
    "band_count": ("INT", {"default":1,"min":1,"max":16}),
    "seed": ("INT", {"default":42,"min":0,"max":0xFFFFFFFFFFFFFFFF}),
}


class UndulatingGlitchField:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"reference":("IMAGE",), **EFFECT_INPUTS}}
    RETURN_TYPES=("MASK","IMAGE","MASK"); RETURN_NAMES=("source_field","field_preview","transition_edges")
    FUNCTION="generate"; CATEGORY=CATEGORY
    def generate(self, reference, **kw):
        n,h,w=map(int,reference.shape[:3]); field,edge=_field(n,h,w,device=reference.device,dtype=reference.dtype,**kw)
        return field,_preview(field),edge


class FourWayFieldComposite:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"images_a":("IMAGE",),"images_b":("IMAGE",),"images_c":("IMAGE",),"images_d":("IMAGE",),"source_field":("MASK",),"frame_alignment":(["trim_shortest","loop_shorter","stretch_shorter"],),"blend_curve":(["smoothstep","linear","hard"],)}}
    RETURN_TYPES=("IMAGE","MASK"); RETURN_NAMES=("images","transition_edges"); FUNCTION="composite"; CATEGORY=CATEGORY
    def composite(self,images_a,images_b,images_c,images_d,source_field,frame_alignment,blend_curve):
        xs,n,h,w=_align([images_a,images_b,images_c,images_d],frame_alignment)
        f=source_field
        if f.ndim==2: f=f[None]
        f=F.interpolate(f[:,None].to(xs[0]),(h,w),mode="bilinear",align_corners=False)[:,0]
        idx=torch.arange(n,device=f.device)%f.shape[0] if frame_alignment=="loop_shorter" else torch.linspace(0,f.shape[0]-1,n,device=f.device).round().long() if frame_alignment=="stretch_shorter" else torch.arange(n,device=f.device)
        return _composite(xs,f.index_select(0,idx).clamp(0,1),blend_curve)


class FourWayUndulatingGlitchMixer:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"images_a":("IMAGE",),"images_b":("IMAGE",),"images_c":("IMAGE",),"images_d":("IMAGE",),"frame_alignment":(["trim_shortest","loop_shorter","stretch_shorter"],),"blend_curve":(["smoothstep","linear","hard"],),**EFFECT_INPUTS}}
    RETURN_TYPES=("IMAGE","IMAGE","MASK"); RETURN_NAMES=("images","field_preview","transition_edges"); FUNCTION="mix"; CATEGORY=CATEGORY
    def mix(self,images_a,images_b,images_c,images_d,frame_alignment,blend_curve,**kw):
        xs,n,h,w=_align([images_a,images_b,images_c,images_d],frame_alignment)
        field,_=_field(n,h,w,device=xs[0].device,dtype=xs[0].dtype,**kw)
        out,edge=_composite(xs,field,blend_curve)
        return out,_preview(field),edge


NODE_CLASS_MAPPINGS={"UG_UndulatingGlitchField":UndulatingGlitchField,"UG_FourWayFieldComposite":FourWayFieldComposite,"UG_FourWayUndulatingGlitchMixer":FourWayUndulatingGlitchMixer}
NODE_DISPLAY_NAME_MAPPINGS={"UG_UndulatingGlitchField":"Undulating Glitch Field (4-Way)","UG_FourWayFieldComposite":"Four-Way Field Composite","UG_FourWayUndulatingGlitchMixer":"Four-Way Undulating Glitch Mixer"}
