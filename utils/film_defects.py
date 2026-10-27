"""Generate bounded, deterministic film-space masks without storing video-sized masks."""

import hashlib
import math
import random

import torch
import torch.nn.functional as F


def random_stream(seed, *parts):
    """Derive local random streams independent of chunking, device and execution order."""
    key = ":".join(map(str, (seed, *parts))).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8, person=b"AlexzDefectsV1").digest(), "big"))


def _smooth_random(seed, channel, time):
    """Interpolate hashed knots without periodic oscillation or accumulated state."""
    index = math.floor(time)
    weight = time - index
    weight = weight * weight * (3 - 2 * weight)
    a = random_stream(seed, channel, index).uniform(-1, 1)
    b = random_stream(seed, channel, index + 1).uniform(-1, 1)
    return a + (b - a) * weight


def _position(rng, width, height, edge_bias):
    """Choose a uniform or edge-biased point in pixel-center coordinates."""
    x, y = rng.random() * width, rng.random() * height
    if rng.random() < edge_bias:
        side = rng.randrange(4)
        distance = rng.uniform(-0.01, 0.12)
        if side < 2:
            x = width * (distance if side == 0 else 1 - distance)
        else:
            y = height * (distance if side == 2 else 1 - distance)
    return x, y


def _patch(mask, cx, cy, rx, ry):
    """Allocate coordinates only inside a particle's clipped footprint."""
    height, width = mask.shape
    x0, x1 = max(0, math.floor(cx - rx - 2)), min(width, math.ceil(cx + rx + 2))
    y0, y1 = max(0, math.floor(cy - ry - 2)), min(height, math.ceil(cy + ry + 2))
    if x1 <= x0 or y1 <= y0:
        return None
    yy = torch.arange(y0, y1, device=mask.device, dtype=mask.dtype)[:, None] + 0.5 - cy
    xx = torch.arange(x0, x1, device=mask.device, dtype=mask.dtype)[None, :] + 0.5 - cx
    return mask[y0:y1, x0:x1], xx, yy


def _particle(mask, rng, kind, cx, cy, diameter, irregularity):
    """Rasterize an irregular speck, curved hair or soft stain with antialiased edges."""
    radius = max(0.3, diameter * 0.5)
    angle = rng.uniform(-math.pi, math.pi)
    cosine, sine = math.cos(angle), math.sin(angle)
    aspect = rng.uniform(0.55, 1.2)
    if kind == "hair":
        length = radius * rng.uniform(12, 25)
        patch = _patch(mask, cx, cy, length + radius * 4, length + radius * 4)
        if patch is None:
            return
        target, xx, yy = patch
        along, across = xx * cosine + yy * sine, -xx * sine + yy * cosine
        curve = irregularity * radius * 3 * ((along / length).square() - 0.4)
        distance = (across - curve).abs()
        opacity = (radius + 0.5 - distance).clamp(0, 1) * (length + 0.5 - along.abs()).clamp(0, 1)
    else:
        spread = 3 if kind == "stain" else 1.5
        patch = _patch(mask, cx, cy, radius * spread, radius * spread)
        if patch is None:
            return
        target, xx, yy = patch
        u, v = xx * cosine + yy * sine, (-xx * sine + yy * cosine) / aspect
        theta = torch.atan2(v, u)
        phase = rng.uniform(-math.pi, math.pi)
        edge = radius * (1 + irregularity * (0.16 * torch.sin(theta * 3 + phase) + 0.09 * torch.cos(theta * 5 - phase)))
        distance = (u.square() + v.square()).sqrt()
        if kind == "stain":
            opacity = torch.exp(-1.5 * (distance / edge.clamp_min(0.1)).square())
        else:
            opacity = (edge + 0.5 - distance).clamp(0, 1)
    opacity *= min(1.0, diameter / 0.7) * rng.uniform(0.55, 1.0)
    target.copy_(torch.maximum(target, opacity))


def _scratch(mask, rng, frame, seed, width_px, irregularity, edge_bias):
    """Continue a wandering longitudinal scratch across successive physical frames."""
    height, width = mask.shape
    cx, _ = _position(rng, width, height, edge_bias)
    drift = width * 0.008 * irregularity
    left = random_stream(seed, "path", frame).uniform(-1, 1)
    right = random_stream(seed, "path", frame + 1).uniform(-1, 1)
    y = (torch.arange(height, device=mask.device, dtype=mask.dtype) + 0.5) / height
    path = cx + drift * (left + (right - left) * y)
    radius = max(0.2, width_px * 0.5) * rng.uniform(0.65, 1.3)
    x0 = max(0, math.floor(cx - drift - radius - 2))
    x1 = min(width, math.ceil(cx + drift + radius + 2))
    if x1 <= x0:
        return
    x = torch.arange(x0, x1, device=mask.device, dtype=mask.dtype) + 0.5
    line = (radius + 0.5 - (x[None, :] - path[:, None]).abs()).clamp(0, 1)
    # Прерывистость связана с координатой вдоль плёнки, а не с размером порции.
    a = random_stream(seed, "gaps", frame).random()
    b = random_stream(seed, "gaps", frame + 1).random()
    modulation = (1 - irregularity) + irregularity * ((a + (b - a) * y) * 2.5).clamp(0, 1)
    line *= modulation[:, None] * min(1.0, width_px / 0.6)
    target = mask[:, x0:x1]
    target.copy_(torch.maximum(target, line))


def surface_masks(height, width, frame, seed, pattern, amount, size, persistence,
                  irregularity, edge_bias, gate, device):
    """Return separate dust, scratch and stain masks for a single material frame."""
    masks = {name: torch.zeros((height, width), device=device, dtype=torch.float32)
             for name in ("dust", "scratches", "stains")}
    kinds = ("dust", "hair") if gate else ("scratches", "dust", "stains")
    for slot in range(math.ceil(amount)):
        phase = random_stream(seed, "phase", slot).randrange(persistence)
        event = (frame + phase) // persistence
        kind = kinds[slot % len(kinds)] if pattern == "mixed" else pattern
        # Пыль на плёнке меняется с физическим кадром; пыль окна держится событием.
        index = frame if not gate and kind == "dust" else event
        rng = random_stream(seed, kind, slot, index)
        if slot >= int(amount) and rng.random() >= amount - int(amount):
            continue
        local_seed = rng.randrange(2**63)
        diameter = width * size / 100 * rng.uniform(0.7, 1.4)
        if kind == "scratches":
            _scratch(masks["scratches"], rng, frame, local_seed, diameter, irregularity, edge_bias)
            continue
        cx, cy = _position(rng, width, height, edge_bias)
        if gate:
            cx += diameter * irregularity * _smooth_random(local_seed, "x", frame / 8)
            cy += diameter * irregularity * _smooth_random(local_seed, "y", frame / 8)
        if kind == "stains" and pattern == "mixed":
            diameter *= 20
        elif kind == "dust" and pattern == "mixed" and not gate:
            diameter *= 2
        target = masks["stains" if kind == "stains" else "dust"]
        _particle(target, rng, "stain" if kind == "stains" else kind, cx, cy, diameter, irregularity)
    return masks


def development_field(height, width, frame, seed, pattern, size, persistence, irregularity, device):
    """Interpolate coarse density fields in space and time without batch normalization."""
    spacing = max(1.0, width * size / 100)
    rows = max(2, min(128, math.ceil(height / spacing) + 1))
    columns = max(2, min(128, math.ceil(width / spacing) + 1))
    time = frame / persistence
    index = math.floor(time)
    weight = time - index
    weight = weight * weight * (3 - 2 * weight)

    def field(channel, r, c):
        """Blend two small deterministic grids and expand only one frame on the device."""
        grids = []
        for knot in (index, index + 1):
            rng = random_stream(seed, channel, knot)
            grids.append(torch.tensor([rng.uniform(-1, 1) for _ in range(r * c)], device=device).view(1, 1, r, c))
        grid = grids[0].lerp(grids[1], weight)
        if irregularity < 1:
            rng = random_stream(seed, channel, "static")
            stationary = torch.tensor([rng.uniform(-1, 1) for _ in range(r * c)], device=device).view(1, 1, r, c)
            grid = stationary.lerp(grid, irregularity)
        return F.interpolate(grid, size=(height, width), mode="bicubic", align_corners=False)[0, 0].clamp(-1, 1)

    if pattern == "mottling":
        return field("mottling", rows, columns)
    if pattern == "streaks":
        return field("streaks", 2, columns)
    return field("mottling", rows, columns) * 0.65 + field("streaks", 2, columns) * 0.35
