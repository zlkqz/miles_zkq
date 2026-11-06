"""GEPA phase placement over the GPU slots already owned by Miles.

The planner never allocates a second placement group, assumes a global CUDA
index, or changes the normal trainer topology. Each temporary engine stays on
one node, with the same GPU count as a normal rollout engine. Exactly one
engine is reserved for reflection; the remaining engines serve policy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSlot:
    slot: int
    bundle_index: int
    node_id: str
    node_ip: str
    physical_gpu_id: int


@dataclass(frozen=True)
class EnginePlacement:
    role: str
    devices: tuple[GpuSlot, ...]

    @property
    def tp(self):
        return len(self.devices)

    @property
    def slot_offset(self):
        return self.devices[0].slot


def plan_gepa_placement(slots, *, rollout_gpus_per_engine):
    """Tile owned devices like training rollout; reserve exactly one reflection engine."""
    devices = tuple(GpuSlot(**slot) if isinstance(slot, dict) else slot for slot in slots)
    tp = rollout_gpus_per_engine
    if type(tp) is not int or tp < 1:
        raise ValueError("rollout_gpus_per_engine must be a positive integer")
    if tp >= len(devices):
        raise ValueError("One reflection engine must leave at least one policy engine")
    if [device.slot for device in devices] != list(range(len(devices))):
        raise ValueError("GPU slots must preserve Miles' full, ordered placement")
    identities = {(device.node_id, device.physical_gpu_id) for device in devices}
    if len(identities) != len(devices):
        raise ValueError("Duplicate physical devices in owned GPU slots")
    engines = _tile(devices, "policy", tp)
    return tuple([*engines[:-1], EnginePlacement("reflection", engines[-1].devices)])


def _tile(devices, role, tp):
    by_node = {}
    for device in devices:
        by_node.setdefault(device.node_id, []).append(device)
    placements = []
    for node, rows in by_node.items():
        if len(rows) % tp:
            raise ValueError(f"{role} has {len(rows)} GPUs on node {node}, not divisible by TP={tp}")
        for start in range(0, len(rows), tp):
            tile = tuple(rows[start : start + tp])
            if [d.slot for d in tile] != list(range(tile[0].slot, tile[0].slot + tp)):
                raise ValueError("Engine placement must use consecutive Miles slots")
            if [d.physical_gpu_id for d in tile] != list(range(tile[0].physical_gpu_id, tile[0].physical_gpu_id + tp)):
                raise ValueError("Native engine launcher requires contiguous physical GPUs within a node")
            placements.append(EnginePlacement(role, tile))
    return placements
