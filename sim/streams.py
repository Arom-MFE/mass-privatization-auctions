"""The four random streams of the simulation, all made from one seed."""

import numpy as np

# The streams in the order the seed makes them; each part draws only from its own
STREAMS = ["world", "round1", "round2", "round3"]


def make_stream(seed, name):
    """Random number generator for one part: "world", "round1", "round2" or "round3"."""
    if name not in STREAMS:
        raise ValueError(f"Unknown stream {name!r}; use one of {STREAMS}")
    # children: one independent seed per stream, in the order of STREAMS
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return np.random.default_rng(children[STREAMS.index(name)])