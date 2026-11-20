"""The exchange, points for cash at one fixed rate; for now, the split of every person's points into locked and usable."""


def split_points(points, locked_share):
    """Locked and usable points of each person: locked_share of their points, and the rest (the paper's phi B_i and (1 - phi) B_i)."""
    if not 0 <= locked_share < 1:
        raise ValueError(f"locked_share must be at least 0 and below 1, not {locked_share}")
    # locked_points: each person's points locked in the index, never bid or sold (the paper's phi B_i)
    locked_points = locked_share * points
    # usable_points: the rest, a difference, so the two parts add back to the points up to rounding in the last digit (the paper's (1 - phi) B_i)
    usable_points = points - locked_points
    return {"locked_points": locked_points, "usable_points": usable_points}
