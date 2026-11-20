"""The exchange, points for cash at one fixed rate: the lock, the split of every person's points after round 1, and the value of a point."""

import numpy as np


def check_lock(locked_share, fund_block):
    """Stops the run unless locked_share equals fund_block: when points are issued, the locked points are exactly the fund's claim (phi = s)."""
    if not np.isclose(locked_share, fund_block):
        raise ValueError(f"locked_share ({locked_share}) must equal fund_block ({fund_block}) when points are issued; "
                         f"the fund grows later through unsold shares, not through locked_share")


def split_points(points, cash_block, locked_share):
    """Locked and usable points of each person after round 1, which used cash_block of every point (the paper's phi B_i and (1 - a - phi) B_i)."""
    if not 0 <= locked_share < 1 - cash_block:
        raise ValueError(f"locked_share must be at least 0 and below 1 - cash_block ({1 - cash_block}), not {locked_share}")
    # round1_points: each person's points used up by round 1, sold with the cash block for cash (the paper's a B_i)
    round1_points = cash_block * points
    # locked_points: each person's points locked as the claim on the fund, never bid, sold or put in the index (the paper's phi B_i)
    locked_points = locked_share * points
    # usable_points: what is left, a difference, so the three parts add back to the points up to the last digit (the paper's (1 - a - phi) B_i)
    usable_points = points - round1_points - locked_points
    return {"locked_points": locked_points, "usable_points": usable_points}


def make_point_value(total_cash_paid, cash_block, total_points):
    """Value of a point: round 1's price per point, the cash it raised over the cash_block of every point it used up (the paper's q*)."""
    return total_cash_paid / (cash_block * total_points)


def make_usable_point_value(total_first_price, worker_block, public_block, total_usable_points):
    """Value of a point seen from the usable points: the worker and public blocks of every firm at first prices, per usable point."""
    return (worker_block + public_block) * total_first_price / total_usable_points


def check_point_value(point_value, usable_point_value):
    """Stops the run unless the two formulas for the value of a point agree, so every point has one price."""
    if not np.isclose(point_value, usable_point_value):
        raise ValueError(f"the value of a point differs between round 1 ({point_value}) and the usable points ({usable_point_value})")
