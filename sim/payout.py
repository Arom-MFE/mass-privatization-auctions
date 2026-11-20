"""The payout after round 1: the check of the split, the split of the cash and every person's payout."""

import numpy as np


def check_split(cash_block, worker_block, public_block, fund_block):
    """Stops the run unless the four blocks of every firm add up to one, with a and w above 0 and c and s at least 0 (a + w + c + s = 1)."""
    # blocks: the four shares of every firm, in the paper's order a, w, c, s
    blocks = np.array([cash_block, worker_block, public_block, fund_block])
    if np.any(blocks < 0) or np.any(blocks > 1):
        raise ValueError(f"each block must be between 0 and 1, not cash {cash_block}, worker {worker_block}, "
                         f"public {public_block}, fund {fund_block}")
    if cash_block <= 0:
        raise ValueError(f"cash_block must be above 0, since round 1 sets the first prices every later price uses, not {cash_block}")
    if worker_block <= 0:
        raise ValueError(f"worker_block must be above 0, since every firm has a worker auction, not {worker_block}")
    if not np.isclose(blocks.sum(), 1):
        raise ValueError(f"the four blocks must add up to one, not {blocks.sum()}")


def split_cash(total_cash_paid, payout_share):
    """Split of round 1 cash: total_payout to citizens and the tax the treasury keeps (the paper's lambda_C and lambda_T)."""
    if not 0 <= payout_share <= 1:
        raise ValueError(f"payout_share must be between 0 and 1, not {payout_share}")
    # total_payout: the cash paid out to citizens (the paper's lambda_C times the sum of P_j)
    total_payout = payout_share * total_cash_paid
    # tax: the treasury's share, a difference so the two add up exactly (the paper's lambda_T times the sum of P_j)
    tax = total_cash_paid - total_payout
    return {"total_payout": total_payout, "tax": tax}


def make_payout_per_point(total_payout, total_points):
    """Cash paid out per point issued, total_payout / total_points (the paper's m0)."""
    return total_payout / total_points


def make_payouts(points, payout_per_point):
    """Cash each person gets at the payout, payout_per_point times their points (the paper's M_i at the payout)."""
    return payout_per_point * points
