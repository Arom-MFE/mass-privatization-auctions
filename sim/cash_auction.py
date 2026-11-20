"""Round 1 of the simulation: the cash auction for every firm's cash block."""

import numpy as np
import pandas as pd

from sim.streams import make_stream


def draw_cash_auction(seed, n_firms, min_investors, avg_investors):
    """Every random number of round 1 from the round1 stream: investors per firm (the paper's n_j), then one normal per investor."""
    if min_investors < 1 or min_investors != int(min_investors):
        raise ValueError(f"min_investors must be a whole number of at least 1, not {min_investors}")
    if avg_investors < min_investors:
        raise ValueError(f"avg_investors ({avg_investors}) must be at least min_investors ({min_investors})")
    # rng: the round 1 stream, used by this part only
    rng = make_stream(seed, "round1")
    # no_show_chance: chance that the next investor in line does not come, which ends the line (p)
    no_show_chance = 1 / (avg_investors - min_investors + 1)
    # n_investors: investors bidding for each firm's cash block; min_investors for sure, then a geometric line
    n_investors = (int(min_investors) - 1) + rng.geometric(no_show_chance, n_firms)
    # z_values: one standard normal per investor, firm 1's investors first, then firm 2's, and so on (the paper's Z_b)
    z_values = rng.standard_normal(n_investors.sum())
    return {"n_investors": n_investors, "z_values": z_values}


def make_reserves(book_value, cash_block):
    """Reserve of each firm's cash block, the lowest price it can sell for (the paper's a K_j)."""
    return cash_block * book_value


def make_bids(n_investors, z_values, reserve, sigma_investor):
    """Bids table: each investor's firm, number within the firm, value (the paper's v_b) and rank, in draw order."""
    if len(z_values) != n_investors.sum():
        raise ValueError(f"{len(z_values)} values drawn for {n_investors.sum()} investors")
    # firm_id: the firm of each investor, firm 1's investors first, then firm 2's
    firm_id = np.repeat(np.arange(1, len(n_investors) + 1), n_investors)
    # investor: each investor's number within its firm, 1, 2, ... in draw order
    investor = pd.Series(firm_id).groupby(firm_id).cumcount().to_numpy() + 1
    # multiple: each investor's value as a multiple of the reserve, never below 1
    multiple = np.exp(sigma_investor * np.abs(z_values))
    # value: each investor's value for the block, the reserve of its firm times its multiple (the paper's v_b)
    value = np.repeat(reserve, n_investors) * multiple
    # bids: one row per investor, in draw order
    bids = pd.DataFrame({"firm_id": firm_id, "investor": investor, "value": value})
    # rank: 1 for the highest value in each firm; ties, possible only at sigma_investor 0, go to the earlier investor
    bids["rank"] = bids.groupby("firm_id")["value"].rank(method="first", ascending=False).astype(int)
    return bids


def top_values(bids, n_firms):
    """Value ranked 1 in each firm, the winner's value, in firm_id order (the paper's v_(1))."""
    # winners: the value ranked 1 in each firm, indexed by firm_id
    winners = bids.loc[bids["rank"] == 1].set_index("firm_id")["value"]
    # Line them up with firms 1 to n_firms
    return winners.reindex(np.arange(1, n_firms + 1)).to_numpy()


def make_cash_paid(bids, reserve):
    """Cash the winner pays for each firm's block: the value ranked 2, or the reserve for a lone investor (the paper's P_j)."""
    # runners_up: the value ranked 2 in each firm that has one, indexed by firm_id
    runners_up = bids.loc[bids["rank"] == 2].set_index("firm_id")["value"]
    # second_value: the runner-up's value of firms 1 to n_firms, missing for a firm with a single investor
    second_value = runners_up.reindex(np.arange(1, len(reserve) + 1)).to_numpy()
    # A lone investor pays the reserve
    return np.where(np.isnan(second_value), reserve, second_value)


def make_first_prices(cash_paid, cash_block):
    """First price of each firm: the whole firm at the price paid for its cash block (the paper's p0_j)."""
    return cash_paid / cash_block
