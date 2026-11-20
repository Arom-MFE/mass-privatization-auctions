"""Part 1 of the simulation: the people, their points, the firms and who works where."""

import numpy as np
import pandas as pd

from sim.streams import make_stream


def draw_world(seed, n_people, n_firms):
    """Every random number of part 1, drawn from the world stream in the paper's order."""
    # rng: the world stream, used by this part only
    rng = make_stream(seed, "world")
    # z_points: one standard normal per person, turned into points
    z_points = rng.standard_normal(n_people)
    # z_headcount: one standard normal per firm, turned into headcounts
    z_headcount = rng.standard_normal(n_firms)
    # z_book: one standard normal per firm, turned into book values
    z_book = rng.standard_normal(n_firms)
    # shuffle: all people in a random order, used to hand out the jobs
    shuffle = rng.permutation(n_people)
    return {"z_points": z_points, "z_headcount": z_headcount, "z_book": z_book, "shuffle": shuffle}


def make_people(n_people):
    """Start the people table: one row per person, with person ids from 1 to n_people."""
    return pd.DataFrame({"person_id": np.arange(1, n_people + 1)})


def make_points(z_points, sigma_points, avg_points):
    """Points of each person: a lognormal with mean avg_points, avg_points exp(sigma_points z_points - sigma_points^2 / 2) (the paper's B_i)."""
    # Subtracting sigma_points^2 / 2 keeps the average at avg_points points per person
    return avg_points * np.exp(sigma_points * z_points - sigma_points**2 / 2)


def count_insiders(n_people, insider_share):
    """Number of insiders: the given share of all people, rounded to whole people."""
    return int(round(insider_share * n_people))


def make_firms(n_firms):
    """Start the firms table: one row per firm, with firm ids from 1 to n_firms."""
    return pd.DataFrame({"firm_id": np.arange(1, n_firms + 1)})


def make_headcounts(z_headcount, sigma_headcount, n_insiders, min_headcount):
    """Workers of each firm: a lognormal share of all insiders, at least min_headcount, adding up to n_insiders (the paper's L_j)."""
    if len(z_headcount) * min_headcount > n_insiders:
        raise ValueError(f"{n_insiders} insiders cannot give each of {len(z_headcount)} firms "
                         f"at least {min_headcount} workers")
    # raw_size: raw size of each firm; only relative sizes matter, so no mean correction is needed (the paper's tilde L_j)
    raw_size = np.exp(sigma_headcount * z_headcount)
    # headcount: each firm's share of the insiders, rounded to whole workers and never below min_headcount
    headcount = np.maximum(min_headcount, np.round(n_insiders * raw_size / raw_size.sum())).astype(int)
    # The largest firm takes the rounding difference, so the headcounts add up to n_insiders exactly
    headcount[np.argmax(headcount)] += n_insiders - headcount.sum()
    return headcount


def make_book_values(z_book, headcount, avg_book_per_worker, sigma_book):
    """Book value of each firm: its headcount times a capital per worker that averages avg_book_per_worker (the paper's K_j)."""
    # Subtracting sigma_book^2 / 2 keeps the average capital per worker at avg_book_per_worker
    return avg_book_per_worker * headcount * np.exp(sigma_book * z_book - sigma_book**2 / 2)


def assign_workers(shuffle, headcount, n_people):
    """Firm id of each person: the first people of the shuffle fill firm 1 up to its headcount, the next fill firm 2, and so on."""
    # firm_id: the firm of each person; everyone starts at 0, which means no firm, since real firm ids start at 1
    firm_id = np.zeros(n_people, dtype=int)
    # The first n_insiders people of the shuffle get jobs, firm by firm in id order
    firm_id[shuffle[:headcount.sum()]] = np.repeat(np.arange(1, len(headcount) + 1), headcount)
    return firm_id


def is_insider(firm_id):
    """Insider dummy: 1 for a person who works in a firm of the pool, 0 for everyone else."""
    return (firm_id > 0).astype(int)


def worker_points(points, firm_id, n_firms):
    """Points held by the workers of each firm together (the paper's V^W_j)."""
    # Add each person's points to their firm's total; bin 0 collects the outsiders and is dropped
    return np.bincount(firm_id, weights=points, minlength=n_firms + 1)[1:]


def summarize_points(points, firm_id, n_firms):
    """Spread of points for everyone, for all insiders and for the workers of each firm."""
    # stats: the statistics to compute, in column order
    stats = ["count", "min", "max", "mean", "median", "std"]
    # point_series: the points as a pandas Series
    point_series = pd.Series(points)
    # table: one row for everyone and one for all insiders together
    table = {"everyone": point_series.agg(stats), "insiders": point_series[firm_id > 0].agg(stats)}
    # Then one row for the workers of each firm
    for j in range(1, n_firms + 1):
        table[f"firm {j}"] = point_series[firm_id == j].agg(stats)
    # summary: one row per group and one column per statistic
    summary = pd.DataFrame(table).T
    # Counts are whole people
    summary["count"] = summary["count"].astype(int)
    return summary
