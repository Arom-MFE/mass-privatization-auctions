"""Writes a run's files: people, firms, the round 1 bids and the numbers."""

import pandas as pd


def make_numbers(params, results):
    """Numbers table of the run: one row per setting, then one per single result, with columns kind, name and value."""
    # setting_rows: one row per params key, in dict order; dtype=object keeps whole numbers whole
    setting_rows = pd.DataFrame({"kind": "setting", "name": list(params), "value": list(params.values())},
                                dtype=object)
    # result_rows: one row per results key, in dict order; dtype=object keeps whole numbers whole
    result_rows = pd.DataFrame({"kind": "result", "name": list(results), "value": list(results.values())},
                               dtype=object)
    return pd.concat([setting_rows, result_rows], ignore_index=True)


def save_run(run, prefix, params, results, people, firms, bids):
    """Writes the run's four files into the folder run, each name starting with prefix; returns nothing."""
    # numbers: the run's settings and its single results, one row each
    numbers = make_numbers(params, results)
    # Without the index, so every file holds exactly its table's columns; running again overwrites them
    people.to_csv(run / f"{prefix}people.csv", index=False)
    firms.to_csv(run / f"{prefix}firms.csv", index=False)
    bids.to_csv(run / f"{prefix}cash_auction_bids.csv", index=False)
    numbers.to_csv(run / f"{prefix}numbers.csv", index=False)
