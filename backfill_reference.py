"""
Backfill historical rolling 7-day broadband reference levels for Orcasound hydrophones.

Usage:
    python backfill_reference.py --start 2023-01-01
    python backfill_reference.py --start 2023-01-01 --end 2024-12-31
    python backfill_reference.py --start 2023-01-01 --hydrophone BUSH_POINT
"""

import argparse
import datetime as dt
import traceback
from dataclasses import asdict

import boto3
import polars as pl

from orcasound_noise.utils import Hydrophone

from ambient_reference import (
    TIMEZONE,
    compute_reference_for_day,
    date_range,
    read_existing,
    write_to_s3,
)


def backfill_hydrophone(
    s3_client, hydrophone: Hydrophone, start: dt.date, end: dt.date
) -> int:
    """Backfill reference levels for a hydrophone over a date range."""
    name = hydrophone.name
    print(f"\nBackfilling {name} from {start} to {end}...")

    existing_df = read_existing(s3_client, hydrophone)
    existing_dates: set[dt.date] = set()
    if existing_df is not None and len(existing_df) > 0:
        dates_col = existing_df.select(pl.col("date")).to_series()
        existing_dates = {
            d.date() if isinstance(d, dt.datetime) else d for d in dates_col
        }
        print(f"  Found {len(existing_dates)} existing rows, skipping those dates")

    new_rows = []
    for current in date_range(start, end):
        if current in existing_dates:
            continue
        try:
            row = compute_reference_for_day(hydrophone, current)
            if row is not None:
                new_rows.append(row)
                print(f"  {current}: bb={row.bb_ref:.1f} comm={row.comm_bb_ref:.1f} ship={row.ship_bb_ref:.1f} dB")
            else:
                print(f"  {current}: no data in window, skipping")
        except Exception as e:
            print(f"  {current}: error - {e}")

    if not new_rows:
        print(f"  No new data computed for {name}.")
        return 0

    new_df = pl.DataFrame([asdict(row) for row in new_rows]).with_columns(
        pl.col("date").cast(pl.Date),
    )

    if existing_df is not None:
        combined = pl.concat([existing_df, new_df]).sort("date")
    else:
        combined = new_df.sort("date")

    write_to_s3(s3_client, hydrophone, combined)
    return len(new_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill broadband reference levels")
    parser.add_argument(
        "--start", required=True, type=lambda s: dt.date.fromisoformat(s),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=lambda s: dt.date.fromisoformat(s), default=None,
        help="End date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--hydrophone", type=str, default=None,
        help=f"Hydrophone name (e.g. BUSH_POINT). Defaults to all. Options: {', '.join(h.name for h in Hydrophone if h.name != 'HPhoneTup')}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    today = dt.datetime.now(TIMEZONE).date()
    end = args.end or today

    if args.hydrophone:
        hydrophones = [Hydrophone[args.hydrophone]]
    else:
        hydrophones = [h for h in Hydrophone if h.name != "HPhoneTup"]

    print(f"Backfill range: {args.start} to {end}")
    print(f"Hydrophones: {', '.join(h.name for h in hydrophones)}")

    s3_client = boto3.client("s3")
    summary = {}

    for hydrophone in hydrophones:
        try:
            count = backfill_hydrophone(s3_client, hydrophone, args.start, end)
            summary[hydrophone.name] = count
        except Exception:
            print(f"\nERROR processing {hydrophone.name}:")
            traceback.print_exc()
            summary[hydrophone.name] = None

    print("\n" + "=" * 60)
    print("Backfill Summary:")
    for name, result in summary.items():
        if result is None:
            print(f"  {name}: FAILED")
        else:
            print(f"  {name}: {result} new rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
