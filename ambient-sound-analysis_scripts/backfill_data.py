"""
Backfill 7 days of broadband + PSD data into data_3.0 for all hydrophones.

Uses the same NoiseAnalysisPipeline settings as the scheduled GitHub Action
(erinmee/orca-action-workflow). Processes in 1-hour chunks and stops 15 minutes
before the current time to avoid overlap with the live pipeline.

Usage:
    conda activate ambient-ref
    python backfill_data.py                          # all hydrophones
    python backfill_data.py --hydrophone orcasound_lab  # single hydrophone
"""

import argparse
import datetime as dt
import traceback

from orcasound_noise.pipeline.pipeline import NoiseAnalysisPipeline
from orcasound_noise.utils import Hydrophone

CHUNK_HOURS = 1
BACKFILL_DAYS = 7
BUFFER_MINUTES = 15

HYDROPHONE_MAP = {h.value.name: h for h in Hydrophone if h.name != "HPhoneTup"}


def backfill_hydrophone(hydrophone: Hydrophone, start: dt.datetime, end: dt.datetime):
    name = hydrophone.value.name
    print(f"\n{'='*60}")
    print(f"Backfilling {name}: {start} -> {end}")
    print(f"{'='*60}")

    pipeline = NoiseAnalysisPipeline(
        hydrophone, delta_f=1, bands=12, delta_t=1, mode="safe"
    )

    chunk = dt.timedelta(hours=CHUNK_HOURS)
    current = start
    success = 0
    errors = 0

    while current < end:
        chunk_end = min(current + chunk, end)
        print(f"  {current} -> {chunk_end} ... ", end="", flush=True)
        try:
            pipeline.generate_parquet_file(
                current, chunk_end, upload_to_s3=True, partitioning=True
            )
            print("OK")
            success += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
        current = chunk_end

    print(f"\n  Done: {success} chunks OK, {errors} errors")
    return success, errors


def main():
    parser = argparse.ArgumentParser(description="Backfill 7 days of data_3.0")
    parser.add_argument(
        "--hydrophone",
        type=str,
        default=None,
        choices=list(HYDROPHONE_MAP.keys()),
        help="Single hydrophone to backfill (default: orcasound_lab)",
    )
    args = parser.parse_args()

    pst = dt.timezone(dt.timedelta(hours=-8), name="PST")
    now = dt.datetime.now(pst)
    end = now - dt.timedelta(minutes=BUFFER_MINUTES)
    start = end - dt.timedelta(days=BACKFILL_DAYS)

    print(f"Backfill window: {start} -> {end}")

    if args.hydrophone:
        targets = [HYDROPHONE_MAP[args.hydrophone]]
    else:
        targets = [HYDROPHONE_MAP["orcasound_lab"]]

    summary = {}
    for hydrophone in targets:
        try:
            s, e = backfill_hydrophone(hydrophone, start, end)
            summary[hydrophone.value.name] = (s, e)
        except Exception:
            print(f"\nFATAL error on {hydrophone.value.name}:")
            traceback.print_exc()
            summary[hydrophone.value.name] = None

    print(f"\n{'='*60}")
    print("Summary:")
    for name, result in summary.items():
        if result is None:
            print(f"  {name}: FAILED")
        else:
            print(f"  {name}: {result[0]} OK, {result[1]} errors")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
