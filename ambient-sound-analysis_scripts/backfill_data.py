"""Backfill 7 days of broadband + PSD data into data_3.0 in 1-hour chunks."""

import argparse
import datetime as dt
import os
import shutil
import subprocess
import tempfile
import traceback

from orcasound_noise.pipeline.pipeline import NoiseAnalysisPipeline
from orcasound_noise.utils import Hydrophone

CHUNK_HOURS = 1
BACKFILL_DAYS = 7
BUFFER_MINUTES = 15

S3_BUCKET = "acoustic-sandbox"
TEST_PREFIX = "test/backfill"

HYDROPHONE_MAP = {h.value.name: h for h in Hydrophone if h.name != "HPhoneTup"}


def sync_to_test(local_folder: str, s3_save_folder: str):
    """Sync local partitioned output to the test prefix in S3."""
    src = os.path.join(local_folder, s3_save_folder)
    dest = f"s3://{S3_BUCKET}/{TEST_PREFIX}/{s3_save_folder}/"
    subprocess.run(
        ["aws", "s3", "sync", src, dest, "--no-overwrite"],
        check=True,
    )
    print(f"  Synced to {dest}")


def backfill_hydrophone(
    hydrophone: Hydrophone, start: dt.datetime, end: dt.datetime, test_mode: bool = False
):
    name = hydrophone.value.name
    s3_save_folder = hydrophone.value.save_folder
    print(f"\n{'='*60}")
    print(f"Backfilling {name}: {start} -> {end}")
    if test_mode:
        print(f"*** TEST MODE — writing to s3://{S3_BUCKET}/{TEST_PREFIX}/{s3_save_folder}/ ***")
    print(f"{'='*60}")

    pipeline = NoiseAnalysisPipeline(
        hydrophone, delta_f=1, bands=12, delta_t=1, mode="safe"
    )

    tmp_dir = tempfile.mkdtemp() if test_mode else None

    chunk = dt.timedelta(hours=CHUNK_HOURS)
    current = start
    success = 0
    errors = 0

    while current < end:
        chunk_end = min(current + chunk, end)
        print(f"  {current} -> {chunk_end} ... ", end="", flush=True)
        try:
            if test_mode:
                pipeline.generate_parquet_file(
                    current, chunk_end,
                    pqt_folder_override=tmp_dir,
                    upload_to_s3=False,
                    partitioning=True,
                )
                sync_to_test(tmp_dir, s3_save_folder)
            else:
                pipeline.generate_parquet_file(
                    current, chunk_end, upload_to_s3=True, partitioning=True
                )
            print("OK")
            success += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
        current = chunk_end

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

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
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Write to test prefix (s3://{S3_BUCKET}/{TEST_PREFIX}/) instead of production",
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
            s, e = backfill_hydrophone(hydrophone, start, end, args.test)
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
