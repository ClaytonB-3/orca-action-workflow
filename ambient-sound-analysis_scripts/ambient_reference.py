"""
Compute rolling 7-day broadband reference levels (5th percentile)
for all Orcasound hydrophones and write results to S3.

Uses the most recent 604,800 data points (7 days at 1-second resolution)
ending at "now", looking back up to 14 calendar days to find enough data.
Stores the result under tomorrow's date so the PSD pipeline always has a
ref value available for any day it runs.

Produces per-hydrophone ref files matching the schema expected by
the pipeline: date, bb_ref, comm_bb_ref, ship_bb_ref.
"""

import argparse
import datetime as dt
import io
import traceback
from dataclasses import asdict, dataclass
from zoneinfo import ZoneInfo

import boto3
import polars as pl

from orcasound_noise.utils import Hydrophone

TEST_PREFIX = "test/ambient_reference"
WINDOW_DAYS = 7
TARGET_ROWS = WINDOW_DAYS * 24 * 60 * 60  # 604,800 seconds of data
MAX_LOOKBACK_DAYS = 14
TIMEZONE = ZoneInfo("US/Pacific")

# data_3.0 broadband columns
BB_COL = "bb_o"
COMM_BB_COL = "comm_bb_o"
SHIP_BB_COL = "ship_bb_o"
TIMESTAMP_COL = "ind"


@dataclass
class ReferenceRow:
    date: dt.date
    bb_ref: float
    comm_bb_ref: float
    ship_bb_ref: float



def s3_ref_key(hydrophone: Hydrophone, test_mode: bool = False) -> str:
    """S3 key matching the path the pipeline reads refs from."""
    name = hydrophone.value.name
    save_folder = hydrophone.value.save_folder
    key = f"{save_folder}/ref/hydrophone={name}/{name}_ref.parquet"
    if test_mode:
        key = f"{TEST_PREFIX}/{key}"
    return key


def bb_paths_for_range(hydrophone: Hydrophone, start: dt.date, end: dt.date) -> list[str]:
    """Build S3 glob paths for broadband parquets over a date range."""
    name = hydrophone.value.name
    bucket = hydrophone.value.save_bucket
    folder = hydrophone.value.save_folder
    paths = []
    d = start
    while d <= end:
        paths.append(
            f"s3://{bucket}/{folder}/broadband/hydrophone={name}"
            f"/year={d.year}/month={d.month:02d}/day={d.day:02d}/*.parquet"
        )
        d += dt.timedelta(days=1)
    return paths


def read_existing(s3_client, hydrophone: Hydrophone, test_mode: bool = False) -> pl.DataFrame | None:
    """Read existing reference parquet from S3, or return None."""
    bucket = hydrophone.value.save_bucket
    key = s3_ref_key(hydrophone, test_mode)
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        return pl.read_parquet(io.BytesIO(data))
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"  Warning: could not read existing data for {hydrophone.name}: {e}")
        return None


def write_to_s3(s3_client, hydrophone: Hydrophone, df: pl.DataFrame, test_mode: bool = False) -> None:
    """Write reference parquet to S3."""
    bucket = hydrophone.value.save_bucket
    key = s3_ref_key(hydrophone, test_mode)
    buf = io.BytesIO()
    df.write_parquet(buf)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"  Wrote {len(df)} rows to s3://{bucket}/{key}")


def find_earliest_data_date(s3_client, hydrophone: Hydrophone) -> dt.date | None:
    """Find the earliest date with broadband data via S3 listing."""
    name = hydrophone.value.name
    bucket = hydrophone.value.save_bucket
    folder = hydrophone.value.save_folder
    prefix = f"{folder}/broadband/hydrophone={name}/year="
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    if resp.get("KeyCount", 0) == 0:
        return None
    # First key is the earliest (S3 lists alphabetically by key)
    # e.g. .../year=2026/month=03/day=01/file.parquet
    key = resp["Contents"][0]["Key"]
    parts = key.split("/")
    year = month = day = None
    for part in parts:
        if part.startswith("year="):
            year = int(part.split("=")[1])
        elif part.startswith("month="):
            month = int(part.split("=")[1])
        elif part.startswith("day="):
            day = int(part.split("=")[1])
    if year and month and day:
        return dt.date(year, month, day)
    return None


def compute_initial_reference(
    hydrophone: Hydrophone, earliest_date: dt.date
) -> tuple[float, float, float] | None:
    """Compute 5th percentile over the first WINDOW_DAYS days of data.

    Used before there is enough data history to compute a rolling reference.
    """
    start_time = dt.datetime.combine(earliest_date, dt.time.min)
    end_time = dt.datetime.combine(earliest_date + dt.timedelta(days=WINDOW_DAYS), dt.time.min)

    paths = bb_paths_for_range(hydrophone, earliest_date, earliest_date + dt.timedelta(days=WINDOW_DAYS - 1))
    try:
        bb_df = (
            pl.scan_parquet(paths, storage_options={"aws_region": "us-west-2"})
            .filter(pl.col(TIMESTAMP_COL).is_between(start_time, end_time, closed="left"))
            .collect()
        )
    except Exception as e:
        print(f"  Warning: bootstrap computation failed: {e}")
        return None

    if bb_df.height == 0:
        return None

    quantiles = bb_df.select(
        pl.col(BB_COL).quantile(0.05).alias("bb_ref"),
        pl.col(COMM_BB_COL).quantile(0.05).alias("comm_bb_ref"),
        pl.col(SHIP_BB_COL).quantile(0.05).alias("ship_bb_ref"),
    ).row(0)

    return (float(quantiles[0]), float(quantiles[1]), float(quantiles[2]))


def compute_reference(
    hydrophone: Hydrophone, now: dt.datetime, ref_date: dt.date
) -> ReferenceRow | None:
    """Compute 5th percentile over the most recent TARGET_ROWS data points.

    Loads data backward from `now`, expanding the calendar window day-by-day
    up to MAX_LOOKBACK_DAYS. If not enough rows are found, logs a warning
    and uses whatever data is available.
    """
    name = hydrophone.value.name

    # Strip timezone for filtering — parquet ind column is naive datetime[ns]
    now_naive = now.replace(tzinfo=None)

    # Start with WINDOW_DAYS + 1 to account for partial current day
    for lookback in range(WINDOW_DAYS + 1, MAX_LOOKBACK_DAYS + 1):
        start_time = now_naive - dt.timedelta(days=lookback)
        paths = bb_paths_for_range(hydrophone, start_time.date(), now_naive.date())
        try:
            bb_df = (
                pl.scan_parquet(paths, storage_options={"aws_region": "us-west-2"})
                .filter(pl.col(TIMESTAMP_COL).is_between(start_time, now_naive, closed="left"))
                .sort(TIMESTAMP_COL, descending=True)
                .head(TARGET_ROWS)
                .collect()
            )
        except Exception as e:
            print(f"  Warning: scan_parquet failed for {name} (lookback={lookback}d): {e}")
            return None

        if bb_df.height >= TARGET_ROWS:
            print(f"  Collected {bb_df.height} rows ({lookback}d lookback)")
            break
    else:
        # Exhausted MAX_LOOKBACK_DAYS
        if bb_df.height == 0:
            return None
        print(
            f"  WARNING: only {bb_df.height}/{TARGET_ROWS} rows found after "
            f"{MAX_LOOKBACK_DAYS}d lookback for {name} — significant data gap likely"
        )

    quantiles = bb_df.select(
        pl.col(BB_COL).quantile(0.05).alias("bb_ref"),
        pl.col(COMM_BB_COL).quantile(0.05).alias("comm_bb_ref"),
        pl.col(SHIP_BB_COL).quantile(0.05).alias("ship_bb_ref"),
    ).row(0)

    return ReferenceRow(
        date=ref_date,
        bb_ref=float(quantiles[0]),
        comm_bb_ref=float(quantiles[1]),
        ship_bb_ref=float(quantiles[2]),
    )


def process_hydrophone(
    s3_client, hydrophone: Hydrophone, now: dt.datetime, tomorrow: dt.date, test_mode: bool = False
) -> int:
    """Process a single hydrophone. Computes one ref row for tomorrow. Returns 1 on success, 0 otherwise."""
    name = hydrophone.name
    print(f"\nProcessing {name}...")

    existing_df = read_existing(s3_client, hydrophone, test_mode)

    # Keep all rows except tomorrow (we'll recompute it)
    if existing_df is not None and len(existing_df) > 0:
        existing_df = existing_df.filter(pl.col("date") != tomorrow)
        print(f"  Existing ref has {len(existing_df)} rows (excluding tomorrow)")
    else:
        existing_df = None
        print("  No existing ref data found")

    # Check if we have enough data history for a rolling window
    earliest_date = find_earliest_data_date(s3_client, hydrophone)
    if earliest_date is None:
        print(f"  No broadband data found for {name}, skipping")
        return 0

    rolling_start_date = earliest_date + dt.timedelta(days=WINDOW_DAYS)
    if now.date() < rolling_start_date:
        # Not enough history — use 5th percentile of first WINDOW_DAYS days
        bootstrap_ref = compute_initial_reference(hydrophone, earliest_date)
        if bootstrap_ref is None:
            print(f"  Bootstrap computation failed for {name}, skipping")
            return 0
        row = ReferenceRow(
            date=tomorrow,
            bb_ref=bootstrap_ref[0],
            comm_bb_ref=bootstrap_ref[1],
            ship_bb_ref=bootstrap_ref[2],
        )
        print(f"  {tomorrow}: bb={row.bb_ref:.1f} comm={row.comm_bb_ref:.1f} "
              f"ship={row.ship_bb_ref:.1f} dB (bootstrap)")
    else:
        row = compute_reference(hydrophone, now, tomorrow)
        if row is None:
            print(f"  No data found for {name}, skipping")
            return 0
        print(f"  {tomorrow}: bb={row.bb_ref:.1f} comm={row.comm_bb_ref:.1f} "
              f"ship={row.ship_bb_ref:.1f} dB")

    new_df = pl.DataFrame([asdict(row)]).with_columns(
        pl.col("date").cast(pl.Date),
    )

    if existing_df is not None and len(existing_df) > 0:
        combined = pl.concat([existing_df, new_df])
    else:
        combined = new_df

    write_to_s3(s3_client, hydrophone, combined, test_mode)
    return 1


def main():
    parser = argparse.ArgumentParser(description="Compute rolling broadband reference levels")
    parser.add_argument("--test", action="store_true",
                        help=f"Write to test prefix ({TEST_PREFIX}/) instead of production paths")
    args = parser.parse_args()

    print("=" * 60)
    print("Rolling Broadband Reference Level Computation")
    if args.test:
        print(f"*** TEST MODE — writing to {TEST_PREFIX}/ ***")
    print("=" * 60)

    s3_client = boto3.client("s3")
    now = dt.datetime.now(TIMEZONE)
    tomorrow = (now + dt.timedelta(days=1)).date()
    print(f"Now: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Computing ref for: {tomorrow}")

    summary = {}
    for hydrophone in Hydrophone:
        if hydrophone.name == "HPhoneTup":
            continue
        try:
            count = process_hydrophone(s3_client, hydrophone, now, tomorrow, args.test)
            summary[hydrophone.name] = count
        except Exception:
            print(f"\nERROR processing {hydrophone.name}:")
            traceback.print_exc()
            summary[hydrophone.name] = None

    print("\n" + "=" * 60)
    print("Summary:")
    for name, result in summary.items():
        if result is None:
            print(f"  {name}: FAILED")
        else:
            print(f"  {name}: {result} new rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
