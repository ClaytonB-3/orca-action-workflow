"""
Compute rolling 7-day broadband reference levels (5th percentile)
for all Orcasound hydrophones and write results to S3.

Produces per-hydrophone ref files matching the schema expected by
the pipeline (PR #79): date, bb_ref, comm_bb_ref, ship_bb_ref.
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

S3_BUCKET = "acoustic-sandbox"
S3_DATA_PREFIX = "ambient-sound-analysis/data_3.0"
TEST_PREFIX = "test/ambient_reference"
WINDOW_DAYS = 7
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


def date_range(start: dt.date, end: dt.date):
    """Yield dates from start through end inclusive."""
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def s3_ref_key(hydrophone: Hydrophone, test_mode: bool = False) -> str:
    """S3 key matching the path the pipeline reads refs from."""
    name = hydrophone.value.name
    key = f"{S3_DATA_PREFIX}/ref/hydrophone={name}/{name}_ref.parquet"
    if test_mode:
        key = f"{TEST_PREFIX}/{key}"
    return key


def bb_paths_for_range(hydrophone: Hydrophone, start: dt.date, end: dt.date) -> list[str]:
    """Build S3 glob paths for broadband parquets over a date range."""
    name = hydrophone.value.name
    paths = []
    d = start
    while d <= end:
        paths.append(
            f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}/broadband/hydrophone={name}"
            f"/year={d.year}/month={d.month:02d}/day={d.day:02d}/*.parquet"
        )
        d += dt.timedelta(days=1)
    return paths


def read_existing(s3_client, hydrophone: Hydrophone, test_mode: bool = False) -> pl.DataFrame | None:
    """Read existing reference parquet from S3, or return None."""
    key = s3_ref_key(hydrophone, test_mode)
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = obj["Body"].read()
        return pl.read_parquet(io.BytesIO(data))
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"  Warning: could not read existing data for {hydrophone.name}: {e}")
        return None


def write_to_s3(s3_client, hydrophone: Hydrophone, df: pl.DataFrame, test_mode: bool = False) -> None:
    """Write reference parquet to S3."""
    key = s3_ref_key(hydrophone, test_mode)
    buf = io.BytesIO()
    df.write_parquet(buf)
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Wrote {len(df)} rows to s3://{S3_BUCKET}/{key}")


def find_earliest_data_date(s3_client, hydrophone: Hydrophone) -> dt.date | None:
    """Find the earliest date with broadband data via S3 listing."""
    name = hydrophone.value.name
    prefix = f"{S3_DATA_PREFIX}/broadband/hydrophone={name}/year="
    resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=1)
    if resp.get("KeyCount", 0) == 0:
        return None
    # First key is the earliest (S3 lists lexicographically)
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


def compute_bootstrap_reference(
    hydrophone: Hydrophone, earliest_date: dt.date
) -> tuple[float, float, float] | None:
    """Compute 5th percentile over the first WINDOW_DAYS days of data."""
    start_time = dt.datetime.combine(earliest_date, dt.time.min)
    end_time = dt.datetime.combine(earliest_date + dt.timedelta(days=WINDOW_DAYS), dt.time.min)

    paths = bb_paths_for_range(hydrophone, earliest_date, earliest_date + dt.timedelta(days=WINDOW_DAYS - 1))
    try:
        bb_df = (
            pl.scan_parquet(paths, storage_options={"aws_region": "us-west-2"})
            .filter(pl.col(TIMESTAMP_COL).is_between(start_time, end_time))
            .collect()
        )
    except Exception:
        return None

    if bb_df.height == 0:
        return None

    quantiles = bb_df.select(
        pl.col(BB_COL).quantile(0.05).alias("bb_ref"),
        pl.col(COMM_BB_COL).quantile(0.05).alias("comm_bb_ref"),
        pl.col(SHIP_BB_COL).quantile(0.05).alias("ship_bb_ref"),
    ).row(0)

    return (float(quantiles[0]), float(quantiles[1]), float(quantiles[2]))


def compute_reference_for_day(
    hydrophone: Hydrophone, target_date: dt.date
) -> ReferenceRow | None:
    """Compute 5th percentile for bb, comm, and ship bands over a 7-day window."""
    end_time = dt.datetime.combine(target_date + dt.timedelta(days=1), dt.time.min)
    start_time = end_time - dt.timedelta(days=WINDOW_DAYS)

    paths = bb_paths_for_range(hydrophone, start_time.date(), end_time.date())
    try:
        bb_df = (
            pl.scan_parquet(paths, storage_options={"aws_region": "us-west-2"})
            .filter(pl.col(TIMESTAMP_COL).is_between(start_time, end_time))
            .collect()
        )
    except Exception:
        return None

    if bb_df.height == 0:
        return None

    quantiles = bb_df.select(
        pl.col(BB_COL).quantile(0.05).alias("bb_ref"),
        pl.col(COMM_BB_COL).quantile(0.05).alias("comm_bb_ref"),
        pl.col(SHIP_BB_COL).quantile(0.05).alias("ship_bb_ref"),
    ).row(0)

    return ReferenceRow(
        date=target_date,
        bb_ref=float(quantiles[0]),
        comm_bb_ref=float(quantiles[1]),
        ship_bb_ref=float(quantiles[2]),
    )


def process_hydrophone(s3_client, hydrophone: Hydrophone, today: dt.date, test_mode: bool = False) -> int:
    """Process a single hydrophone. Returns number of new rows added."""
    name = hydrophone.name
    print(f"\nProcessing {name}...")

    existing_df = read_existing(s3_client, hydrophone, test_mode)

    if existing_df is not None and len(existing_df) > 0:
        # Keep only historical rows (before today); we'll recompute today and forward
        existing_df = existing_df.filter(pl.col("date") < today)
        if len(existing_df) > 0:
            last_date = existing_df.select(pl.col("date").max()).item()
            if isinstance(last_date, dt.datetime):
                last_date = last_date.date()
            start_date = last_date + dt.timedelta(days=1)
            print(f"  Found existing data through {last_date}, computing from {start_date}")
        else:
            existing_df = None
            start_date = today - dt.timedelta(days=WINDOW_DAYS)
            print(f"  No historical data before today, starting from {start_date}")
    else:
        existing_df = None
        start_date = today - dt.timedelta(days=WINDOW_DAYS)
        print(f"  No existing data found, starting from {start_date}")

    if start_date > today:
        print("  Already up to date.")
        return 0

    # Find earliest data date and compute bootstrap reference for early days
    earliest_date = find_earliest_data_date(s3_client, hydrophone)
    bootstrap_ref = None
    rolling_start_date = None
    if earliest_date is not None:
        rolling_start_date = earliest_date + dt.timedelta(days=WINDOW_DAYS)
        if start_date < rolling_start_date:
            bootstrap_ref = compute_bootstrap_reference(hydrophone, earliest_date)
            if bootstrap_ref:
                print(f"  Bootstrap ref (first {WINDOW_DAYS} days from {earliest_date}): "
                      f"bb={bootstrap_ref[0]:.1f} comm={bootstrap_ref[1]:.1f} ship={bootstrap_ref[2]:.1f} dB")

    new_rows = []
    for current in date_range(start_date, today):
        try:
            if rolling_start_date and current < rolling_start_date and bootstrap_ref:
                row = ReferenceRow(
                    date=current,
                    bb_ref=bootstrap_ref[0],
                    comm_bb_ref=bootstrap_ref[1],
                    ship_bb_ref=bootstrap_ref[2],
                )
                new_rows.append(row)
                print(f"  {current}: bb={row.bb_ref:.1f} comm={row.comm_bb_ref:.1f} ship={row.ship_bb_ref:.1f} dB (bootstrap)")
            else:
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
        combined = pl.concat([existing_df, new_df])
    else:
        combined = new_df

    write_to_s3(s3_client, hydrophone, combined, test_mode)
    return len(new_rows)


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
    today = dt.datetime.now(TIMEZONE).date()
    print(f"Today: {today}")

    summary = {}
    for hydrophone in Hydrophone:
        if hydrophone.name == "HPhoneTup":
            continue
        try:
            count = process_hydrophone(s3_client, hydrophone, today, args.test)
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
