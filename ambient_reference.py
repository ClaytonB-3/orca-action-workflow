"""
Compute rolling 7-day broadband reference levels (5th percentile)
for all Orcasound hydrophones and write results to S3.

Produces per-hydrophone ref files matching the schema expected by
the pipeline (PR #79): date, bb_ref, comm_bb_ref, ship_bb_ref.
"""

import datetime as dt
import io
import traceback
from dataclasses import asdict, dataclass
from zoneinfo import ZoneInfo

import boto3
import polars as pl

from orcasound_noise.analysis.partitioned_accessor import PartitionedAccessor
from orcasound_noise.utils import Hydrophone

S3_BUCKET = "acoustic-sandbox"
WINDOW_DAYS = 7
TIMEZONE = ZoneInfo("US/Pacific")

# data_3.0 broadband columns
BB_COL = "bb_o"
COMM_BB_COL = "comm_bb_o"
SHIP_BB_COL = "ship_bb_o"


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


def s3_ref_key(hydrophone: Hydrophone) -> str:
    """S3 key matching the path the pipeline reads refs from."""
    name = hydrophone.value.name
    folder = hydrophone.value.save_folder
    return f"{folder}/ref/hydrophone={name}/{name}_refs.parquet"


def read_existing(s3_client, hydrophone: Hydrophone) -> pl.DataFrame | None:
    """Read existing reference parquet from S3, or return None."""
    key = s3_ref_key(hydrophone)
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = obj["Body"].read()
        return pl.read_parquet(io.BytesIO(data))
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"  Warning: could not read existing data for {hydrophone.name}: {e}")
        return None


def write_to_s3(s3_client, hydrophone: Hydrophone, df: pl.DataFrame) -> None:
    """Write reference parquet to S3."""
    key = s3_ref_key(hydrophone)
    buf = io.BytesIO()
    df.write_parquet(buf)
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Wrote {len(df)} rows to s3://{S3_BUCKET}/{key}")


def compute_reference_for_day(
    hydrophone: Hydrophone, target_date: dt.date
) -> ReferenceRow | None:
    """Compute 5th percentile for bb, comm, and ship bands over a 7-day window."""
    end_time = dt.datetime.combine(target_date + dt.timedelta(days=1), dt.time.min)
    start_time = end_time - dt.timedelta(days=WINDOW_DAYS)

    accessor = PartitionedAccessor(hydrophone, start_time, end_time)
    _, bb_df = accessor.get_dataframes()

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


def process_hydrophone(s3_client, hydrophone: Hydrophone, today: dt.date) -> int:
    """Process a single hydrophone. Returns number of new rows added."""
    name = hydrophone.name
    print(f"\nProcessing {name}...")

    existing_df = read_existing(s3_client, hydrophone)

    if existing_df is not None and len(existing_df) > 0:
        last_date = existing_df.select(pl.col("date").max()).item()
        if isinstance(last_date, dt.datetime):
            last_date = last_date.date()
        start_date = last_date + dt.timedelta(days=1)
        print(f"  Found existing data through {last_date}, computing from {start_date}")
    else:
        existing_df = None
        start_date = today - dt.timedelta(days=WINDOW_DAYS)
        print(f"  No existing data found, starting from {start_date}")

    if start_date > today:
        print("  Already up to date.")
        return 0

    new_rows = []
    for current in date_range(start_date, today):
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
        combined = pl.concat([existing_df, new_df])
    else:
        combined = new_df

    write_to_s3(s3_client, hydrophone, combined)
    return len(new_rows)


def main():
    print("=" * 60)
    print("Rolling Broadband Reference Level Computation")
    print("=" * 60)

    s3_client = boto3.client("s3")
    today = dt.datetime.now(TIMEZONE).date()
    print(f"Today: {today}")

    summary = {}
    for hydrophone in Hydrophone:
        try:
            count = process_hydrophone(s3_client, hydrophone, today)
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
