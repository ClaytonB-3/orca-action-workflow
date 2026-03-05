"""
Compute rolling 7-day broadband reference levels (5th percentile)
for all Orcasound hydrophones and write results to S3.
"""

import datetime as dt
import io
import traceback
from dataclasses import asdict, dataclass
from zoneinfo import ZoneInfo

import boto3
import polars as pl

from orcasound_noise.pipeline.partitioned_accessor import PartitionedAccessor
from orcasound_noise.utils import Hydrophone

S3_BUCKET = "acoustic-sandbox"
S3_PREFIX = "ambient-sound-analysis/data/rolling_reference"
WINDOW_DAYS = 7
TIMEZONE = ZoneInfo("US/Pacific")


@dataclass
class ReferenceRow:
    date: dt.date
    hydrophone: str
    rolling_reference_db: float
    sample_count: int
    window_start: dt.datetime
    window_end: dt.datetime


def date_range(start: dt.date, end: dt.date):
    """Yield dates from start through end inclusive."""
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def s3_key(hydrophone_name: str) -> str:
    return f"{S3_PREFIX}/{hydrophone_name}_reference.parquet"


def read_existing(s3_client, hydrophone_name: str) -> pl.DataFrame | None:
    """Read existing reference parquet from S3, or return None."""
    key = s3_key(hydrophone_name)
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = obj["Body"].read()
        return pl.read_parquet(io.BytesIO(data))
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"  Warning: could not read existing data for {hydrophone_name}: {e}")
        return None


def write_to_s3(s3_client, hydrophone_name: str, df: pl.DataFrame) -> None:
    """Write reference parquet to S3."""
    key = s3_key(hydrophone_name)
    buf = io.BytesIO()
    df.write_parquet(buf)
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Wrote {len(df)} rows to s3://{S3_BUCKET}/{key}")


def find_earliest_data(hydrophone: Hydrophone, today: dt.date) -> dt.date | None:
    """Find earliest date with available data by probing progressively older dates."""
    # Try going back up to 2 years in weekly steps, then narrow down
    earliest_found = None

    # First: coarse scan backwards in 30-day steps up to 730 days
    for days_back in range(0, 731, 30):
        candidate = today - dt.timedelta(days=days_back)
        start = dt.datetime.combine(candidate, dt.time.min).replace(tzinfo=TIMEZONE)
        end = start + dt.timedelta(days=1)
        try:
            accessor = PartitionedAccessor(hydrophone, start, end)
            _, bb_lazy = accessor.get_dataframes()
            bb_df = bb_lazy.collect()
            if bb_df.height > 0:
                earliest_found = candidate
        except Exception as e:
            print(f"  Probe {candidate}: {e}")

    if earliest_found is None:
        return None

    # Fine-tune: scan forward from (earliest_found - 30 days) in daily steps
    search_start = earliest_found - dt.timedelta(days=30)
    for day_offset in range(31):
        candidate = search_start + dt.timedelta(days=day_offset)
        if candidate > today:
            break
        start = dt.datetime.combine(candidate, dt.time.min).replace(tzinfo=TIMEZONE)
        end = start + dt.timedelta(days=1)
        try:
            accessor = PartitionedAccessor(hydrophone, start, end)
            _, bb_lazy = accessor.get_dataframes()
            bb_df = bb_lazy.collect()
            if bb_df.height > 0:
                return candidate
        except Exception as e:
            print(f"  Probe {candidate}: {e}")
            continue

    return earliest_found


def compute_reference_for_day(
    hydrophone: Hydrophone, target_date: dt.date
) -> ReferenceRow | None:
    """Compute 5th percentile broadband dB for 7-day window ending on target_date."""
    end_time = dt.datetime.combine(
        target_date + dt.timedelta(days=1), dt.time.min
    ).replace(tzinfo=TIMEZONE)
    start_time = end_time - dt.timedelta(days=WINDOW_DAYS)

    accessor = PartitionedAccessor(hydrophone, start_time, end_time)
    _, bb_lazy = accessor.get_dataframes()
    bb_df = bb_lazy.collect()

    if bb_df.height == 0:
        return None

    ref_db = bb_df.select(pl.col("0").quantile(0.05)).item()

    return ReferenceRow(
        date=target_date,
        hydrophone=hydrophone.name,
        rolling_reference_db=float(ref_db),
        sample_count=bb_df.height,
        window_start=start_time.replace(tzinfo=None),
        window_end=end_time.replace(tzinfo=None),
    )


def process_hydrophone(s3_client, hydrophone: Hydrophone, today: dt.date) -> int:
    """Process a single hydrophone. Returns number of new rows added."""
    name = hydrophone.name
    print(f"\nProcessing {name}...")

    existing_df = read_existing(s3_client, name)

    if existing_df is not None and len(existing_df) > 0:
        last_date = existing_df.select(pl.col("date").max()).item()
        if isinstance(last_date, dt.datetime):
            last_date = last_date.date()
        start_date = last_date + dt.timedelta(days=1)
        print(f"  Found existing data through {last_date}, computing from {start_date}")
    else:
        existing_df = None
        print("  No existing data found, searching for earliest available data...")
        earliest = find_earliest_data(hydrophone, today)
        if earliest is None:
            print(f"  No data found for {name}, skipping.")
            return 0
        # Need at least WINDOW_DAYS of data for a meaningful reference
        start_date = earliest + dt.timedelta(days=WINDOW_DAYS)
        print(f"  Earliest data: {earliest}, starting reference from {start_date}")

    if start_date > today:
        print("  Already up to date.")
        return 0

    new_rows = []
    for current in date_range(start_date, today):
        try:
            row = compute_reference_for_day(hydrophone, current)
            if row is not None:
                new_rows.append(row)
                print(f"  {current}: {row.rolling_reference_db:.1f} dB ({row.sample_count} samples)")
            else:
                print(f"  {current}: no data in window, skipping")
        except Exception as e:
            print(f"  {current}: error - {e}")

    if not new_rows:
        print(f"  No new data computed for {name}.")
        return 0

    new_df = pl.DataFrame([asdict(row) for row in new_rows]).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("sample_count").cast(pl.Int64),
        pl.col("window_start").cast(pl.Datetime),
        pl.col("window_end").cast(pl.Datetime),
    )

    if existing_df is not None:
        combined = pl.concat([existing_df, new_df])
    else:
        combined = new_df

    write_to_s3(s3_client, name, combined)
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
