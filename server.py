"""Otokuna MCP Server — exposes Tokyo rental property deal-finding tools to LLMs."""

import io
import json
import os
import re
import datetime
from typing import Annotated

import boto3
import pandas as pd
from fastmcp import FastMCP
from pydantic import Field


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

BUCKET_NAME = os.environ.get("OTOKUNA_BUCKET", "otokuna-data-prod-1760689817")
SFN_REGION = os.environ.get("OTOKUNA_SFN_REGION", "ap-northeast-1")
SFN_ARN = os.environ.get("OTOKUNA_SFN_ARN", "")
S3_REGION = os.environ.get("OTOKUNA_S3_REGION", "ap-northeast-1")

# S3 key patterns (not secrets, safe to hardcode defaults)
SCRAPED_KEY_PREFIX = "dumped_data/daily"
SCRAPED_KEY_TEMPLATE = "{}/\u6771\u4eac\u90fd.pickle"  # {}/東京都.pickle
PREDICTIONS_KEY_PREFIX = "predictions/daily"
PREDICTION_KEY_TEMPLATE = "{}/prediction.pickle"
PREDICTION_KEY_PATTERN = r"(.*)/prediction\.pickle"

DEFAULT_COLUMNS = [
    "otokuna_score",
    "monthly_cost",
    "monthly_cost_predicted",
    "rent",
    "admin_fee",
    "deposit",
    "gratuity",
    "layout",
    "area",
    "building_age",
    "building_address",
    "ward",
    "building_title",
    "walk_time_station_min",
    "url",
]


mcp = FastMCP(
    "Otokuna",
    instructions=(
        "Otokuna is a Tokyo rental property deal-finder. "
        "It scores properties by comparing predicted vs actual monthly cost "
        "(otokuna_score > 1 means cheaper than expected — a good deal). "
        "Use the tools to list available data, trigger new searches from Suumo URLs, "
        "and retrieve property listings as CSV for analysis."
    ),
)


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------


def _s3_bucket():
    return boto3.resource("s3", region_name=S3_REGION).Bucket(BUCKET_NAME)


def _sfn_client():
    return boto3.client("stepfunctions", region_name=SFN_REGION)


def _download_dataframe(key: str) -> pd.DataFrame:
    with io.BytesIO() as stream:
        _s3_bucket().download_fileobj(Key=key, Fileobj=stream)
        stream.seek(0)
        return pd.read_pickle(stream)


def _join_dataframes(
    scraped_df: pd.DataFrame, prediction_df: pd.DataFrame
) -> pd.DataFrame:
    prediction_df = prediction_df.assign(otokuna_score=lambda df_: df_.y_pred / df_.y)
    df = prediction_df.join(scraped_df)
    df.sort_values(by="otokuna_score", ascending=False, inplace=True)
    df.rename(
        inplace=True,
        columns={"y": "monthly_cost", "y_pred": "monthly_cost_predicted"},
    )
    return df


def _iso2date(iso: str) -> str:
    return datetime.datetime.fromisoformat(iso).strftime("%Y-%m-%d")


def _resolve_columns(columns: str | None, df: pd.DataFrame) -> list[str]:
    if columns is None:
        return [c for c in DEFAULT_COLUMNS if c in df.columns]
    if columns == "all":
        return list(df.columns)
    requested = [c.strip() for c in columns.split(",")]
    return [c for c in requested if c in df.columns]


def _df_to_csv(
    df: pd.DataFrame,
    sort_by: str,
    ascending: bool,
    limit: int,
    columns: str | None,
) -> str:
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
    cols = _resolve_columns(columns, df)
    return df[cols].head(limit).to_csv()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
def list_daily_predictions() -> str:
    """List available dates that have daily property prediction data.

    Returns a JSON array of date strings (YYYY-MM-DD) sorted chronologically.
    Use these dates with get_daily_results to retrieve the actual property data.
    """
    bucket = _s3_bucket()
    prediction_objects = bucket.objects.filter(Prefix=PREDICTIONS_KEY_PREFIX)
    pattern = os.path.join(PREDICTIONS_KEY_PREFIX, PREDICTION_KEY_PATTERN)

    dates = set()
    for obj in prediction_objects:
        m = re.match(pattern, obj.key)
        if m:
            dates.add(_iso2date(m.group(1)))

    return json.dumps(sorted(dates))


@mcp.tool
def list_jobs() -> str:
    """List all custom request jobs that have completed.

    Returns a JSON array of job objects with: job_id, user_id, datetime, search_url,
    search_conditions. Use job_id with get_job_results to retrieve the property data.
    """
    bucket = _s3_bucket()
    jobs = []
    for obj in bucket.objects.filter(Prefix="jobs"):
        if not obj.key.endswith("job_info.json"):
            continue
        info = json.loads(obj.get()["Body"].read())
        jst = datetime.timezone(datetime.timedelta(hours=9))
        dt = datetime.datetime.fromtimestamp(info["timestamp"], tz=jst)
        jobs.append(
            {
                "job_id": info["job_id"],
                "user_id": info["user_id"],
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S JST"),
                "search_url": info["search_url"],
                "search_conditions": info["search_conditions"],
            }
        )

    jobs.sort(key=lambda j: j["datetime"])
    return json.dumps(jobs, ensure_ascii=False)


@mcp.tool
def submit_request(
    suumo_url: Annotated[
        str, Field(description="Suumo search URL to scrape and predict")
    ],
    user_id: Annotated[
        str, Field(description="User identifier for the request")
    ] = "mcp",
) -> str:
    """Submit a Suumo search URL to trigger the scraping and prediction pipeline.

    The pipeline scrapes all listings matching the URL, predicts expected monthly
    costs using an ML model, and scores each property. This takes a few minutes.
    Use get_job_status with the returned execution_arn to check progress.
    """
    if not SFN_ARN:
        return json.dumps({"error": "OTOKUNA_SFN_ARN not configured"})

    input_data = {"user_id": user_id, "search_url": suumo_url}
    response = _sfn_client().start_execution(
        stateMachineArn=SFN_ARN,
        input=json.dumps(input_data),
    )
    return json.dumps(
        {
            "execution_arn": response["executionArn"],
            "started_at": response["startDate"].isoformat(),
            "status": "RUNNING",
        }
    )


@mcp.tool
def get_job_status(
    execution_arn: Annotated[
        str, Field(description="The execution ARN returned by submit_request")
    ],
) -> str:
    """Check the status of a submitted pipeline execution.

    Returns status (RUNNING, SUCCEEDED, FAILED, TIMED_OUT, ABORTED) and timing info.
    When SUCCEEDED, use list_jobs to find the job_id, then get_job_results for data.
    """
    response = _sfn_client().describe_execution(executionArn=execution_arn)
    result = {
        "status": response["status"],
        "started_at": response["startDate"].isoformat(),
    }
    if "stopDate" in response:
        result["stopped_at"] = response["stopDate"].isoformat()
    return json.dumps(result)


@mcp.tool
def get_daily_results(
    date: Annotated[
        str, Field(description="Date in YYYY-MM-DD format from list_daily_predictions")
    ],
    sort_by: Annotated[str, Field(description="Column to sort by")] = "otokuna_score",
    ascending: Annotated[
        bool, Field(description="Sort ascending if true, descending if false")
    ] = False,
    limit: Annotated[int, Field(description="Max rows to return", ge=1, le=500)] = 50,
    columns: Annotated[
        str | None,
        Field(
            description="Comma-separated column names, 'all' for everything, or omit for default set"
        ),
    ] = None,
) -> str:
    """Get daily property prediction results as CSV.

    Returns a CSV string with property listings for the given date. Each row is a
    rental property with its otokuna_score (>1 = good deal), monthly costs, location,
    layout, and Suumo listing URL. Default sort is by otokuna_score descending (best deals first).
    """
    # Find the ISO datetime for this date
    bucket = _s3_bucket()
    prediction_objects = bucket.objects.filter(Prefix=PREDICTIONS_KEY_PREFIX)
    pattern = os.path.join(PREDICTIONS_KEY_PREFIX, PREDICTION_KEY_PATTERN)

    iso_datetime = None
    for obj in prediction_objects:
        m = re.match(pattern, obj.key)
        if m and _iso2date(m.group(1)) == date:
            iso_datetime = m.group(1)

    if iso_datetime is None:
        return f"No data found for date: {date}"

    scraped_key = os.path.join(SCRAPED_KEY_PREFIX, SCRAPED_KEY_TEMPLATE).format(
        iso_datetime
    )
    prediction_key = os.path.join(
        PREDICTIONS_KEY_PREFIX, PREDICTION_KEY_TEMPLATE
    ).format(iso_datetime)

    scraped_df = _download_dataframe(scraped_key)
    prediction_df = _download_dataframe(prediction_key)
    df = _join_dataframes(scraped_df, prediction_df)

    return _df_to_csv(df, sort_by, ascending, limit, columns)


@mcp.tool
def get_job_results(
    job_id: Annotated[str, Field(description="Job ID from list_jobs")],
    sort_by: Annotated[str, Field(description="Column to sort by")] = "otokuna_score",
    ascending: Annotated[
        bool, Field(description="Sort ascending if true, descending if false")
    ] = False,
    limit: Annotated[int, Field(description="Max rows to return", ge=1, le=500)] = 50,
    columns: Annotated[
        str | None,
        Field(
            description="Comma-separated column names, 'all' for everything, or omit for default set"
        ),
    ] = None,
) -> str:
    """Get custom request prediction results as CSV.

    Returns a CSV string with property listings for the given job. Each row is a
    rental property with its otokuna_score (>1 = good deal), monthly costs, location,
    layout, and Suumo listing URL. Default sort is by otokuna_score descending (best deals first).
    """
    # Find job info from S3
    bucket = _s3_bucket()
    job_info = None
    for obj in bucket.objects.filter(Prefix=f"jobs/{job_id}"):
        if obj.key.endswith("job_info.json"):
            job_info = json.loads(obj.get()["Body"].read())
            break

    if job_info is None:
        # Try scanning all jobs
        for obj in bucket.objects.filter(Prefix="jobs"):
            if not obj.key.endswith("job_info.json"):
                continue
            info = json.loads(obj.get()["Body"].read())
            if info["job_id"] == job_id:
                job_info = info
                break

    if job_info is None:
        return f"No data found for job_id: {job_id}"

    scraped_df = _download_dataframe(job_info["scraped_data_key"])
    prediction_df = _download_dataframe(job_info["prediction_data_key"])
    df = _join_dataframes(scraped_df, prediction_df)

    return _df_to_csv(df, sort_by, ascending, limit, columns)
