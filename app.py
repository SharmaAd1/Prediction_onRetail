import json
import math
import os
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st
from google import genai
from google.genai.errors import APIError
from google.genai import types
from prophet import Prophet

from db_connect import (
    DatabaseConnectionError,
    load_retail_sales,
    mysql_is_configured,
)


DATA_FILE = Path(__file__).with_name("retail_data_c.csv")
REQUIRED_COLUMNS = {
    "InvoiceDate",
    "CustomerID",
    "ProductID",
    "ProductName",
    "Category",
    "Quantity",
    "UnitPrice",
    "StockQuantity",
    "Region",
    "Revenue",
}
GEMINI_MODEL = "gemini-3.5-flash-lite"

st.set_page_config(
    page_title="Retail Revenue Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

dark_mode = st.sidebar.toggle("🌙  Dark mode", value=False, key="dark_mode")
THEME = {
    "canvas": "#0d1526" if dark_mode else "#edf3ff",
    "ink": "#eef3ff" if dark_mode else "#17233c",
    "muted": "#a5b2ca" if dark_mode else "#63718b",
    "surface": "rgba(27, 38, 59, .92)" if dark_mode else "rgba(255, 255, 255, .86)",
    "border": "rgba(141, 160, 191, .24)" if dark_mode else "rgba(83, 107, 151, .18)",
    "shadow": "rgba(0, 0, 0, .28)" if dark_mode else "rgba(40, 70, 140, .12)",
}

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: __CANVAS__ !important;
        color: __INK__ !important;
    }
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMain"], section.main {
        background: transparent !important;
        position: relative;
        z-index: 1;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: -20%;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(ellipse at 18% 18%, rgba(56, 103, 244, .18), transparent 34%),
            radial-gradient(ellipse at 82% 72%, rgba(18, 165, 148, .14), transparent 32%);
        animation: ambientDrift 18s ease-in-out infinite alternate;
    }
    [data-testid="stHeader"] {
        background: __CANVAS__ !important;
        background: color-mix(in srgb, __CANVAS__ 78%, transparent) !important;
        backdrop-filter: blur(14px);
    }
    [data-testid="stSidebar"] {
        background: __SURFACE__ !important;
        backdrop-filter: blur(18px);
        border-right: 1px solid __BORDER__;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    .block-container {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, label, [data-testid="stCaptionContainer"] {
        color: __INK__;
    }
    [data-testid="stCaptionContainer"] { color: __MUTED__ !important; }
    .block-container h2, .block-container h3 {
        letter-spacing: -.025em;
    }
    .block-container h2, .block-container h3 {
        position: relative;
        padding-left: 14px;
    }
    .block-container h2::before, .block-container h3::before {
        content: "";
        position: absolute;
        left: 0;
        top: .22em;
        bottom: .22em;
        width: 4px;
        border-radius: 4px;
        background: linear-gradient(180deg, #3867f4, #12a594);
    }
    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        background: __SURFACE__ !important;
        backdrop-filter: blur(14px);
        border: 1px solid __BORDER__;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px __SHADOW__;
        transition: transform .24s ease, box-shadow .24s ease, border-color .24s ease;
    }
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        inset: 0 0 auto;
        height: 3px;
        background: linear-gradient(90deg, #3867f4, #12a594);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: #3867f4;
        box-shadow: 0 16px 36px __SHADOW__;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
        color: __MUTED__ !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
        color: __INK__ !important;
    }
    [data-testid="stMetricValue"] div {
        font-size: 1.65rem !important;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }
    [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
        box-sizing: border-box;
        width: 100%;
        max-width: 100%;
        background: __SURFACE__;
        border: 1px solid __BORDER__;
        border-radius: 18px;
        padding: 10px;
        overflow: hidden;
        box-shadow: 0 8px 24px __SHADOW__;
        transition: transform .24s ease, box-shadow .24s ease;
    }
    [data-testid="stVegaLiteChart"] .vega-embed,
    [data-testid="stVegaLiteChart"] canvas,
    [data-testid="stVegaLiteChart"] svg {
        max-width: 100% !important;
    }
    [data-testid="stVegaLiteChart"]:hover, [data-testid="stArrowVegaLiteChart"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 36px __SHADOW__;
    }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-testid="stDateInput"] input, [data-testid="stDateInput"] button {
        background-color: __SURFACE__ !important;
        border-color: __BORDER__ !important;
        color: __INK__ !important;
        border-radius: 12px !important;
    }
    .stButton button {
        border: 0;
        border-radius: 12px;
        background: linear-gradient(135deg, #3867f4, #12a594);
        color: white;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(56, 103, 244, .35);
        color: white;
    }
    .retail-hero {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        min-height: 230px;
        padding: 32px 36px;
        border: 1px solid __BORDER__;
        border-radius: 24px;
        background: linear-gradient(120deg, __SURFACE__, __CANVAS__);
        box-shadow: 0 14px 40px __SHADOW__;
        animation: heroRise .8s cubic-bezier(.2, .8, .2, 1) both;
    }
    .retail-hero::before, .retail-hero::after {
        content: "";
        position: absolute;
        z-index: -1;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        filter: blur(48px);
        opacity: .42;
        animation: glowFloat 9s ease-in-out infinite alternate;
    }
    .retail-hero::before {
        top: -145px;
        left: 12%;
        background: #3867f4;
    }
    .retail-hero::after {
        right: 7%;
        bottom: -170px;
        background: #12a594;
        animation-delay: -4s;
    }
    .hero-copy { position: relative; z-index: 1; max-width: 760px; }
    .hero-kicker {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #12a594;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .hero-pulse {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #12a594;
        box-shadow: 0 0 14px #12a594;
        animation: pulse 1.8s ease-out infinite;
    }
    .retail-hero h1 {
        margin: 14px 0 10px;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.08;
        letter-spacing: -.045em;
        background: linear-gradient(90deg, __INK__, #3867f4 58%, #12a594);
        background-size: 200% 100%;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleShimmer 8s ease-in-out infinite alternate;
    }
    .retail-hero p { max-width: 650px; margin: 0; color: __MUTED__; line-height: 1.6; }
    .hero-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
    .hero-chip {
        padding: 6px 12px;
        border: 1px solid __BORDER__;
        border-radius: 999px;
        background: __SURFACE__;
        color: __INK__;
        font-size: .75rem;
        font-weight: 600;
        backdrop-filter: blur(8px);
    }
    .hero-orbit {
        position: absolute;
        top: 30px;
        right: 7%;
        width: 170px;
        height: 170px;
        pointer-events: none;
        animation: orbitFloat 7s ease-in-out infinite alternate;
    }
    .hero-orbit span {
        position: absolute;
        inset: 0;
        border: 1px solid rgba(56, 103, 244, .42);
        border-radius: 50%;
        transform: rotateX(68deg);
    }
    .hero-orbit span:nth-child(2) {
        inset: 20px;
        border-color: rgba(18, 165, 148, .56);
        transform: rotateY(68deg);
    }
    .hero-orbit span:nth-child(3) {
        inset: 42px;
        border-color: rgba(56, 103, 244, .62);
        transform: rotateX(68deg) rotateY(24deg);
    }
    .hero-orbit i {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3867f4, #12a594);
        box-shadow: 0 0 18px #3867f4;
    }
    @keyframes ambientDrift {
        from { transform: translate3d(-1%, -1%, 0) scale(1); }
        to { transform: translate3d(1%, 1%, 0) scale(1.05); }
    }
    @keyframes heroRise {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes glowFloat {
        from { transform: translate3d(-10px, -6px, 0) scale(.92); }
        to { transform: translate3d(18px, 12px, 0) scale(1.08); }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(18, 165, 148, .55); }
        70%, 100% { box-shadow: 0 0 0 9px rgba(18, 165, 148, 0); }
    }
    @keyframes orbitFloat {
        from { transform: translateY(-4px) rotate(-5deg); }
        to { transform: translateY(8px) rotate(5deg); }
    }
    @keyframes titleShimmer { to { background-position: 100% 0; } }
    @media (max-width: 700px) {
        .retail-hero { min-height: 0; padding: 25px 22px; }
        .hero-orbit { right: -58px; opacity: .45; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """
    .replace("__CANVAS__", THEME["canvas"])
    .replace("__INK__", THEME["ink"])
    .replace("__MUTED__", THEME["muted"])
    .replace("__SURFACE__", THEME["surface"])
    .replace("__BORDER__", THEME["border"])
    .replace("__SHADOW__", THEME["shadow"]),
    unsafe_allow_html=True,
)

@st.cache_data
def load_sales_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The sales CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="raise")
    data["Revenue"] = pd.to_numeric(data["Revenue"], errors="raise")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="raise")
    return data


@st.cache_data(ttl=300)
def load_database_sales_data() -> pd.DataFrame:
    data = load_retail_sales()
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The MySQL sales table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="raise")
    data["Revenue"] = pd.to_numeric(data["Revenue"], errors="raise")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="raise")
    return data


@st.cache_data(show_spinner="Building your revenue forecast...")
def make_forecast(
    data: pd.DataFrame,
    category: str,
    region: str,
    forecast_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = data
    if category != "All categories":
        selected = selected.loc[selected["Category"] == category]
    if region != "All regions":
        selected = selected.loc[selected["Region"] == region]
    if selected.empty:
        raise ValueError("There are no sales records for this category and region.")

    daily = (
        selected.groupby("InvoiceDate", as_index=False)["Revenue"]
        .sum()
        .rename(columns={"InvoiceDate": "ds", "Revenue": "y"})
        .sort_values("ds")
    )
    if len(daily) < 2:
        raise ValueError("Select a date range containing at least two sales days.")

    calendar = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = (
        daily.set_index("ds")
        .reindex(calendar, fill_value=0)
        .rename_axis("ds")
        .reset_index()
    )

    model = Prophet(
        daily_seasonality=False,
        interval_width=0.8,
    )
    model.fit(daily)
    prediction = model.predict(
        model.make_future_dataframe(periods=forecast_days, freq="D")
    )
    future = prediction.loc[
        prediction["ds"] > daily["ds"].max(),
        ["ds", "yhat", "yhat_lower", "yhat_upper"],
    ].copy()
    if len(future) != forecast_days:
        raise ValueError(
            f"Expected {forecast_days} forecast days, got {len(future)}."
        )
    if not all(
        math.isfinite(value)
        for value in future[["yhat", "yhat_lower", "yhat_upper"]].to_numpy().flat
    ):
        raise ValueError("The forecast contains non-finite revenue estimates.")
    future[["yhat", "yhat_lower", "yhat_upper"]] = future[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)
    return daily, future


def format_indian_number(value: float, decimals: int = 0) -> str:
    formatted = f"{abs(value):.{decimals}f}"
    integer, _, fraction = formatted.partition(".")
    if len(integer) > 3:
        groups = [integer[-3:]]
        integer = integer[:-3]
        while integer:
            groups.insert(0, integer[-2:])
            integer = integer[:-2]
        integer = ",".join(groups)
    sign = "-" if value < 0 else ""
    return f"{sign}{integer}.{fraction}" if fraction else f"{sign}{integer}"


def format_indian_currency(value: float, decimals: int = 0) -> str:
    return f"₹{format_indian_number(value, decimals)}"


def query_retail_sales(
    sales_data: pd.DataFrame,
    operation: str,
    category: Optional[str] = None,
    region: Optional[str] = None,
    product_id: Optional[str] = None,
    product_name: Optional[str] = None,
    customer_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    group_by: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, object]:
    if operation not in {"summary", "group_totals", "records"}:
        raise ValueError(f"Unsupported sales query operation: {operation}")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")
    if offset < 0:
        raise ValueError("offset must be zero or greater.")

    filtered = sales_data
    for column, value in (
        ("Category", category),
        ("Region", region),
        ("ProductID", product_id),
        ("ProductName", product_name),
        ("CustomerID", customer_id),
    ):
        if value is not None:
            filtered = filtered.loc[
                filtered[column].astype(str).str.casefold() == value.casefold()
            ]
    for column, value in (("start_date", start_date), ("end_date", end_date)):
        if value is not None:
            try:
                date_value = pd.Timestamp(value).date()
            except (TypeError, ValueError) as error:
                raise ValueError(f"{column} must be a valid date.") from error
            dates = filtered["InvoiceDate"].dt.date
            filtered = filtered.loc[
                dates >= date_value if column == "start_date" else dates <= date_value
            ]

    if filtered.empty:
        return {"row_count": 0, "message": "No matching sales records."}

    if operation == "summary":
        return {
            "row_count": len(filtered),
            "date_range": [
                filtered["InvoiceDate"].min().strftime("%Y-%m-%d"),
                filtered["InvoiceDate"].max().strftime("%Y-%m-%d"),
            ],
            "categories": sorted(filtered["Category"].dropna().unique().tolist()),
            "regions": sorted(filtered["Region"].dropna().unique().tolist()),
            "transaction_count": len(filtered),
            "unique_customers": int(filtered["CustomerID"].nunique()),
            "total_revenue": float(filtered["Revenue"].sum()),
            "average_revenue_per_transaction": float(filtered["Revenue"].mean()),
            "total_quantity": float(filtered["Quantity"].sum()),
        }

    if operation == "group_totals":
        allowed_groups = {"Category", "Region", "ProductName", "ProductID", "CustomerID"}
        if group_by not in allowed_groups:
            raise ValueError(
                f"group_by must be one of: {', '.join(sorted(allowed_groups))}."
            )
        grouped = (
            filtered.groupby(group_by, as_index=False, dropna=False)
            .agg(
                transaction_count=("Revenue", "size"),
                total_revenue=("Revenue", "sum"),
                total_quantity=("Quantity", "sum"),
            )
            .sort_values("total_revenue", ascending=False)
        )
        return {
            "total_groups": len(grouped),
            "offset": offset,
            "limit": limit,
            "groups": json.loads(
                grouped.iloc[offset : offset + limit].to_json(orient="records")
            ),
        }

    return {
        "total_matching_rows": len(filtered),
        "offset": offset,
        "limit": limit,
        "records": json.loads(
            filtered.iloc[offset : offset + limit].to_json(
                orient="records", date_format="iso"
            )
        ),
    }


RETAIL_SALES_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="query_retail_sales",
            description=(
                "Read retail sales rows and aggregates. Use for questions about "
                "revenue, transactions, customers, products, categories, regions, "
                "or dates. This tool is read-only."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "operation": types.Schema(
                        type=types.Type.STRING,
                        enum=["summary", "group_totals", "records"],
                        description=(
                            "summary for metrics, group_totals for totals by one "
                            "field, records for raw matching rows."
                        ),
                    ),
                    "category": types.Schema(type=types.Type.STRING),
                    "region": types.Schema(type=types.Type.STRING),
                    "product_id": types.Schema(type=types.Type.STRING),
                    "product_name": types.Schema(type=types.Type.STRING),
                    "customer_id": types.Schema(type=types.Type.STRING),
                    "start_date": types.Schema(
                        type=types.Type.STRING,
                        description="Optional inclusive start date, YYYY-MM-DD.",
                    ),
                    "end_date": types.Schema(
                        type=types.Type.STRING,
                        description="Optional inclusive end date, YYYY-MM-DD.",
                    ),
                    "group_by": types.Schema(
                        type=types.Type.STRING,
                        enum=[
                            "Category",
                            "Region",
                            "ProductName",
                            "ProductID",
                            "CustomerID",
                        ],
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Page size from 1 to 100.",
                    ),
                    "offset": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of matching rows or groups to skip.",
                    ),
                },
                required=["operation"],
            ),
        )
    ]
)


st.markdown(
    """
    <section class="retail-hero">
      <div class="hero-copy">
        <div class="hero-kicker"><span class="hero-pulse"></span> Retail performance overview</div>
        <h1>Retail Revenue Intelligence</h1>
        <p>Explore sales performance, track category and regional trends, and forecast daily revenue.</p>
        <div class="hero-chips">
          <span class="hero-chip">Revenue trends</span>
          <span class="hero-chip">Category insights</span>
          <span class="hero-chip">Prophet forecast</span>
        </div>
      </div>
      <div class="hero-orbit" aria-hidden="true"><span></span><span></span><span></span><i></i></div>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    use_mysql = mysql_is_configured()
except DatabaseConnectionError as error:
    st.error(f"Invalid MySQL configuration: {error}")
    st.stop()

if not DATA_FILE.is_file() and not use_mysql:
    st.error(f"Sales data file not found: {DATA_FILE.name}")
    st.info("Place retail_data_c.csv beside app.py or configure MySQL secrets.")
    st.stop()

if use_mysql:
    try:
        sales = load_database_sales_data()
    except (DatabaseConnectionError, OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not load sales from MySQL: {error}")
        st.stop()
    st.caption("Data source: MySQL")
else:
    try:
        sales = load_sales_data(str(DATA_FILE))
    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not load the sales data: {error}")
        st.stop()
    st.caption("Data source: local CSV")

min_date = sales["InvoiceDate"].min().date()
max_date = sales["InvoiceDate"].max().date()

with st.sidebar:
    st.header("📊 Revenue intelligence")
    st.caption("Filter retail performance and forecast future sales.")
    category_options = ["All categories", *sorted(sales["Category"].dropna().unique())]
    region_options = ["All regions", *sorted(sales["Region"].dropna().unique())]
    category = st.selectbox("Product category", category_options)
    region = st.selectbox("Region", region_options)
    date_range = st.date_input(
        "Historical date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30, step=1)
    st.caption("The forecast starts after the end of the selected date range.")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

filtered = sales.loc[
    sales["InvoiceDate"].dt.date.between(start_date, end_date)
].copy()
if category != "All categories":
    filtered = filtered.loc[filtered["Category"] == category]
if region != "All regions":
    filtered = filtered.loc[filtered["Region"] == region]

if filtered.empty:
    st.warning("No transactions match the selected filters and date range.")
    st.stop()

daily_actuals = (
    filtered.groupby("InvoiceDate", as_index=False)
    .agg(Revenue=("Revenue", "sum"), Orders=("Revenue", "size"))
    .sort_values("InvoiceDate")
)

revenue_total = filtered["Revenue"].sum()
average_order = filtered["Revenue"].mean()
order_count = len(filtered)
customer_count = filtered["CustomerID"].nunique()

metric_columns = st.columns(4)
metric_columns[0].metric("Revenue", format_indian_currency(revenue_total))
metric_columns[1].metric("Transactions", f"{order_count:,}")
metric_columns[2].metric("Average transaction", format_indian_currency(average_order))
metric_columns[3].metric("Customers", f"{customer_count:,}")

st.subheader("Daily sales")
daily_actuals["Revenue (formatted)"] = daily_actuals["Revenue"].map(
    lambda value: format_indian_currency(value)
)
axis_style = {
    "labelColor": THEME["muted"],
    "titleColor": THEME["ink"],
    "gridColor": THEME["border"],
    "domainColor": THEME["border"],
    "tickColor": THEME["border"],
}

daily_chart = (
    alt.Chart(daily_actuals)
    .mark_line(color="#3867f4", strokeWidth=2)
    .encode(
        x=alt.X(
            "InvoiceDate:T",
            title="Date",
            axis=alt.Axis(format="%b %Y", tickCount=8, labelAngle=-30),
        ),
        y=alt.Y(
            "Revenue:Q",
            title="Revenue (₹)",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(
                labelExpr="(datum.value < 0 ? '-' : '') + replace(replace(format(abs(datum.value), '.0f'), /,/g, ''), /(\\d)(?=(\\d{2})+\\d{3}$)/g, '$1,')"
            ),
        ),
        tooltip=[
            alt.Tooltip("InvoiceDate:T", title="Date"),
            alt.Tooltip("Revenue (formatted):N", title="Revenue"),
        ],
    )
    .properties(height=300, width="container")
    .configure(background="transparent")
    .configure_view(stroke=None)
    .configure_axis(**axis_style)
)
st.altair_chart(daily_chart, theme=None)

forecast_source = sales.loc[
    sales["InvoiceDate"].dt.date.between(start_date, end_date)
]
try:
    history, forecast = make_forecast(
        forecast_source, category, region, forecast_days
    )
except (ValueError, RuntimeError) as error:
    st.error(f"Could not generate the forecast: {error}")
    st.stop()

st.subheader(f"Next {forecast_days} days")
history_plot = history.rename(columns={"ds": "Date", "y": "Revenue"})
forecast_plot = forecast.rename(
    columns={
        "ds": "Date",
        "yhat": "Revenue",
        "yhat_lower": "Lower estimate",
        "yhat_upper": "Upper estimate",
    }
)
forecast_plot["Revenue (formatted)"] = forecast_plot["Revenue"].map(
    lambda value: format_indian_currency(value)
)
forecast_plot["Lower estimate (formatted)"] = forecast_plot[
    "Lower estimate"
].map(lambda value: format_indian_currency(value))
forecast_plot["Upper estimate (formatted)"] = forecast_plot[
    "Upper estimate"
].map(lambda value: format_indian_currency(value))
history_plot["Revenue (formatted)"] = history_plot["Revenue"].map(
    lambda value: format_indian_currency(value)
)
forecast_line = pd.concat(
    [history_plot.tail(1)[["Date", "Revenue"]], forecast_plot[["Date", "Revenue"]]],
    ignore_index=True,
)
forecast_line["Revenue (formatted)"] = forecast_line["Revenue"].map(
    lambda value: format_indian_currency(value)
)
forecast_chart = (
    alt.Chart(forecast_plot)
    .mark_area(color="#12a594", opacity=0.16)
    .encode(
        x=alt.X(
            "Date:T",
            title="Date",
            axis=alt.Axis(format="%b %d", tickCount=8, labelAngle=-30),
        ),
        y=alt.Y(
            "Lower estimate:Q",
            title="Revenue (₹)",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(
                labelExpr="(datum.value < 0 ? '-' : '') + replace(replace(format(abs(datum.value), '.0f'), /,/g, ''), /(\\d)(?=(\\d{2})+\\d{3}$)/g, '$1,')"
            ),
        ),
        y2="Upper estimate:Q",
        tooltip=[
            alt.Tooltip("Date:T", title="Date"),
            alt.Tooltip("Lower estimate (formatted):N", title="Lower estimate"),
            alt.Tooltip("Upper estimate (formatted):N", title="Upper estimate"),
        ],
    )
    + alt.Chart(history_plot)
    .mark_line(color="#3867f4", strokeWidth=2)
    .encode(
        x="Date:T",
        y="Revenue:Q",
        tooltip=[
            alt.Tooltip("Date:T", title="Date"),
            alt.Tooltip("Revenue (formatted):N", title="Actual revenue"),
        ],
    )
    + alt.Chart(forecast_line)
    .mark_line(color="#12a594", strokeWidth=2.5)
    .encode(
        x="Date:T",
        y="Revenue:Q",
        tooltip=[
            alt.Tooltip("Date:T", title="Date"),
            alt.Tooltip("Revenue (formatted):N", title="Forecast"),
        ],
    )
).properties(height=380, width="container").configure(
    background="transparent"
).configure_view(stroke=None).configure_axis(**axis_style)
st.altair_chart(
    forecast_chart,
    theme=None,
)

forecast_metrics = st.columns(3)
forecast_metrics[0].metric(
    "Expected forecast revenue",
    format_indian_currency(forecast["yhat"].sum()),
)
forecast_metrics[1].metric(
    "Average forecast per day",
    format_indian_currency(forecast["yhat"].mean()),
)
forecast_metrics[2].metric(
    "Peak forecast day",
    forecast.loc[forecast["yhat"].idxmax(), "ds"].strftime("%b %d, %Y"),
)

with st.expander("View forecast details"):
    forecast_table = forecast.rename(
        columns={
            "ds": "Date",
            "yhat": "Expected revenue",
            "yhat_lower": "Lower estimate",
            "yhat_upper": "Upper estimate",
        }
    ).copy()
    forecast_table["Date"] = forecast_table["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        forecast_table.style.format(
            {
                "Expected revenue": lambda value: format_indian_currency(value, 2),
                "Lower estimate": lambda value: format_indian_currency(value, 2),
                "Upper estimate": lambda value: format_indian_currency(value, 2),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with st.sidebar:
    st.divider()
    st.subheader("✨ Ask Gemini")
    st.caption(
        "Gemini can query sales data read-only. Database credentials are never shared."
    )
    with st.form("gemini_forecast_form"):
        gemini_question = st.text_area(
            "Your question",
            placeholder="What does this forecast suggest about revenue?",
            key="gemini_forecast_question",
        )
        ask_gemini = st.form_submit_button(
            "Ask Gemini",
            type="primary",
            use_container_width=True,
        )

if ask_gemini:
    if not gemini_question.strip():
        st.sidebar.warning("Enter a question about the current sales and forecast.")
    else:
        try:
            gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        except (KeyError, FileNotFoundError):
            gemini_api_key = None
        gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        if not gemini_api_key:
            st.sidebar.error(
                "Gemini is not configured. Add GEMINI_API_KEY to Streamlit "
                "secrets or the environment."
            )
        else:
            forecast_summary = {
                "filters": {
                    "category": category,
                    "region": region,
                    "date_range": [start_date.isoformat(), end_date.isoformat()],
                },
                "dashboard_revenue": round(float(revenue_total), 2),
                "forecast_days": [
                    {
                        "date": row.ds.strftime("%Y-%m-%d"),
                        "expected_revenue": round(float(row.yhat), 2),
                        "lower_estimate": round(float(row.yhat_lower), 2),
                        "upper_estimate": round(float(row.yhat_upper), 2),
                    }
                    for row in forecast.itertuples(index=False)
                ],
            }
            prompt = (
                "Answer the user using the read-only retail sales query tool for "
                "historical database facts. The current dashboard forecast context "
                "is included below and is an estimate, not a database fact. Never "
                "claim that you can modify data, access credentials, or read files.\n\n"
                f"Current dashboard forecast:\n"
                f"{json.dumps(forecast_summary, ensure_ascii=False)}"
                f"\n\nUser question:\n{gemini_question.strip()}"
            )

            try:
                with st.spinner("Gemini is analyzing the current forecast..."):
                    with genai.Client(api_key=gemini_api_key) as client:
                        contents = [
                            types.Content(
                                role="user",
                                parts=[types.Part(text=prompt)],
                            )
                        ]
                        generation_config = types.GenerateContentConfig(
                            system_instruction=(
                                "You are a retail sales forecasting assistant. "
                                "Use query_retail_sales for historical database facts. "
                                "Treat its data as read-only and use only values "
                                "returned by the tool. Distinguish historical facts "
                                "from model estimates, never invent values, and state "
                                "when data is insufficient. Format "
                                "rupee amounts using Indian digit grouping, such as "
                                "₹12,34,567. Never request credentials "
                                "or claim to access the database directly."
                            ),
                            tools=[RETAIL_SALES_TOOL],
                        )
                        for _ in range(5):
                            response = client.models.generate_content(
                                model=GEMINI_MODEL,
                                contents=contents,
                                config=generation_config,
                            )
                            function_calls = response.function_calls or []
                            if not function_calls:
                                break

                            function_responses = []
                            for function_call in function_calls:
                                if function_call.name != "query_retail_sales":
                                    result = {"error": "Unsupported database operation."}
                                else:
                                    try:
                                        result = query_retail_sales(
                                            sales,
                                            **(function_call.args or {}),
                                        )
                                    except (TypeError, ValueError) as error:
                                        result = {"error": str(error)}
                                function_responses.append(
                                    types.Part.from_function_response(
                                        name=function_call.name,
                                        response=result,
                                    )
                                )
                            contents.extend(
                                [
                                    response.candidates[0].content,
                                    types.Content(
                                        role="user",
                                        parts=function_responses,
                                    ),
                                ]
                            )
                        else:
                            st.sidebar.error(
                                "Gemini could not complete its sales data lookup. "
                                "Try a narrower question."
                            )
                            response = None

                if response is not None and response.text:
                    st.session_state["gemini_forecast_answer"] = response.text
                elif response is not None:
                    st.sidebar.error("Gemini returned an empty response.")
            except APIError as error:
                st.sidebar.error(f"Gemini request failed: {error}")

if answer := st.session_state.get("gemini_forecast_answer"):
    with st.sidebar.chat_message("assistant"):
        st.markdown(answer)

st.caption(
    "Forecasts are estimates based on historical daily revenue; actual results may differ."
)
