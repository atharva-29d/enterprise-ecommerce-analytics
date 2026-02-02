import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import seaborn as sns
import matplotlib.pyplot as plt

from difflib import SequenceMatcher
from scipy.spatial.distance import cosine

from prophet import Prophet
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest

from sentence_transformers import SentenceTransformer

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config("Enterprise Ecommerce Analytics Engine", layout="wide")

# =====================================================
# SESSION STATE
# =====================================================
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

# =====================================================
# HEADER
# =====================================================
st.title("📊 Enterprise Ecommerce Analytics Engine")
st.caption("Upload → AI maps columns → confirm → Start → business insights")

# =====================================================
# EXECUTIVE INTRO
# =====================================================
with st.expander("📖 What Does This Dashboard Tell Me?", expanded=True):
    st.markdown("""
This dashboard transforms ecommerce transactions into **easy-to-understand business insights**.

• Customer behavior  
• Future revenue outlook  
• Unusual activity  
• Prediction reliability  

Designed for marketing, finance and leadership teams.
""")

# =====================================================
# LOAD NLP MODEL
# =====================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# =====================================================
# NLP ROLE DEFINITIONS
# =====================================================
ROLE_PHRASES = {
    "date": "transaction date timestamp purchase created at time",
    "price": "transaction amount revenue sales total order value",
    "customer": "customer id buyer user client account",
    "order": "order id invoice receipt transaction number",
    "rating": "review rating score stars feedback satisfaction",
    "shipping_cost": "shipping freight delivery cost logistics charge",
    "delivery_actual": "delivered date received completed arrival",
    "delivery_estimated": "estimated delivery expected promised date",
    "category": "product category item type department",
    "payment_type": "payment method tender card cash upi netbanking",
    "state": "state region province territory city location",
}

ROLE_KEYWORDS = {
    "date": ["date","time","timestamp","created","purchase"],
    "price": ["price","amount","revenue","sales","total","value"],
    "customer": ["customer","user","buyer","client"],
    "order": ["order","invoice","receipt","txn"],
    "rating": ["rating","review","score","stars","feedback"],
    "shipping_cost": ["shipping","freight","delivery_cost","logistics"],
    "delivery_actual": ["delivered","received","arrival"],
    "delivery_estimated": ["estimated","expected","promised"],
    "category": ["category","product","department","type"],
    "payment_type": ["payment","method","card","upi","wallet"],
    "state": ["state","region","city","province"],
}

# 👇 MUST be here
ROLE_EMBEDS = embedder.encode(list(ROLE_PHRASES.values()))


# =====================================================
# AUTO COLUMN DETECTION
# =====================================================
def normalize(txt):
    return re.sub(r"[^a-z0-9]", "", txt.lower())
@st.cache_data
def embed_text(txt):
    return embedder.encode([txt])[0]


def hybrid_score(col, series):

    scores = {r: 0 for r in ROLE_PHRASES}

    col_vec = embed_text(col)

    for i, role in enumerate(ROLE_PHRASES):
        scores[role] += (1 - cosine(col_vec, ROLE_EMBEDS[i])) * 0.4

    name = normalize(col)

    for role, kws in ROLE_KEYWORDS.items():
        for k in kws:
            scores[role] += SequenceMatcher(None, name, k).ratio() * 0.3

    sample = series.dropna().head(200)

    # ----- numeric detection -----
    numeric_ratio = pd.to_numeric(sample, errors="coerce").notna().mean()

    scores["price"] += numeric_ratio * 0.2
    scores["shipping_cost"] += numeric_ratio * 0.2
    scores["rating"] += numeric_ratio * 0.15

    # ----- datetime detection -----
    dt_ratio = pd.to_datetime(sample, errors="coerce").notna().mean()

    scores["date"] += dt_ratio * 0.25
    scores["delivery_actual"] += dt_ratio * 0.25
    scores["delivery_estimated"] += dt_ratio * 0.25

    # Cardinality signals
    unique_ratio = series.nunique() / max(len(series), 1)

    # Many repeats -> customer-like
    scores["customer"] += (1 - unique_ratio) * 0.4

    # Mostly unique -> order-like
    scores["order"] += unique_ratio * 0.4

    scores["date"] += pd.to_datetime(sample, errors="coerce").notna().mean() * 0.15
    scores["price"] += pd.to_numeric(sample, errors="coerce").notna().mean() * 0.15

    clean = normalize(col)

    if unique_ratio > 0.95:
        scores["customer"] -= 0.3

    for role in ROLE_PHRASES:
        if role in clean:
            scores[role] += 0.5

    # --- Geographic column penalty ---
    geo_keywords = ["state", "country", "city", "region", "province", "zipcode", "zip"]



    for g in geo_keywords:
        if g in clean:
            scores["customer"] -= 0.8
            scores["order"] -= 0.3

    clean = normalize(col)

    if any(k in clean for k in ["state", "city", "region", "province"]):
        scores["state"] += 0.8
        scores["customer"] -= 0.5

    return scores


@st.cache_data(show_spinner="🤖 AI analyzing dataset...")
def auto_detect_columns(df):

    role_scores = {r: {} for r in ROLE_PHRASES}

    sample_df = df.sample(min(3000, len(df)), random_state=42)

    for col in sample_df.columns:
        s = hybrid_score(col, sample_df[col])

        for r in role_scores:
            role_scores[r][col] = s[r]

    best, confidence = {}, {}

    for role, vals in role_scores.items():
        sorted_cols = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        best[role] = sorted_cols[0][0]

        if len(sorted_cols) > 1:
            gap = sorted_cols[0][1] - sorted_cols[1][1]
            abs_score = sorted_cols[0][1]

            confidence[role] = round(min(100, (0.6 * abs_score + 0.4 * gap) * 100), 1)


        else:
            confidence[role] = 100.0

        clean = normalize(col)



    return best, confidence


# =====================================================
# HEAVY PIPELINES (CACHED)
# =====================================================
@st.cache_data(show_spinner="📊 Creating customer segments...")
def compute_rfm(df):

    snapshot = df["date"].max() + pd.Timedelta(days=1)

    return (
        df.groupby("customer")
        .agg(
            Recency=("date", lambda x: (snapshot - x.max()).days),
            Frequency=("order", "nunique"),
            Monetary=("price", "sum"),
        )
        .reset_index()
    )


@st.cache_data(show_spinner="📈 Training forecasting model...")
def run_forecast(ts, horizon):

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(ts)

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    # Prevent negative revenue forecasts
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    return model, forecast



@st.cache_data(show_spinner="🚨 Detecting unusual days...")
def detect_anomalies(ts):

    iso = IsolationForest(contamination=0.02)

    out = ts.copy()
    out["AnomalyFlag"] = iso.fit_predict(out[["y"]])

    return out


@st.cache_data(show_spinner="📊 Evaluating forecast accuracy...")
@st.cache_data(show_spinner="📊 Evaluating forecast accuracy...")
def evaluate_forecast(ts):

    split = int(len(ts) * 0.8)
    train, test = ts.iloc[:split], ts.iloc[split:]

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(train)

    future = model.make_future_dataframe(periods=len(test))
    pred = model.predict(future).tail(len(test))

    mae = mean_absolute_error(test["y"], pred["yhat"])
    rmse = np.sqrt(mean_squared_error(test["y"], pred["yhat"]))

    return mae, rmse



# =====================================================
# SIDEBAR UPLOAD
# =====================================================
st.sidebar.header("📂 Upload Dataset")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.stop()

df_raw = pd.read_csv(uploaded)

auto_map, conf_map = auto_detect_columns(df_raw)
role_map = auto_map.copy()


# =====================================================
# COLUMN MAPPING
# =====================================================
st.sidebar.subheader("🧭 Column Mapping")

cols = df_raw.columns.tolist()
idx = lambda r: cols.index(auto_map[r]) if auto_map[r] in cols else 0

cust_col = st.sidebar.selectbox("Customer Column", cols, index=idx("customer"))
order_col = st.sidebar.selectbox("Order Column", cols, index=idx("order"))
price_col = st.sidebar.selectbox("Revenue Column", cols, index=idx("price"))
date_col = st.sidebar.selectbox("Date Column", cols, index=idx("date"))



# =====================================================
# CONFIDENCE DISPLAY
# =====================================================
st.sidebar.markdown("### 🤖 AI Mapping Confidence")

PRIMARY_ROLES = ["date", "price", "customer", "order"]

for role in PRIMARY_ROLES:

    if role not in conf_map:
        continue

    pct = float(conf_map[role])
    label = role.replace("_", " ").title()

    if pct >= 60:
        st.sidebar.success(f"{label}: {pct}% — very confident")
    elif pct >= 35:
        st.sidebar.warning(f"{label}: {pct}% — double-check")
    else:
        st.sidebar.error(f"{label}: {pct}% — manual review")

# ⚠ soft warning
low_conf = [r for r in PRIMARY_ROLES if conf_map.get(r, 100) < 35]

if low_conf:
    names = ", ".join(r.title() for r in low_conf)
    st.sidebar.warning(
        f"⚠ Low confidence in: {names}. "
        "Please double-check selected columns."
    )

# 🚀 START BUTTON
if st.sidebar.button("🚀 Start Analysis"):
    st.session_state.analysis_started = True

# ⛔ Stop AFTER rendering button
if not st.session_state.analysis_started:
    st.info("Confirm columns and click Start.")
    st.stop()


# =====================================================
# DATA PREP
# =====================================================
df = df_raw[[cust_col, order_col, price_col, date_col]].copy()
df.columns = ["customer", "order", "price", "date"]

df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df["date"].isna().sum():
    st.error("Invalid date values detected.")
    st.stop()

df = df.dropna()
df = df[df["price"] >= 0]

# =====================================================
# AUTO OPS + SATISFACTION FEATURES (AI-DRIVEN)
# =====================================================

ops_flags = {}

# ---- Rating / unhappy ----
if "rating" in role_map and role_map["rating"] in df_raw.columns:
    rating_col = role_map["rating"]
    df["rating"] = pd.to_numeric(df_raw[rating_col], errors="coerce")
    df["is_unhappy"] = (df["rating"] <= 2).astype(int)
    ops_flags["rating"] = True
else:
    ops_flags["rating"] = False

# ---- Delivery delay ----
if (
    "delivery_actual" in role_map and
    "delivery_estimated" in role_map and
    role_map["delivery_actual"] in df_raw.columns and
    role_map["delivery_estimated"] in df_raw.columns
):
    da = role_map["delivery_actual"]
    de = role_map["delivery_estimated"]

    df["delivery_actual"] = pd.to_datetime(df_raw[da], errors="coerce")
    df["delivery_estimated"] = pd.to_datetime(df_raw[de], errors="coerce")

    df["delivery_delay_days"] = (
        df["delivery_actual"] - df["delivery_estimated"]
    ).dt.days

    def delay_bucket(x):
        if pd.isna(x):
            return "Unknown"
        elif x <= 0:
            return "On Time"
        elif x <= 3:
            return "Late (1–3 days)"
        else:
            return "Very Late (>3 days)"

    df["delay_bucket"] = df["delivery_delay_days"].apply(delay_bucket)
    ops_flags["delivery"] = True
else:
    ops_flags["delivery"] = False

# ---- Shipping ratio ----
if (
    "shipping_cost" in role_map and
    role_map["shipping_cost"] in df_raw.columns
):
    ship = role_map["shipping_cost"]

    df["freight_to_price_ratio"] = (
        pd.to_numeric(df_raw[ship], errors="coerce") /
        pd.to_numeric(df["price"], errors="coerce").replace(0, np.nan)
    ).clip(upper=5)

    ops_flags["shipping"] = True
else:
    ops_flags["shipping"] = False

# ---- Category ----
if "category" in role_map and role_map["category"] in df_raw.columns:
    df["product_category"] = df_raw[role_map["category"]].astype(str)
    ops_flags["category"] = True
else:
    ops_flags["category"] = False

# ---- Payment ----
if "payment_type" in role_map and role_map["payment_type"] in df_raw.columns:
    df["payment_type"] = df_raw[role_map["payment_type"]].astype(str)
    ops_flags["payment"] = True
else:
    ops_flags["payment"] = False

# ---- Geography ----
if "state" in role_map and role_map["state"] in df_raw.columns:
    df["state"] = df_raw[role_map["state"]].astype(str)
    ops_flags["state"] = True
else:
    ops_flags["state"] = False



# =====================================================
# BUSINESS KPIs
# =====================================================
st.subheader("📊 Business Overview")

rev = df["price"].sum()
orders = df["order"].nunique()
custs = df["customer"].nunique()
aov = rev / orders

c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Revenue", f"{rev:,.0f}")
c2.metric("🧾 Orders", orders)
c3.metric("👥 Customers", custs)
c4.metric("📦 Avg Order Value", f"{aov:,.0f}")

# =====================================================
# TIME SERIES (internal ds/y)
# =====================================================
ts = df.groupby(df["date"].dt.date)["price"].sum().reset_index()
ts.columns = ["ds", "y"]
ts["ds"] = pd.to_datetime(ts["ds"])

# =====================================================
# TABS
# =====================================================
tabs = [
    "👥 Customer Segments",
    "📈 Revenue Forecast",
    "🚨 Unusual Activity",
    "📊 Model Accuracy",
]

if any(ops_flags.values()):
    tabs.append("🚚 Ops & Satisfaction")

tab_objs = st.tabs(tabs)
tab_lookup = dict(zip(tabs, tab_objs))


# =====================================================
# SEGMENTS
# =====================================================
with tab_lookup["👥 Customer Segments"]:


    rfm = compute_rfm(df)

    X = StandardScaler().fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    scores = {}

    for k in range(2, 5):
        km_temp = MiniBatchKMeans(n_clusters=k, random_state=42)
        labels = km_temp.fit_predict(X)
        scores[k] = silhouette_score(X, labels)

    best_k = max(scores, key=scores.get)

    st.info(f"📌 Optimal clusters selected: {best_k}")

    km = MiniBatchKMeans(n_clusters=best_k, random_state=42)
    rfm["SegmentID"] = km.fit_predict(X)

    segment_map = {
        0: {
            "name": "High-Value Customers 👑",
            "desc": "Buy frequently and spend a lot. Core revenue drivers.",
            "action": "Reward loyalty, premium offers, VIP programs."
        },
        1: {
            "name": "New Customers 🌱",
            "desc": "Recently made their first purchases. Still forming habits.",
            "action": "Onboarding emails, first-repeat discounts."
        },
        2: {
            "name": "Inactive / At-Risk 😴",
            "desc": "Haven’t purchased in a long time. Likely to churn.",
            "action": "Win-back campaigns, coupons, reminders."
        },
        3: {
            "name": "Repeat Customers 🔁",
            "desc": "Purchase regularly but not top spenders yet.",
            "action": "Upsell bundles, cross-sell, loyalty nudges."
        },
    }

    rfm["Segment Name"] = rfm["SegmentID"].map(
        lambda x: segment_map[x]["name"]
    )

    rfm["Segment Description"] = rfm["SegmentID"].map(
        lambda x: segment_map[x]["desc"]
    )

    rfm["Recommended Action"] = rfm["SegmentID"].map(
        lambda x: segment_map[x]["action"]
    )

    st.metric("Segment Quality Score", round(silhouette_score(X, rfm["SegmentID"]), 3))
    st.dataframe(
        rfm[
            [
                "customer",
                "Recency",
                "Frequency",
                "Monetary",
                "Segment Name",
                "Segment Description",
                "Recommended Action",
            ]
        ].head(50)
    )

# =====================================================
# FORECAST
# =====================================================
with tab_lookup["📈 Revenue Forecast"]:

    st.markdown("""
## 📘 How to Read This Revenue Forecast

This chart predicts **how your daily revenue may change in the future**.

### 🔵 Lines & Areas
• **Line** → predicted revenue  
• **Shaded region** → possible range  
• Wider = more uncertainty  

### 🧠 What the model learns
• weekly buying patterns  
• seasonality  
• long-term growth or decline  

### 🎯 How to use this
✔ inventory planning  
✔ marketing timing  
✔ budgeting  
✔ hiring & scaling  

⚠ Predictions are guidance, not guarantees.
""")

    horizon = st.slider("Forecast Horizon (Days)", 30, 180, 90)
    if len(ts) < 120:
        st.warning("⚠ Limited historical data — forecast may be unstable.")

    model, forecast = run_forecast(ts, horizon)

    fig = model.plot(forecast)
    ax = fig.gca()
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.set_title("Daily Revenue Forecast")

    st.pyplot(fig)

# =====================================================
# ANOMALIES
# =====================================================
with tab_lookup["🚨 Unusual Activity"]:

    st.markdown("""
### 🚨 Unusual Sales Days

Days that behaved very differently from normal patterns.

Investigate:
• promotions  
• holidays  
• outages  
• viral products
""")

    anom = detect_anomalies(ts)

    anom_disp = anom.rename(columns={
        "ds": "Date",
        "y": "Revenue",
    })

    anom_disp["Unusual?"] = anom_disp["AnomalyFlag"].map(
        {-1: "Yes 🚨", 1: "No"}
    )

    st.dataframe(
        anom_disp[["Date", "Revenue", "Unusual?"]].tail(50)
    )

# =====================================================
# MODEL ACCURACY
# =====================================================
with tab_lookup["📊 Model Accuracy"]:

    st.markdown("""
### 📊 Forecast Reliability

Lower error = better predictions.
""")

    mae, rmse = evaluate_forecast(ts)

    st.metric("MAE", round(mae, 2))
    st.metric("RMSE", round(rmse, 2))

    st.info(
        f"Average daily revenue ≈ {ts['y'].mean():,.0f}. "
        f"MAE {mae:,.0f} ⇒ typical error ≈ {(mae/ts['y'].mean())*100:.1f}%."
    )

if "🚚 Ops & Satisfaction" in tab_lookup:

    with tab_lookup["🚚 Ops & Satisfaction"]:

        st.subheader("🚚 Operations & Customer Satisfaction")

        if ops_flags["delivery"]:
            st.markdown("### Delivery Performance")

            agg = df.groupby("delay_bucket")["delivery_delay_days"].mean()

            fig, ax = plt.subplots()
            agg.plot(kind="bar", ax=ax)
            ax.set_ylabel("Avg Delay (days)")
            st.pyplot(fig)

            st.metric(
                "Late Delivery Rate",
                f"{(df['delivery_delay_days'] > 0).mean()*100:.2f}%"
            )

        if ops_flags["rating"] and ops_flags["delivery"]:
            st.markdown("### Ratings vs Delivery")

            fig, ax = plt.subplots()
            plot_df = df[["delay_bucket", "rating"]].dropna()

            if plot_df["delay_bucket"].nunique() > 1:
                fig, ax = plt.subplots()
                sns.boxplot(data=plot_df, x="delay_bucket", y="rating", ax=ax)
                st.pyplot(fig)
            else:
                st.info("Not enough delivery buckets with ratings to draw boxplot.")

            st.pyplot(fig)

        if ops_flags["shipping"] and ops_flags["rating"]:
            st.markdown("### Shipping Cost vs Rating")

            fig, ax = plt.subplots()
            plot_df = df[["freight_to_price_ratio", "rating"]].dropna()

            if not plot_df.empty:
                fig, ax = plt.subplots()
                sns.scatterplot(
                    data=plot_df,
                    x="freight_to_price_ratio",
                    y="rating",
                    alpha=0.4
                )
                st.pyplot(fig)
            else:
                st.info("Not enough data for freight vs rating plot.")

            st.pyplot(fig)

        if ops_flags["payment"] and ops_flags["rating"]:
            st.markdown("### Payment Method vs Rating")

            pay_agg = df.groupby("payment_type")["rating"].mean()

            fig, ax = plt.subplots()
            pay_agg.plot(kind="bar", ax=ax)
            ax.set_ylabel("Avg Rating")
            st.pyplot(fig)
