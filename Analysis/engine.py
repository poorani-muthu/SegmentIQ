"""
analysis/engine.py
End-to-end customer segmentation pipeline:
  1. Data Cleaning
  2. RFM Feature Engineering
  3. Feature Scaling
  4. Optimal K Selection (Elbow + Silhouette)
  5. K-Means Clustering
  6. PCA + UMAP-style reduction (using PCA for 2D)
  7. Segment Profiling & Business Interpretation
  8. Actionable Recommendations
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist


# ── 1. DATA CLEANING ─────────────────────────────────────────────────────────

def clean_data(df_raw: pd.DataFrame) -> dict:
    """Full cleaning pipeline with audit trail."""
    audit = {}
    df = df_raw.copy()

    audit["raw_rows"]          = len(df)
    audit["raw_customers"]     = int(df["Customer ID"].nunique())

    # Drop missing Customer IDs
    before = len(df)
    df = df.dropna(subset=["Customer ID"])
    audit["dropped_missing_cid"] = before - len(df)

    # Drop cancellations (Invoice starts with C)
    before = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    audit["dropped_cancellations"] = before - len(df)

    # Drop zero/negative quantity or price
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    audit["dropped_bad_qty_price"] = before - len(df)

    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    audit["dropped_duplicates"] = before - len(df)

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # LineTotal
    df["LineTotal"] = df["Quantity"] * df["Price"]

    # Cast Customer ID
    df["Customer ID"] = df["Customer ID"].astype(int)

    audit["clean_rows"]      = len(df)
    audit["clean_customers"] = int(df["Customer ID"].nunique())
    audit["date_min"]        = str(df["InvoiceDate"].min().date())
    audit["date_max"]        = str(df["InvoiceDate"].max().date())
    audit["total_revenue"]   = float(round(df["LineTotal"].sum(), 2))
    audit["countries"]       = int(df["Country"].nunique())

    return {"df": df, "audit": audit}


# ── 2. RFM FEATURE ENGINEERING ───────────────────────────────────────────────

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary per customer."""
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Recency   = ("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency = ("Invoice",     "nunique"),
        Monetary  = ("LineTotal",   "sum"),
    ).reset_index()

    # Also compute AOV and purchase span
    rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"]

    span = df.groupby("Customer ID")["InvoiceDate"].agg(lambda x: (x.max() - x.min()).days)
    rfm["PurchaseSpan"] = rfm["Customer ID"].map(span)

    # Country lookup
    country_map = df.groupby("Customer ID")["Country"].first()
    rfm["Country"] = rfm["Customer ID"].map(country_map)

    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["AOV"]      = rfm["AOV"].round(2)

    return rfm


# ── 3. SCALING ────────────────────────────────────────────────────────────────

def scale_features(rfm: pd.DataFrame):
    """Log-transform skewed features, then StandardScale."""
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features].copy()

    # Log1p transform (Monetary and Frequency are right-skewed)
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"]  = np.log1p(X["Monetary"])
    X["Recency"]   = np.log1p(X["Recency"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, features


# ── 4. OPTIMAL K ──────────────────────────────────────────────────────────────

def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Return elbow (inertia) and silhouette scores for each k."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia   = km.inertia_
        sil       = silhouette_score(X_scaled, labels)
        db        = davies_bouldin_score(X_scaled, labels)
        results.append({"k": k, "inertia": inertia, "silhouette": sil, "davies_bouldin": db})
    return pd.DataFrame(results)


def pick_best_k(k_df: pd.DataFrame) -> int:
    """Pick k with highest silhouette score."""
    return int(k_df.loc[k_df["silhouette"].idxmax(), "k"])


# ── 5. CLUSTERING ─────────────────────────────────────────────────────────────

def run_kmeans(X_scaled, k: int):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


# ── 6. PCA 2D & 3D ────────────────────────────────────────────────────────────

def run_pca(X_scaled, n=3):
    pca = PCA(n_components=n, random_state=42)
    components = pca.fit_transform(X_scaled)
    explained  = pca.explained_variance_ratio_
    return components, explained, pca


# ── 7. SEGMENT PROFILING ──────────────────────────────────────────────────────

SEGMENT_NAMES = {
    "high_r_high_f_high_m": ("Champions",        "#22c55e", "🏆"),
    "high_r_med_f_med_m":   ("Loyal Customers",  "#3b82f6", "💙"),
    "high_r_low_f_low_m":   ("Promising",        "#a78bfa", "🌱"),
    "low_r_high_f_high_m":  ("At-Risk",          "#f59e0b", "⚠️"),
    "low_r_low_f_low_m":    ("Hibernating",      "#94a3b8", "😴"),
    "fallback":             ("Needs Attention",  "#ef4444", "🔴"),
}

def _classify_segment(row, rfm_quantiles):
    r_med = rfm_quantiles["Recency"][0.5]
    f_med = rfm_quantiles["Frequency"][0.5]
    m_med = rfm_quantiles["Monetary"][0.5]
    f_75  = rfm_quantiles["Frequency"][0.75]
    m_75  = rfm_quantiles["Monetary"][0.75]

    if row["Recency"] <= r_med and row["Frequency"] >= f_75 and row["Monetary"] >= m_75:
        return "Champions"
    elif row["Recency"] <= r_med and row["Frequency"] >= f_med:
        return "Loyal Customers"
    elif row["Recency"] <= r_med and row["Frequency"] < f_med:
        return "Promising"
    elif row["Recency"] > r_med and row["Frequency"] >= f_med and row["Monetary"] >= m_med:
        return "At-Risk"
    elif row["Recency"] > r_med and row["Frequency"] < f_med:
        return "Hibernating"
    else:
        return "Needs Attention"


SEG_META = {
    "Champions": {
        "color":       "#22c55e",
        "emoji":       "🏆",
        "value":       "High Value",
        "description": "Best customers — bought recently, buy often, spend the most.",
        "strategy":    "Reward them. Launch VIP loyalty tiers, early access to new products, and referral programs. They are your brand ambassadors.",
        "actions": [
            "Launch exclusive VIP membership with early product access",
            "Invite to beta-test new product lines",
            "Offer referral incentives — they will convert their network",
            "Send personalised 'thank you' campaigns",
        ],
        "clv_outlook": "Highest CLV — protect at all cost.",
        "risk":        "Low",
    },
    "Loyal Customers": {
        "color":       "#3b82f6",
        "emoji":       "💙",
        "value":       "High Value",
        "description": "Buy regularly and have a solid relationship with the brand.",
        "strategy":    "Upsell and cross-sell. Push them toward the Champions tier by increasing basket size.",
        "actions": [
            "Bundle offers to increase average order value",
            "Personalised product recommendations based on history",
            "Loyalty point multipliers during slow seasons",
            "Tier-upgrade messaging: 'You are 2 orders away from VIP'",
        ],
        "clv_outlook": "Strong CLV — invest in upgrading to Champions.",
        "risk":        "Low–Medium",
    },
    "Promising": {
        "color":       "#a78bfa",
        "emoji":       "🌱",
        "value":       "Growth Potential",
        "description": "Recent customers who have not yet built frequency.",
        "strategy":    "Nurture quickly. The first 90 days determine if they become loyal or churn.",
        "actions": [
            "Onboarding email series with usage tips",
            "First-repeat-purchase discount within 30 days",
            "Targeted category expansion offers",
            "Social proof content (reviews, bestsellers)",
        ],
        "clv_outlook": "High upside if converted — act within 60 days.",
        "risk":        "Medium",
    },
    "At-Risk": {
        "color":       "#f59e0b",
        "emoji":       "⚠️",
        "value":       "Intervention Required",
        "description": "Previously good customers who have not purchased recently.",
        "strategy":    "Win back with urgency. Personalise outreach — they know the brand already.",
        "actions": [
            "Personalised win-back email: 'We miss you + 15% off'",
            "Survey to understand why they stopped buying",
            "Showcase new arrivals they have not seen",
            "Limited-time reactivation bundle offer",
        ],
        "clv_outlook": "Recoverable — but window is closing.",
        "risk":        "High",
    },
    "Hibernating": {
        "color":       "#94a3b8",
        "emoji":       "😴",
        "value":       "Dormant",
        "description": "Low recency, low frequency, low spend. At risk of permanent churn.",
        "strategy":    "Low-cost re-engagement or graceful exit. Do not over-invest — focus budget on At-Risk instead.",
        "actions": [
            "Low-cost email blast: seasonal sale or clearance",
            "Suppression list review — remove truly inactive (saves email costs)",
            "Last-chance win-back: 'Come back — 25% off, expires in 7 days'",
            "Retargeting ads with bestseller content",
        ],
        "clv_outlook": "Low unless reactivated. Prioritise At-Risk first.",
        "risk":        "Very High",
    },
    "Needs Attention": {
        "color":       "#ef4444",
        "emoji":       "🔴",
        "value":       "Intervention Required",
        "description": "Mixed signals — moderate recency but inconsistent spend or frequency.",
        "strategy":    "Diagnose the cohort. Run a/b test with a discount vs free shipping to find what motivates them.",
        "actions": [
            "A/B test discount vs free-shipping incentive",
            "Category preference survey",
            "Targeted push notification campaigns",
            "Review support tickets — may have had a bad experience",
        ],
        "clv_outlook": "Uncertain — needs investigation.",
        "risk":        "High",
    },
}


def profile_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    q = {col: rfm[col].quantile([0.25, 0.5, 0.75]).to_dict()
         for col in ["Recency", "Frequency", "Monetary"]}
    rfm["SegmentName"] = rfm.apply(_classify_segment, axis=1, rfm_quantiles=q)
    return rfm


def build_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = rfm.groupby("SegmentName").agg(
        CustomerCount = ("Customer ID", "count"),
        AvgRecency    = ("Recency",     "mean"),
        AvgFrequency  = ("Frequency",   "mean"),
        AvgMonetary   = ("Monetary",    "mean"),
        TotalRevenue  = ("Monetary",    "sum"),
        AvgAOV        = ("AOV",         "mean"),
    ).reset_index()
    summary["RevenueShare"] = (summary["TotalRevenue"] / summary["TotalRevenue"].sum() * 100).round(1)
    summary["CustomerShare"] = (summary["CustomerCount"] / summary["CustomerCount"].sum() * 100).round(1)
    for col in ["AvgRecency","AvgFrequency","AvgMonetary","TotalRevenue","AvgAOV"]:
        summary[col] = summary[col].round(2)
    return summary


# ── MASTER RUNNER ─────────────────────────────────────────────────────────────

def run_full_pipeline(csv_path: str) -> dict:
    df_raw = pd.read_csv(csv_path)

    # 1. Clean
    clean_result = clean_data(df_raw)
    df   = clean_result["df"]
    audit = clean_result["audit"]

    # 2. RFM
    rfm = compute_rfm(df)

    # 3. Scale
    X_scaled, scaler, features = scale_features(rfm)

    # 4. Optimal K
    k_df = find_optimal_k(X_scaled, k_range=range(2, 9))
    best_k = pick_best_k(k_df)

    # 5. KMeans
    km, labels = run_kmeans(X_scaled, best_k)
    rfm["Cluster"] = labels

    # 6. PCA
    pca_3d, explained_3d, pca_model = run_pca(X_scaled, n=3)
    pca_2d = pca_3d[:, :2]

    rfm["PCA1"] = pca_3d[:, 0]
    rfm["PCA2"] = pca_3d[:, 1]
    rfm["PCA3"] = pca_3d[:, 2]

    # 7. Segment profiling
    rfm = profile_segments(rfm)
    seg_summary = build_segment_summary(rfm)

    # Top products per segment
    df_merged = df.merge(rfm[["Customer ID","SegmentName"]], on="Customer ID")
    top_products = (
        df_merged.groupby(["SegmentName","Description"])["LineTotal"]
        .sum().reset_index()
        .sort_values(["SegmentName","LineTotal"], ascending=[True,False])
        .groupby("SegmentName").head(5)
    )

    # Country breakdown
    country_rev = (
        df.groupby("Country")["LineTotal"].sum()
        .sort_values(ascending=False).head(10)
        .reset_index().rename(columns={"LineTotal":"Revenue"})
    )

    # Monthly revenue trend
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = df.groupby("YearMonth")["LineTotal"].sum().reset_index()
    monthly.columns = ["Month","Revenue"]

    return {
        "audit":        audit,
        "rfm":          rfm,
        "k_analysis":   k_df.to_dict(orient="records"),
        "best_k":       best_k,
        "explained_var": [round(float(e)*100, 1) for e in explained_3d],
        "seg_summary":  seg_summary.to_dict(orient="records"),
        "top_products": top_products.to_dict(orient="records"),
        "country_rev":  country_rev.to_dict(orient="records"),
        "monthly_rev":  monthly.to_dict(orient="records"),
        "seg_meta":     SEG_META,
    }
