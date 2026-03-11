# 🎯 Customer Segmentation & BI Analysis
### By Poorani M | Portfolio Project | Business Analyst · Data Analyst

---

## What This Project Does

End-to-end customer segmentation using **RFM Analysis**, **K-Means Clustering**, and **PCA**
on a synthetic Online Retail II dataset (97K+ transactions, 4,000+ customers).

Produces a **live interactive BI dashboard** with:
- Full data cleaning audit trail
- RFM feature engineering
- Optimal K selection (Elbow + Silhouette)
- 2D + 3D PCA cluster visualization
- 6 business segments with named personas
- Actionable strategy per segment
- Executive Summary (CMO-ready)
- Methodology documentation

---

## Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Backend      | Python 3.10+ · Flask              |
| Analysis     | Pandas · NumPy · Scikit-learn · SciPy |
| Visualisation| Plotly.js (CDN, browser-side)      |
| Frontend     | Vanilla JS · CSS3 (no framework)  |
| Data         | Synthetic Online Retail II (UCI-style) |

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset (already done — skip if data/online_retail.csv exists)
```bash
python3 data/generate_data.py
```

### 3. Pre-compute analysis (already done — skip if static/analysis_data.json exists)
```bash
python3 precompute.py
```

### 4. Start the dashboard
```bash
python3 app.py
```

### 5. Open in browser
```
http://localhost:5000
```

---

## Project Structure

```
customer_segmentation/
├── app.py                    # Flask web server
├── precompute.py             # Run analysis + cache to JSON
├── requirements.txt
├── README.md
│
├── analysis/
│   ├── __init__.py
│   └── engine.py             # Full pipeline: clean → RFM → scale → cluster → segment
│
├── data/
│   ├── generate_data.py      # Synthetic dataset generator
│   └── online_retail.csv     # Generated dataset (97K rows)
│
├── static/
│   └── analysis_data.json    # Pre-computed results (fast serving)
│
└── templates/
    └── index.html            # Dashboard UI (7 pages, Plotly.js)
```

---

## Analysis Pipeline

```
Raw Data (97,064 rows)
        │
        ▼
[ 1. Data Cleaning ]
  - Drop null Customer IDs  (-949 rows)
  - Remove cancellations    (-1,885 rows)
  - Remove bad qty/price    (-0 rows)
  - Remove duplicates       (-11 rows)
        │
        ▼
Clean Dataset (94,219 rows · 3,998 customers)
        │
        ▼
[ 2. RFM Feature Engineering ]
  Recency  = days since last purchase
  Frequency = unique invoices per customer
  Monetary  = total spend
  + AOV, PurchaseSpan
        │
        ▼
[ 3. Feature Scaling ]
  Log1p transformation (skew correction)
  StandardScaler (zero mean, unit variance)
        │
        ▼
[ 4. Optimal K Selection ]
  K=2..8 evaluated
  Silhouette Score → peak = optimal K
  Elbow Curve → cross-validation
        │
        ▼
[ 5. K-Means Clustering ]
  n_clusters = optimal K
  n_init = 10, random_state = 42
        │
        ▼
[ 6. PCA Dimensionality Reduction ]
  3 components → 2D + 3D visualization
  PC1 ~82% variance, PC2 ~14% variance
        │
        ▼
[ 7. Segment Profiling & Business Interpretation ]
  Champions · Loyal Customers · Promising
  At-Risk · Needs Attention · Hibernating
  → Actionable strategy per segment
  → Executive Summary
```

---

## Segments Discovered

| Segment          | Value Level       | Key Trait                    |
|------------------|-------------------|------------------------------|
| 🏆 Champions      | High Value        | Recent + frequent + high spend |
| 💙 Loyal Customers| High Value        | Regular buyers, solid relationship |
| 🌱 Promising      | Growth Potential  | Recent but infrequent         |
| ⚠️ At-Risk        | Intervention Req. | Previously good, now inactive |
| 😴 Hibernating    | Dormant           | Low across all RFM dimensions |
| 🔴 Needs Attention| Intervention Req. | Mixed signals, inconsistent   |

---

## Skills Demonstrated

- **Data Cleaning** — documented, reproducible, fully audited
- **Feature Engineering** — RFM, log transforms, scaling
- **Unsupervised ML** — K-Means with rigorous K selection
- **Dimensionality Reduction** — PCA with explained variance
- **Business Interpretation** — segment naming, CLV outlook, risk rating
- **Actionable Recommendations** — 4 concrete actions per segment
- **Executive Communication** — CMO-ready 1-page summary
- **Full-Stack Deployment** — Flask API + interactive Plotly.js dashboard

---

*Portfolio project by Poorani M — targeting Business Analyst and Data Analyst roles.*
