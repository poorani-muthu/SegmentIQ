"""
precompute.py — Run the full pipeline once and cache results to JSON.
This makes the Flask app fast (no on-request recomputation).
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.engine import run_full_pipeline

def serialize(obj):
    """Handle non-JSON-serializable types."""
    import numpy as np, pandas as pd
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, pd.Period):      return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")

print("Running full pipeline... (this may take ~30 seconds)")
result = run_full_pipeline("data/online_retail.csv")

# Convert rfm DataFrame to records
rfm_df = result.pop("rfm")
result["rfm_sample"] = rfm_df.sample(min(2000, len(rfm_df)), random_state=42).to_dict(orient="records")
result["rfm_full"]   = rfm_df.to_dict(orient="records")

# Build segment-level stats for radar chart
seg_stats = []
for seg in result["seg_summary"]:
    name = seg["SegmentName"]
    seg_stats.append({
        "segment":   name,
        "recency":   round(100 - min(seg["AvgRecency"] / 3.6, 100), 1),   # invert: lower recency = better
        "frequency": round(min(seg["AvgFrequency"] * 5, 100), 1),
        "monetary":  round(min(seg["AvgMonetary"] / 50, 100), 1),
        "aov":       round(min(seg["AvgAOV"] / 30, 100), 1),
        "color":     result["seg_meta"].get(name, {}).get("color", "#888"),
    })
result["seg_stats"] = seg_stats

with open("static/analysis_data.json", "w") as f:
    json.dump(result, f, default=serialize)

print(f"✓ Cached to static/analysis_data.json")
print(f"  Segments: {[s['SegmentName'] for s in result['seg_summary']]}")
print(f"  RFM rows: {len(result['rfm_full'])}")
print(f"  Total Revenue: £{result['audit']['total_revenue']:,.2f}")
