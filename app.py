"""
app.py — Flask web app for Customer Segmentation Dashboard
Run: python3 app.py
Open: http://localhost:5000
"""
import os, json
from flask import Flask, render_template, jsonify

app = Flask(__name__)

def load_data():
    path = os.path.join(os.path.dirname(__file__), "static", "analysis_data.json")
    with open(path) as f:
        return json.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/data")
def api_data():
    data = load_data()
    # Send everything except the huge full rfm
    data.pop("rfm_full", None)
    return jsonify(data)

@app.route("/api/segment/<segment_name>")
def api_segment(segment_name):
    data = load_data()
    rfm  = data.get("rfm_full", data.get("rfm_sample", []))
    customers = [r for r in rfm if r.get("SegmentName") == segment_name]
    meta = data["seg_meta"].get(segment_name, {})
    seg_row = next((s for s in data["seg_summary"] if s["SegmentName"] == segment_name), {})
    top_prod = [p for p in data["top_products"] if p.get("SegmentName") == segment_name]
    return jsonify({
        "customers": customers[:500],
        "meta":      meta,
        "stats":     seg_row,
        "top_products": top_prod,
    })

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🚀  Customer Segmentation Dashboard")
    print("  Open: http://localhost:5000")
    print("="*60 + "\n")
    # To this:
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
