"""
Generates a realistic e-commerce transaction dataset modelled on the
UCI Online Retail II dataset (real-world structure, synthetic values).
~20,000 invoices, ~4,000 customers, UK + international.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
N_CUSTOMERS   = 4_000
N_INVOICES    = 22_000
START_DATE    = datetime(2010, 12, 1)
END_DATE      = datetime(2011, 12, 9)

COUNTRIES = {
    "United Kingdom": 0.85, "Germany": 0.04, "France": 0.03,
    "EIRE": 0.02, "Spain": 0.01, "Netherlands": 0.01,
    "Belgium": 0.01, "Switzerland": 0.01, "Portugal": 0.01, "Australia": 0.01,
}

PRODUCTS = [
    ("WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("REGENCY CAKESTAND 3 TIER", 12.75),
    ("JUMBO BAG RED RETROSPOT", 1.65),
    ("PARTY BUNTING", 4.95),
    ("LUNCH BAG RED RETROSPOT", 1.65),
    ("SET OF 3 CAKE TINS PANTRY DESIGN", 4.65),
    ("STRAWBERRY CERAMIC TRINKET BOX", 1.25),
    ("ROSES REGENCY TEACUP AND SAUCER", 1.25),
    ("PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("ALARM CLOCK BAKELIKE PINK", 3.75),
    ("PLASTERS IN TIN CIRCUS PARADE", 1.65),
    ("DOORMAT FAIRY CAKE", 7.08),
    ("LARGE LETTER BOX RETROSPOT", 4.95),
    ("ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("CERAMIC STORAGE JAR SHABBY", 3.75),
    ("CHRISTMAS TREE STAR", 1.25),
    ("VINTAGE SNAP CARDS", 1.25),
    ("VICTORIAN GLASS HANGING T-LIGHT", 1.45),
    ("RETROSPOT HEART HOT WATER BOTTLE", 4.95),
    ("SAVE THE PLANET MUG", 1.25),
    ("WOOD 2 DRAWER CABINET WHITE FINISH", 8.50),
    ("CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("KNITTED UNION FLAG HOT WATER BOTTLE", 3.39),
    ("RED WOOLLY HOTTIE WHITE HEART", 3.39),
    ("SET OF 4 PANTRY JELLY MOULDS", 1.25),
    ("GLASS STAR FROSTED T-LIGHT HOLDER", 1.25),
    ("HAND WARMER UNION JACK", 1.65),
    ("HAND WARMER RED POLKA DOT", 1.65),
    ("BATH BUILDING BLOCK WORD", 3.75),
    ("TRADITIONAL CHOCOLATE PLAYING CARDS", 1.25),
]

STOCK_CODES = [f"8{1000+i}" for i in range(len(PRODUCTS))]

# ── Customer segments (latent) ────────────────────────────────────────────
# Champions (10%), Loyal (20%), At-Risk (15%), Hibernating (30%), New (25%)
segment_weights = [0.10, 0.20, 0.15, 0.30, 0.25]
segment_names   = ["Champions", "Loyal Customers", "At-Risk", "Hibernating", "New Customers"]

customer_ids = list(range(12000, 12000 + N_CUSTOMERS))
customer_segments = np.random.choice(
    range(len(segment_names)), size=N_CUSTOMERS, p=segment_weights
)
customer_country = np.random.choice(
    list(COUNTRIES.keys()), size=N_CUSTOMERS,
    p=list(COUNTRIES.values())
)

# Segment-specific behaviour
seg_params = {
    # seg: (recency_days_range, freq_range, avg_basket_range)
    0: (1,  30,  8, 25, 40, 150),   # Champions: recent, frequent, big basket
    1: (10, 90,  5, 20, 20,  80),   # Loyal
    2: (60, 180, 2,  8, 15,  60),   # At-Risk
    3: (150,340, 1,  3,  8,  30),   # Hibernating
    4: (1,  60,  1,  3, 10,  50),   # New
}

rows = []
date_range_days = (END_DATE - START_DATE).days

for cust_idx, cust_id in enumerate(customer_ids):
    seg = customer_segments[cust_idx]
    r0, r1, f0, f1, b0, b1 = seg_params[seg]

    # How many invoices for this customer?
    n_orders = np.random.randint(f0, f1 + 1)

    # Latest purchase: recency_days before END_DATE
    recency_days = np.random.randint(r0, r1 + 1)
    latest = END_DATE - timedelta(days=recency_days)

    for _ in range(n_orders):
        # Spread orders back in time
        order_date = latest - timedelta(days=np.random.randint(0, max(1, (date_range_days - recency_days))))
        order_date = max(order_date, START_DATE)

        invoice_no = f"INV{np.random.randint(100000, 999999)}"
        n_items = np.random.randint(1, 8)

        for _ in range(n_items):
            prod_idx = np.random.randint(0, len(PRODUCTS))
            desc, unit_price = PRODUCTS[prod_idx]
            # Add slight noise to price
            unit_price = round(unit_price * np.random.uniform(0.9, 1.1), 2)
            qty = np.random.randint(1, 15)

            rows.append({
                "Invoice":       invoice_no,
                "StockCode":     STOCK_CODES[prod_idx],
                "Description":   desc,
                "Quantity":      qty,
                "InvoiceDate":   order_date.strftime("%Y-%m-%d %H:%M:%S"),
                "Price":         unit_price,
                "Customer ID":   cust_id,
                "Country":       customer_country[cust_idx],
            })

df = pd.DataFrame(rows)

# Inject ~2% cancellations (negative qty, C prefix)
cancel_mask = np.random.random(len(df)) < 0.02
df.loc[cancel_mask, "Invoice"] = "C" + df.loc[cancel_mask, "Invoice"].str[3:]
df.loc[cancel_mask, "Quantity"] = -df.loc[cancel_mask, "Quantity"]

# Inject ~1% missing Customer IDs
missing_mask = np.random.random(len(df)) < 0.01
df.loc[missing_mask, "Customer ID"] = np.nan

df.to_csv("/home/claude/customer_segmentation/data/online_retail.csv", index=False)
print(f"Dataset created: {len(df):,} rows, {df['Customer ID'].nunique():.0f} unique customers")
print(df.head())
