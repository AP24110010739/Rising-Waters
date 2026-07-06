"""
Dataset generator — matches the actual 'arbethi/rainfall-dataset' structure
from Kaggle (as seen in screenshots):
Columns: Temp, Humidity, Cloud Cover, ANNUAL, Jan-Feb, Mar-May,
         Jun-Sep, Oct-Dec, avgjune, sub, flood (target: 0/1)
115 entries
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 115

# Flood mask (~39% flood events, matching actual dataset distribution)
flood_mask = np.zeros(N, dtype=int)
flood_indices = sorted(np.random.choice(N, size=45, replace=False))
flood_mask[flood_indices] = 1

# Generate features correlated with flood label
Temp       = np.where(flood_mask, np.random.randint(28, 32, N), np.random.randint(27, 31, N)).astype(float)
Humidity   = np.where(flood_mask, np.random.randint(74, 90, N), np.random.randint(65, 80, N)).astype(float)
CloudCover = np.where(flood_mask, np.random.randint(35, 50, N), np.random.randint(28, 46, N)).astype(float)
ANNUAL     = np.where(flood_mask, np.random.uniform(3000, 4000, N), np.random.uniform(2500, 3500, N)).round(1)
JanFeb     = np.where(flood_mask, np.random.uniform(5,   80,  N), np.random.uniform(8,   100, N)).round(1)
MarMay     = np.where(flood_mask, np.random.uniform(200, 450, N), np.random.uniform(250, 430, N)).round(1)
JunSep     = np.where(flood_mask, np.random.uniform(1900,2700,N), np.random.uniform(1500,2500,N)).round(1)
OctDec     = np.where(flood_mask, np.random.uniform(400, 750, N), np.random.uniform(300, 700, N)).round(1)
avgjune    = np.random.uniform(100, 400, N).round(6)
sub        = np.random.uniform(200, 900, N).round(1)

df = pd.DataFrame({
    "Temp":        Temp,
    "Humidity":    Humidity,
    "Cloud Cover": CloudCover,
    "ANNUAL":      ANNUAL,
    "Jan-Feb":     JanFeb,
    "Mar-May":     MarMay,
    "Jun-Sep":     JunSep,
    "Oct-Dec":     OctDec,
    "avgjune":     avgjune,
    "sub":         sub,
    "flood":       flood_mask
})

# Seed first 5 rows to match screenshots exactly
df.iloc[0] = [29, 70, 30, 3248.6, 73.4, 386.2, 2122.8, 666.1, 274.866667, 649.9, 0]
df.iloc[1] = [28, 75, 40, 3326.6,  9.3, 275.7, 2403.4, 638.2, 130.300000, 256.4, 1]
df.iloc[2] = [28, 75, 42, 3271.2, 21.7, 336.3, 2343.0, 570.1, 186.200000, 308.9, 0]
df.iloc[3] = [29, 71, 44, 3129.7, 26.7, 339.4, 2398.2, 365.3, 366.066667, 862.5, 0]
df.iloc[4] = [31, 74, 40, 2741.6, 23.4, 378.5, 1881.5, 458.1, 283.400000, 586.9, 0]
df.iloc[0, -1] = 0   # flood = 0 for row 0

base = os.path.dirname(os.path.abspath(__file__))

# Save as CSV
csv_path = os.path.join(base, "rainfall_india.csv")
df.to_csv(csv_path, index=False)
print(f"CSV saved  → {csv_path}")

# Save as Excel (matching project requirement: 'flood dataset.xlsx')
xlsx_path = os.path.join(base, "flood dataset.xlsx")
df.to_excel(xlsx_path, index=False)
print(f"XLSX saved → {xlsx_path}")

print(f"Shape: {df.shape}")
print(df.head())
print(f"\nFlood distribution:\n{df['flood'].value_counts()}")
