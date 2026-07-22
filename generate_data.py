"""
Generates three datasets for the project:
1. india_fuel_prices.csv   - daily petrol/diesel prices, 3 metro cities, 2021-2026
2. us_fuel_prices.csv      - weekly gasoline prices, 2021-2026
3. india_bike_resale.csv   - ~6000 used bike listings with a built-in, lagged
                             relationship to fuel price spikes for high-mileage bikes

Calibrated to real published anchor points (as of mid-2026):
- Delhi petrol ~Rs 107.24, diesel ~Rs 95.97
- Mumbai petrol ~Rs 113.12, diesel ~Rs 104.00
- Bengaluru petrol ~Rs 110.98, diesel ~Rs 101.86
- US regular gasoline historically ranges ~$2.20-$5.00/gallon 2021-2026 (COVID dip,
  2022 spike after Ukraine war, gradual moderation after)

This is SYNTHETIC data shaped to match real-world patterns - not scraped or
copied from any dataset. Clearly disclose this in the app / README.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(7)

# ============================================================
# 1. INDIA FUEL PRICES (daily, 3 cities, Jan 2021 - Jun 2026)
# ============================================================
start = datetime(2021, 1, 1)
end = datetime(2026, 6, 30)
days = (end - start).days + 1
dates = [start + timedelta(days=i) for i in range(days)]

# City base offsets calibrated to real anchor points (Bengaluru as base ~110.98)
city_offsets_petrol = {"Delhi": -3.74, "Mumbai": 2.14, "Bengaluru": 0.0}
city_offsets_diesel = {"Delhi": -5.89, "Mumbai": 2.14, "Bengaluru": 0.0}

# Build a national trend line ending near Bengaluru's real anchor values
# Petrol: from ~85 (Jan 2021) to ~111 (mid-2026), with a few real-world-style shocks
n = len(dates)
t = np.linspace(0, 1, n)
trend_petrol = 85 + t * (110.98 - 85)
trend_diesel = 78 + t * (101.86 - 78)

# Add known-shape shocks: 2022 crude oil spike (~day 400-500), a tax-cut dip (~day 620),
# and small day-to-day noise (since 2017 India revises prices daily)
shock = np.zeros(n)
spike_start, spike_end = 380, 520
shock[spike_start:spike_end] += np.linspace(0, 6, spike_end - spike_start)
shock[spike_end:spike_end+150] += np.linspace(6, 3, 150)
dip_start = 600
shock[dip_start:dip_start+60] -= np.linspace(0, 4, 60)
shock[dip_start+60:dip_start+300] -= 4

noise = np.random.normal(0, 0.15, n).cumsum() * 0.05  # small random walk, dampened
noise = noise - noise.mean()

rows = []
for city in ["Delhi", "Mumbai", "Bengaluru"]:
    petrol = trend_petrol + shock + noise + city_offsets_petrol[city]
    diesel = trend_diesel + shock * 0.9 + noise * 0.9 + city_offsets_diesel[city]
    for i, d in enumerate(dates):
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "city": city,
            "petrol_price": round(float(petrol[i]), 2),
            "diesel_price": round(float(diesel[i]), 2),
        })

india_fuel = pd.DataFrame(rows)
india_fuel.to_csv("india_fuel_prices.csv", index=False)
print(f"india_fuel_prices.csv -> {india_fuel.shape}")

# ============================================================
# 2. US FUEL PRICES (weekly, national average, 2021-2026)
# ============================================================
us_dates = pd.date_range(start, end, freq="7D")
m = len(us_dates)
tu = np.linspace(0, 1, m)

# Real-world shape: ~$2.3 (early 2021, COVID low) -> spike to ~$5.0 (mid-2022,
# Ukraine war) -> moderates to ~$3.3-3.6 range by 2025-26
us_price = 2.3 + np.zeros(m)
covid_recovery_end = int(m * 0.08)
us_price[:covid_recovery_end] = np.linspace(2.3, 2.9, covid_recovery_end)

spike_s = covid_recovery_end
spike_e = int(m * 0.28)
us_price[spike_s:spike_e] = np.linspace(2.9, 5.0, spike_e - spike_s)

decline_e = int(m * 0.45)
us_price[spike_e:decline_e] = np.linspace(5.0, 3.6, decline_e - spike_e)

us_price[decline_e:] = np.linspace(3.6, 3.35, m - decline_e)

us_noise = np.random.normal(0, 0.06, m)
us_price = us_price + us_noise

us_fuel = pd.DataFrame({
    "week_date": us_dates.strftime("%Y-%m-%d"),
    "avg_gasoline_price_usd_per_gallon": np.round(us_price, 3),
})
us_fuel.to_csv("us_fuel_prices.csv", index=False)
print(f"us_fuel_prices.csv -> {us_fuel.shape}")

# ============================================================
# 3. INDIA USED BIKE RESALE LISTINGS (~6000 rows)
# ============================================================
# (model, mileage_kmpl, new_price_in_lakhs) - approx real-world ex-showroom prices
models = [
    ("Hero Splendor Plus", 65, 0.75),
    ("Honda Activa 6G", 55, 0.80),
    ("Bajaj Pulsar 150", 45, 1.10),
    ("TVS Jupiter", 52, 0.78),
    ("Royal Enfield Classic 350", 35, 2.00),
    ("Yamaha FZ-S", 42, 1.15),
    ("Hero HF Deluxe", 68, 0.65),
    ("Suzuki Access 125", 50, 0.85),
    ("Bajaj Platina", 62, 0.72),
    ("KTM Duke 200", 32, 1.90),
]
cities = ["Delhi", "Mumbai", "Bengaluru"]
owners = ["1st Owner", "2nd Owner", "3rd Owner"]

# map bike -> city fuel index over time (for the lag effect)
fuel_by_city_date = india_fuel.set_index(["city", "date"])

n_listings = 6000
bike_rows = []
listing_dates = pd.to_datetime(np.random.choice(dates[365:], n_listings))  # skip first year (no lag history)

for i in range(n_listings):
    model, mileage, base_price_lakh = models[np.random.randint(len(models))]
    city = np.random.choice(cities)
    owner = np.random.choice(owners, p=[0.55, 0.32, 0.13])
    listing_date = listing_dates[i]
    bike_age_years = np.random.uniform(0.5, 8)
    kms_driven = int(bike_age_years * np.random.uniform(4000, 9000))

    base_price = base_price_lakh * 100  # convert lakhs -> thousands of rupees (1 lakh = 100k)
    # depreciation curve
    depreciated = base_price * (0.88 ** bike_age_years)
    # owner penalty
    depreciated *= {"1st Owner": 1.0, "2nd Owner": 0.92, "3rd Owner": 0.85}[owner]
    # kms penalty
    depreciated *= max(0.6, 1 - (kms_driven / 200000))

    # ---- THE CORE SIGNAL: fuel price spike -> resale premium for high-mileage bikes ----
    # look up petrol price ~45 days before listing date (lag effect) vs ~1 year before that
    lag_date = listing_date - timedelta(days=45)
    baseline_date = lag_date - timedelta(days=365)
    lag_date_str = lag_date.strftime("%Y-%m-%d")
    baseline_date_str = baseline_date.strftime("%Y-%m-%d")

    try:
        recent_price = fuel_by_city_date.loc[(city, lag_date_str), "petrol_price"]
        year_ago_price = fuel_by_city_date.loc[(city, baseline_date_str), "petrol_price"]
        pct_change = (recent_price - year_ago_price) / year_ago_price
    except KeyError:
        pct_change = 0.0

    # mileage_kmpl > 55 = "fuel efficient" bikes get a resale premium scaled to fuel price rise
    if mileage >= 55:
        fuel_premium = 1 + max(0, pct_change) * 0.6  # up to ~+ a few % when fuel spikes
    else:
        fuel_premium = 1 + max(0, pct_change) * 0.08  # low-mileage bikes barely affected

    final_price = depreciated * fuel_premium * np.random.uniform(0.95, 1.05)  # noise

    bike_rows.append({
        "listing_date": listing_date.strftime("%Y-%m-%d"),
        "model_name": model,
        "city": city,
        "mileage_kmpl": mileage,
        "age_years": round(bike_age_years, 1),
        "kms_driven": kms_driven,
        "owner": owner,
        "price_inr_thousands": round(final_price, 1),
        "fuel_pct_change_1yr": round(pct_change * 100, 2),
    })

bikes = pd.DataFrame(bike_rows)
bikes.to_csv("india_bike_resale.csv", index=False)
print(f"india_bike_resale.csv -> {bikes.shape}")
print("\nSample:")
print(bikes.head())
