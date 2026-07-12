"""
BACKTEST - Real Forecast Accuracy Validation

Meqsed: "WAPE 29.07%" reqemi yalniz telim CV-den gelir -
gizli (gorunmeyen) dataya qarshi real proqnoz performansini
bilmirik. Bu skript:

1. Son 7 gunu (22-28 Iyun) "gizledir" (holdout)
2. Modeli YALNIZ qalan datada oyredir (production-daki
   EYNI train_forecast_model funksiyasi ile)
3. generate_future_predictions-in oz recursive mentiqi ile
   gizledilmish 7 gunu proqnozlashdirir (production-daki EYNI
   kod, sadece ferqli tarix araligi)
4. Proqnozu REAL (gizledilmish) deyerlerle muqayise edir
5. Gun-gun WAPE bolgusu ile recursive error accumulation
   fərziyyəsini reqemle tesdiqleyir

Bu, production kodunun DAVRANISHINI deyishdirmir - sadece
onu gorunmeyen dataya qarshi sinayir, evvelceden (juriden
evvel) durust menzere verir.

Ishletmek uchun: python backtest_forecast.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from src.data_loader.load_data import load_all_data
from src.data_loader.preprocess import preprocess_demand
from src.features.build_features import (
    create_time_features,
    create_advanced_features
)
from src.models.train_forecast_model import train_forecast_model
from src.models.predict_future import generate_future_predictions


def run_backtest(holdout_days=7):

    print("=" * 60)
    print("BACKTEST - REAL FORECAST ACCURACY VALIDATION")
    print("=" * 60)
    print(
        "Bu skript modeli KECMISH datada oyredir, sonra son "
        + str(holdout_days) + " gunu (gizli) proqnozlashdirir ve REAL "
        "deyerlerle muqayise edir. Production kodu deyishmir."
    )

    # =========================================
    # 1. RAW DATA YUKLE VE PREPROCESS ET
    # (production main.py ile EYNI addimlar)
    # =========================================

    datasets = load_all_data()
    demand_df = preprocess_demand(datasets["desi_talep"])
    demand_df["Tarih"] = pd.to_datetime(demand_df["Tarih"])

    max_date = demand_df["Tarih"].max()
    cutoff_date = max_date - pd.Timedelta(days=holdout_days)

    # =========================================
    # 2. SPLIT - HOLDOUT-U "GIZLE"
    # Feature engineering-den EVVEL bolunur ki, lag/rolling
    # feature-lar YALNIZ gorunen (train) datadan hesablansin -
    # holdout-un hech bir melumati modele sizmasin.
    # =========================================

    train_raw = demand_df[demand_df["Tarih"] <= cutoff_date].copy()
    holdout_raw = demand_df[demand_df["Tarih"] > cutoff_date].copy()

    print("")
    print("Full data range   : " + str(demand_df["Tarih"].min().date()) + " -> " + str(max_date.date()))
    print("Train (visible)   : " + str(train_raw["Tarih"].min().date()) + " -> " + str(train_raw["Tarih"].max().date()) + " (" + str(len(train_raw)) + " rows)")
    print("Holdout (hidden)  : " + str(holdout_raw["Tarih"].min().date()) + " -> " + str(holdout_raw["Tarih"].max().date()) + " (" + str(len(holdout_raw)) + " rows)")

    if len(holdout_raw) == 0:
        print("XETA: Holdout dovrunde data yoxdur - cutoff tarixini yoxla.")
        return None

    # =========================================
    # 3. FEATURE ENGINEERING - YALNIZ TRAIN DATADA
    # =========================================

    print("")
    print("Feature engineering (train data only)...")
    train_data = create_time_features(train_raw)
    train_data = create_advanced_features(train_data)

    # =========================================
    # 4. MODEL TELIMI - YALNIZ GORUNEN DATADA
    # (production-daki EYNI train_forecast_model funksiyasi)
    # =========================================

    print("")
    print("Training model on TRAIN data only (holdout is invisible)...")
    print("-" * 60)
    model = train_forecast_model(train_data)
    print("-" * 60)

    # =========================================
    # 5. HOLDOUT DOVRUNU PROQNOZLA
    # (production-daki EYNI generate_future_predictions,
    # sadece ferqli forecast_start)
    # =========================================

    forecast_start = (cutoff_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print("")
    print("Generating forecast for hidden period starting " + forecast_start + "...")

    forecast_df = generate_future_predictions(
        model,
        train_data,
        forecast_start=forecast_start,
        forecast_days=holdout_days
    )

    # =========================================
    # 6. PROQNOZU REAL DEYERLERLE MUQAYISE ET
    # =========================================

    if "is_morning" not in holdout_raw.columns:
        if "Saat" in holdout_raw.columns:
            holdout_raw["is_morning"] = holdout_raw["Saat"].astype(str).str.startswith("9").astype(int)
        else:
            holdout_raw["is_morning"] = 1

    holdout_raw["match_key"] = (
        holdout_raw["Çıkış Transfer Merkezi"].astype(str) + "__" +
        holdout_raw["Varış Transfer Merkezi"].astype(str) + "__" +
        holdout_raw["Tarih"].dt.strftime("%Y-%m-%d") + "__" +
        holdout_raw["is_morning"].astype(str)
    )

    forecast_df["is_morning_calc"] = (
        forecast_df["Talep Tamamlama Saati"].astype(str)
        .str.startswith("9").astype(int)
    )
    forecast_df["match_key"] = (
        forecast_df["Çıkış Transfer Merkezi"].astype(str) + "__" +
        forecast_df["Varış Transfer Merkezi"].astype(str) + "__" +
        forecast_df["Tarih"].astype(str) + "__" +
        forecast_df["is_morning_calc"].astype(str)
    )

    merged = forecast_df.merge(
        holdout_raw[["match_key", "Toplam Desi", "Tarih"]],
        on="match_key",
        how="inner",
        suffixes=("", "_actual")
    )

    print("")
    print("Matched " + str(len(merged)) + " forecast-vs-actual pairs (out of " + str(len(forecast_df)) + " forecasts, " + str(len(holdout_raw)) + " actuals)")

    if len(merged) == 0:
        print("XETA: Hech bir uygunluq tapilmadi - tarix/route uygunlugunu yoxla.")
        return None

    # =========================================
    # 7. REAL MAE / WAPE
    # =========================================

    actual = merged["Toplam Desi"].values
    predicted = merged["Tahmin Edilen Desi"].values

    real_mae = mean_absolute_error(actual, predicted)
    mean_actual = np.mean(actual)
    real_wape = (real_mae / mean_actual * 100) if mean_actual > 0 else 0

    cv_mae = model.get("cv_mae", 0) if isinstance(model, dict) else 0

    q1 = train_data["Toplam Desi"].quantile(0.10)
    q3 = train_data["Toplam Desi"].quantile(0.90)
    iqr = q3 - q1
    cleaned_mean = train_data[
        (train_data["Toplam Desi"] >= q1 - 1.2 * iqr) &
        (train_data["Toplam Desi"] <= q3 + 1.2 * iqr)
    ]["Toplam Desi"].mean()
    cv_wape = (cv_mae / cleaned_mean * 100) if cv_mae and cleaned_mean else 0

    print("")
    print("=" * 60)
    print("REAL BACKTEST RESULTS (previously unseen data)")
    print("=" * 60)
    print("Real MAE            : " + format(real_mae, ".2f"))
    print("Real WAPE           : " + format(real_wape, ".2f") + "%")
    print("Mean Actual Demand  : " + format(mean_actual, ".2f"))
    print("Matched Samples     : " + str(len(merged)))

    # =====================================
    # GUN-GUN WAPE BOLGUSU
    # Recursive error accumulation ferziyyesini reqemle
    # tesdiqleyir: xeta 7 gun boyunca nece artir?
    # Movcud "merged" dataframe-inin ustune - hech bir
    # modele/pipeline-a toxunulmur, yalniz analiz.
    # =====================================

    print("")
    print("DAY-BY-DAY WAPE BREAKDOWN (recursive error accumulation):")
    print("  " + "Day".ljust(6) + "Date".ljust(14) + "MAE".ljust(12) + "WAPE".ljust(10) + "Samples")

    daily_wapes = []
    for day_date, day_group in merged.groupby("Tarih"):
        day_actual = day_group["Toplam Desi"].values
        day_predicted = day_group["Tahmin Edilen Desi"].values
        day_mae = mean_absolute_error(day_actual, day_predicted)
        day_mean = np.mean(day_actual)
        day_wape = (day_mae / day_mean * 100) if day_mean > 0 else 0
        daily_wapes.append((day_date, day_mae, day_wape, len(day_group)))

    for i, (day_date, day_mae, day_wape, n) in enumerate(daily_wapes, start=1):
        row = "  Day " + str(i).ljust(3) + str(day_date).ljust(14) + format(day_mae, ".2f").ljust(12) + format(day_wape, ".2f").ljust(9) + "% " + str(n)
        print(row)

    # SAMPLE SIZE DIAGNOSTIC - YENI
    # Day 7 anomal az sample-a malikdirse, WAPE hesabi
    # statistik baximdan etibarsiz ola biler (yuksek varyans,
    # kicik n). Bunu yoxlamaq uchun holdout_raw-daki her gunun
    # HEQIQI (butun) route/saat sayini de gosteririk.
    print("")
    print("SAMPLE SIZE DIAGNOSTIC (per day in holdout_raw, before matching):")
    for day_date, day_group in holdout_raw.groupby(holdout_raw["Tarih"].dt.date):
        n_morning = len(day_group[day_group["is_morning"] == 1])
        n_evening = len(day_group[day_group["is_morning"] == 0])
        print("  " + str(day_date) + " : total=" + str(len(day_group)) + " (09:00=" + str(n_morning) + ", 17:00=" + str(n_evening) + ")")

    if len(daily_wapes) >= 2:
        sample_sizes = [n for (_, _, _, n) in daily_wapes]
        min_n = min(sample_sizes)
        max_n = max(sample_sizes)

        first_day_wape = daily_wapes[0][2]
        last_day_wape = daily_wapes[-1][2]
        print("")
        print("  First day WAPE : " + format(first_day_wape, ".2f") + "%  (closest to 1-step-ahead CV)")
        print("  Last day WAPE  : " + format(last_day_wape, ".2f") + "%  (7-step recursive drift)")

        if max_n > 0 and min_n / max_n < 0.5:
            print("  WARNING: Sample sizes vary significantly across days")
            print("  (min=" + str(min_n) + ", max=" + str(max_n) + ") - WAPE comparison")
            print("  across days may be statistically unreliable where n is small.")
            print("  Interpret day-by-day trend with caution, especially for")
            print("  days with n well below the typical daily count.")

        if last_day_wape > first_day_wape:
            gap_days = last_day_wape - first_day_wape
            print("  -> Error appears to grow " + format(gap_days, ".2f") + "pp over the horizon,")
            print("     consistent with (but not statistically proven by) the")
            print("     recursive error accumulation hypothesis - see sample")
            print("     size warning above before treating this as confirmed.")
        else:
            print("  -> Error does NOT clearly increase over horizon -")
            print("     accumulation hypothesis not confirmed by this run.")

    print("")
    print("COMPARISON WITH TRAINING CV ESTIMATE:")
    print("  Training CV WAPE  : " + format(cv_wape, ".2f") + "%")
    print("  Real Backtest WAPE: " + format(real_wape, ".2f") + "%")

    diff = abs(real_wape - cv_wape)
    print("  Difference        : " + format(diff, ".2f") + " percentage points")

    if diff < 5:
        print("  -> CONSISTENT: CV estimate is reliable, no meaningful gap.")
    elif diff < 10:
        print("  -> MODERATE GAP: CV estimate is somewhat optimistic/pessimistic.")
    else:
        print("  -> LARGE GAP: Real performance differs significantly from CV.")
        print("     This is a known limitation of recursive multi-step")
        print("     forecasting (see day-by-day breakdown above) - not a bug.")

    print("=" * 60)

    return {
        "real_mae": round(real_mae, 2),
        "real_wape": round(real_wape, 2),
        "cv_wape": round(cv_wape, 2),
        "gap": round(diff, 2),
        "matched_samples": len(merged),
        "daily_wapes": daily_wapes,
    }


if __name__ == "__main__":
    run_backtest(holdout_days=7)