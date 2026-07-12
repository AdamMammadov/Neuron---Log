import pandas as pd
import numpy as np


def generate_future_predictions(
    model,
    historical_df,
    forecast_start="2026-06-29",
    forecast_days=7
):
    """
    Generate future demand forecasts.

    Gelişmiş Çözüm — Route-Specific + Global Hybrid:
    - Hər route üçün əvvəlcə route-specific model yoxlanır
    - Yoxdursa global ensemble istifadə edilir
    - 09:00 və 17:00 ayrı-ayrı forecast
    - Talep ID: D00001 formatı

    Recursive Forecasting — 3-Layer Smoothing:
    Layer 1: lag_1 = 70% pred + 30% hist_mean
    Layer 2: Outlier clipping (±2.5 std)
    Layer 3: Memory smoothing (80/20)
    """

    latest_data = historical_df.copy()

    sort_cols = [
        "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi",
        "Tarih"
    ]
    if "is_morning" in latest_data.columns:
        sort_cols.append("is_morning")

    latest_data = latest_data.sort_values(sort_cols)

    routes = latest_data[
        ["Çıkış Transfer Merkezi", "Varış Transfer Merkezi"]
    ].drop_duplicates()

    forecast_rows = []
    future_dates  = pd.date_range(start=forecast_start, periods=forecast_days)
    talep_counter = [1]

    saat_listesi = [
        {"saat_str": "09:00", "is_morning": 1},
        {"saat_str": "17:00", "is_morning": 0},
    ]

    # Route-specific models
    route_models = model.get("route_models", {}) if isinstance(model, dict) else {}

    # =====================================
    # ROUTE LOOP
    # =====================================

    for _, route in routes.iterrows():

        origin      = route["Çıkış Transfer Merkezi"]
        destination = route["Varış Transfer Merkezi"]
        route_key   = f"{origin}__{destination}"

        route_df = latest_data[
            (latest_data["Çıkış Transfer Merkezi"] == origin) &
            (latest_data["Varış Transfer Merkezi"] == destination)
        ].sort_values("Tarih")

        if len(route_df) < 2:
            continue

        # Route encoding
        if "Rota_Code" in route_df.columns:
            route_code = int(route_df.iloc[-1]["Rota_Code"])
        else:
            route_code = abs(hash(origin + "_" + destination)) % 100000

        route_fallback_mean = route_df["Toplam Desi"].mean()
        origin_map      = model.get("origin_target_map", {}) if isinstance(model, dict) else {}
        destination_map = model.get("destination_target_map", {}) if isinstance(model, dict) else {}
        origin_target_mean      = origin_map.get(origin, route_fallback_mean)
        destination_target_mean = destination_map.get(destination, route_fallback_mean)

        # Route-specific model varsa istifadə et
        use_route_model = route_key in route_models
        active_model    = route_models[route_key] if use_route_model else model

        # =====================================
        # SAAT BAZLI LOOP
        # =====================================

        for saat_info in saat_listesi:

            is_morning = saat_info["is_morning"]
            saat_str   = saat_info["saat_str"]

            # Saat bazlı history
            if "is_morning" in route_df.columns:
                route_saat_df = route_df[
                    route_df["is_morning"] == is_morning
                ].sort_values("Tarih")
            else:
                route_saat_df = route_df.sort_values("Tarih")

            if len(route_saat_df) < 7:
                route_saat_df = route_df.sort_values("Tarih")

            historical_mean = route_saat_df["Toplam Desi"].tail(14).mean()
            historical_std  = route_saat_df["Toplam Desi"].tail(14).std()

            if np.isnan(historical_std) or historical_std == 0:
                historical_std = historical_mean * 0.05

            route_target_mean = historical_mean
            history_values    = list(route_saat_df["Toplam Desi"].tail(30).values)

            # =====================================
            # FUTURE DATE LOOP
            # =====================================

            for step_idx, future_date in enumerate(future_dates, start=1):

                month       = future_date.month
                day         = future_date.day
                day_of_week = future_date.dayofweek
                week_of_year = int(future_date.isocalendar().week)
                is_weekend  = 1 if day_of_week in [5, 6] else 0

                ay       = month
                gun_hafta = day_of_week
                gun_ay   = day
                hafta_yil = week_of_year
                haftasonu = is_weekend

                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)
                day_sin   = np.sin(2 * np.pi * day_of_week / 7)
                day_cos   = np.cos(2 * np.pi * day_of_week / 7)
                ay_sin    = np.sin(2 * np.pi * ay / 12)
                ay_cos    = np.cos(2 * np.pi * ay / 12)

                holiday_list = (
                    model.get("holidays", [])
                    if isinstance(model, dict) else []
                )

                days_to_holiday = 999
                if len(holiday_list) > 0:
                    nearest = min(abs((h - future_date).days) for h in holiday_list)
                    days_to_holiday = min(nearest, 30)

                is_public_holiday = int(future_date in holiday_list)
                before_holiday    = int((future_date + pd.Timedelta(days=1)) in holiday_list)
                after_holiday     = int((future_date - pd.Timedelta(days=1)) in holiday_list)

                season    = month % 12 // 3
                is_winter = int(season == 0)
                is_spring = int(season == 1)
                is_summer = int(season == 2)
                is_autumn = int(season == 3)

                is_month_start = int(future_date.is_month_start)
                is_month_end   = int(future_date.is_month_end)

                # =====================================
                # RECURSIVE LAGS — SAAT BAZLI
                # =====================================

                lag_1 = (history_values[-1] * 0.70) + (historical_mean * 0.30)
                lag_2 = history_values[-2] if len(history_values) >= 2 else lag_1
                lag_3 = history_values[-3] if len(history_values) >= 3 else lag_2
                lag_7 = (
                    (history_values[-7] * 0.80) + (historical_mean * 0.20)
                    if len(history_values) >= 7 else lag_3
                )
                lag_14 = history_values[-14] if len(history_values) >= 14 else lag_7

                rolling_mean_3  = np.mean(history_values[-3:])
                rolling_mean_7  = np.mean(history_values[-7:])
                rolling_mean_14 = np.mean(history_values[-14:])

                rolling_std_3  = np.std(history_values[-3:])
                rolling_std_7  = np.std(history_values[-7:])
                rolling_std_14 = np.std(history_values[-14:])

                rolling_max_3  = np.max(history_values[-3:])
                rolling_max_7  = np.max(history_values[-7:])
                rolling_max_14 = np.max(history_values[-14:])

                lag_diff_1_7  = lag_1 - lag_7
                lag_ratio_1_7 = lag_1 / (lag_7 + 1)
                rolling_ratio = rolling_mean_3 / (rolling_mean_14 + 1)

                # =====================================
                # FEATURE ROW
                # =====================================

                feature_row = pd.DataFrame([{
                    "month": month, "day": day,
                    "day_of_week": day_of_week,
                    "week_of_year": week_of_year,
                    "is_weekend": is_weekend,
                    "is_morning": is_morning,
                    "month_sin": month_sin, "month_cos": month_cos,
                    "day_sin": day_sin, "day_cos": day_cos,
                    "ay": ay, "gun_hafta": gun_hafta,
                    "gun_ay": gun_ay, "hafta_yil": hafta_yil,
                    "haftasonu": haftasonu,
                    "ay_sin": ay_sin, "ay_cos": ay_cos,
                    "is_public_holiday": is_public_holiday,
                    "before_holiday": before_holiday,
                    "after_holiday": after_holiday,
                    "days_to_holiday": days_to_holiday,
                    "is_month_start": is_month_start,
                    "is_month_end": is_month_end,
                    "season": season,
                    "is_winter": is_winter, "is_spring": is_spring,
                    "is_summer": is_summer, "is_autumn": is_autumn,
                    "Rota_Code": route_code,
                    "Origin_Target_Enc": origin_target_mean,
                    "Destination_Target_Enc": destination_target_mean,
                    "Route_Target_Enc": route_target_mean,
                    "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
                    "lag_7": lag_7, "lag_14": lag_14,
                    "rolling_mean_3": rolling_mean_3,
                    "rolling_mean_7": rolling_mean_7,
                    "rolling_mean_14": rolling_mean_14,
                    "rolling_std_3": rolling_std_3,
                    "rolling_std_7": rolling_std_7,
                    "rolling_std_14": rolling_std_14,
                    "rolling_max_3": rolling_max_3,
                    "rolling_max_7": rolling_max_7,
                    "rolling_max_14": rolling_max_14,
                    "lag_diff_1_7": lag_diff_1_7,
                    "lag_ratio_1_7": lag_ratio_1_7,
                    "rolling_ratio": rolling_ratio,
                }])

                # =====================================
                # FEATURE ALIGNMENT
                # =====================================

                if isinstance(active_model, dict):
                    feature_names = active_model["features"]
                elif hasattr(active_model, "feature_names_"):
                    feature_names = active_model.feature_names_
                else:
                    feature_names = feature_row.columns.tolist()

                for col in feature_names:
                    if col not in feature_row.columns:
                        feature_row[col] = 0

                feature_row = feature_row[feature_names]

                # =====================================
                # ENSEMBLE PREDICTION
                # Route-specific model varsa onu istifadə et
                # =====================================

                if isinstance(active_model, dict):
                    lgb_pred = active_model["lgbm"].predict(feature_row)[0]
                    xgb_pred = active_model["xgb"].predict(feature_row)[0]
                    cat_pred = active_model["cat"].predict(feature_row)[0]

                    lgb_w = active_model["weights"]["lgb_weight"]
                    xgb_w = active_model["weights"]["xgb_weight"]
                    cat_w = active_model["weights"]["cat_weight"]

                    pred_log = lgb_w*lgb_pred + xgb_w*xgb_pred + cat_w*cat_pred
                else:
                    pred_log = active_model.predict(feature_row)[0]

                prediction = np.expm1(pred_log)

                # =====================================
                # OUTLIER CONTROL
                # =====================================

                current_std = historical_std if historical_std > 0 \
                    else (historical_mean * 0.05)
                upper_limit = historical_mean + (2.5 * current_std)
                lower_limit = max(0, historical_mean - (2.0 * current_std))

                if historical_std > 0:
                    prediction = np.clip(prediction, lower_limit, upper_limit)

                prediction = max(0, prediction)
                prediction = min(prediction, historical_mean * 3.0)

                # =====================================
                # UPDATE RECURSIVE MEMORY
                # =====================================

                smoothed = (prediction * 0.80) + (historical_mean * 0.20)
                history_values.append(smoothed)
                history_values = history_values[-30:]

                # =====================================
                # FORECAST CONFIDENCE
                # =====================================

                base_variance_ratio = historical_std / max(historical_mean, 1)
                dynamic_confidence = (
                    92.5
                    - (base_variance_ratio * 15.0)
                    - (step_idx * 2.15)
                )
                dynamic_confidence = max(61.4, min(96.8, dynamic_confidence))

                # =====================================
                # TALEP ID
                # =====================================

                talep_id = f"D{talep_counter[0]:05d}"
                talep_counter[0] += 1

                forecast_rows.append({
                    "Talep ID":                talep_id,
                    "Tarih":                   future_date.strftime("%Y-%m-%d"),
                    "Talep Tamamlama Saati":  saat_str,
                    "Çıkış Transfer Merkezi":  origin,
                    "Varış Transfer Merkezi":  destination,
                    "Tahmin Edilen Desi":      round(prediction, 2),
                    "Forecast Confidence":     round(dynamic_confidence, 2),
                    "Model Type": "Route-Specific" if use_route_model else "Global",
                })

    # =====================================
    # FINAL
    # =====================================

    forecast_df = pd.DataFrame(forecast_rows)

    route_specific_count = len(forecast_df[forecast_df["Model Type"] == "Route-Specific"]) \
        if "Model Type" in forecast_df.columns else 0
    global_count = len(forecast_df) - route_specific_count

    print("\nRECURSIVE FUTURE DEMAND PREDICTIONS GENERATED")
    print(f"  Method     : Route-Specific + Global Hybrid")
    print(f"  Horizon    : {forecast_days} days from {forecast_start}")
    print(f"  Saatler    : 09:00 ve 17:00 (ayrı ayrı)")
    print(f"  Route-Specific forecasts : {route_specific_count}")
    print(f"  Global fallback forecasts: {global_count}")
    print(f"  Total Rows : {len(forecast_df)}")
    print(f"  Talep ID   : D00001 → D{talep_counter[0]-1:05d}")

    return forecast_df