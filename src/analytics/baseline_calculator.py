import math
import pandas as pd


def calculate_baseline_cost(
    forecast_df,
    vehicle_df,
    distance_df
):
    """
    Baseline Cost Calculator — v3

    PDF Q&A-dan öyrənilənlər:
    1. Kullanım süresi = elleçleme + bekleme + yolculuk
    2. Baseline eyni qranularlıqda olmalıdır:
       09:00 + 17:00 tələbləri birləşdirilir (route×gün bazında)
    3. Ən böyük araçla, konsolidasiya/rental olmadan naive hesab
    """

    # =========================
    # VEHICLE INFO
    # =========================

    vehicle_info = {}

    for _, row in vehicle_df.iterrows():
        arac_adi = row.get("Araç Adı", row.get("arac_adi", ""))
        vehicle_info[arac_adi] = {
            "capacity":    row.get("Kapasite (desi)", 0),
            "spot_hourly": row.get("Spot Araç Saatlik Kira (TL)",
                           row.get("Spot Araç Sabit Günlük Maliyet (TL)", 0)),
            "spot_km":     row.get("Spot Kilometre Başına Maliyet (TL)", 0),
        }

    # =========================
    # DISTANCE LOOKUP
    # =========================

    # FIX: hər araç tipi üçün öz sürəti saxlanılır
    # (əvvəllər yalnız Tır sürəti bütün araç tipləri üçün
    # istifadə olunurdu — baseline-ı qeyri-dəqiq edirdi)
    VEHICLE_SPEED_COL = {
        "Tır":          "Tir_Suresi_Saat",
        "Kamyon":       "Kamyon_Suresi_Saat",
        "Hafif Kamyon": "Hafif_Kamyon_Suresi_Saat",
        "Kamyonet":     "Kamyonet_Suresi_Saat",
    }

    distance_lookup = {}

    for _, row in distance_df.iterrows():
        if "cikis" in row.index:
            key = (row["cikis"], row["varis"])
            distance_lookup[key] = {
                "km": row.get("mesafe_km", 0),
                "Tir_Suresi_Saat":          row.get("Tir_Suresi_Saat", 0),
                "Kamyon_Suresi_Saat":       row.get("Kamyon_Suresi_Saat", 0),
                "Hafif_Kamyon_Suresi_Saat": row.get("Hafif_Kamyon_Suresi_Saat", 0),
                "Kamyonet_Suresi_Saat":     row.get("Kamyonet_Suresi_Saat", 0),
            }
        elif "origin" in row.index:
            key = (row["origin"], row["destination"])
            fallback_hours = row.get("distance_km", 0) / 70
            distance_lookup[key] = {
                "km": row.get("distance_km", 0),
                "Tir_Suresi_Saat":          fallback_hours,
                "Kamyon_Suresi_Saat":       fallback_hours,
                "Hafif_Kamyon_Suresi_Saat": fallback_hours,
                "Kamyonet_Suresi_Saat":     fallback_hours,
            }

    # =========================
    # QRANULARLİQ DÜZƏLİŞİ — YENİ
    # 09:00 + 17:00 → route×gün bazında birləşdir
    # Optimallaşdırılmış plan ilə eyni qranularlıq
    # =========================

    desi_col = "Tahmin Edilen Desi" \
        if "Tahmin Edilen Desi" in forecast_df.columns \
        else "Tahminlenen Desi"

    group_cols = ["Çıkış Transfer Merkezi", "Varış Transfer Merkezi", "Tarih"]
    available = [c for c in group_cols if c in forecast_df.columns]

    if desi_col in forecast_df.columns and available:
        grouped = forecast_df.groupby(available)[desi_col].sum().reset_index()
        grouped.columns = list(available) + ["demand"]
    else:
        grouped = forecast_df.copy()
        grouped["demand"] = grouped.get(desi_col, 0)

    # =========================
    # BASELINE HESABI
    # Ən UCUZ uyğun araçla (real dispetçer seçimi), elleçleme+yolculuq daxil
    # =========================

    baseline_cost = 0.0

    for _, row in grouped.iterrows():
        origin      = row.get("Çıkış Transfer Merkezi", "")
        destination = row.get("Varış Transfer Merkezi", "")
        demand      = row.get("demand", 0)

        if demand <= 0:
            continue

        dist_info   = distance_lookup.get((origin, destination), {})
        distance_km = dist_info.get("km", 0)

        best_trip_cost = None

        for vtype, v_info in vehicle_info.items():
            capacity = v_info["capacity"]
            if capacity <= 0:
                continue

            # FIX: hər araç tipinin öz seyahat sürəti istifadə edilir
            speed_col    = VEHICLE_SPEED_COL.get(vtype, "Tir_Suresi_Saat")
            travel_hours = dist_info.get(speed_col, 0)

            vehicles_needed = math.ceil(demand / capacity)

            # Maliyet = saatlik_kira × (elleçleme + yolculuk) + km × km_maliyet
            # PDF: kullanım süresi = elleçleme + bekleme + yolculuk
            # Baseline-da bekleme yoxdur (konsolidasiya yox)
            ellecleme_hours = (demand * 0.01) / 60  # çıxış + varış
            usage_hours = ellecleme_hours + travel_hours + ellecleme_hours

            trip_cost = vehicles_needed * (
                v_info["spot_hourly"] * usage_hours
                + distance_km * v_info["spot_km"]
            )

            if best_trip_cost is None or trip_cost < best_trip_cost:
                best_trip_cost = trip_cost

        baseline_cost += best_trip_cost or 0.0

    return round(baseline_cost, 2)


def generate_cost_savings_report(baseline_cost, optimized_cost):
    """
    Baseline vs Optimized müqayisə hesabatı
    """

    savings = baseline_cost - optimized_cost

    savings_rate = round(
        (savings / baseline_cost * 100)
        if baseline_cost > 0 else 0,
        2
    )

    print("\n=====================================")
    print("AI COST OPTIMIZATION SAVINGS REPORT")
    print("=====================================")
    print(f"Baseline Cost (No Optimization): {round(baseline_cost, 2):,.2f} TL")
    print(f"Optimized Cost (NEURON AI):      {round(optimized_cost, 2):,.2f} TL")
    print(f"Total Savings:                   {round(savings, 2):,.2f} TL")
    print(f"Savings Rate:                    {savings_rate}%")
    print("=====================================")

    return {
        "baseline_cost":  baseline_cost,
        "optimized_cost": optimized_cost,
        "savings":        round(savings, 2),
        "savings_rate":   savings_rate
    }