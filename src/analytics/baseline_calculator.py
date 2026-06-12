import math
import pandas as pd


def calculate_baseline_cost(
    forecast_df,
    vehicle_df,
    distance_df
):
    """
    Baseline Cost Calculator

    Konsolidasiya və rental priority olmadan,
    hər shipment üçün sadəcə ən böyük spot araç
    ilə hesablanan naive maliyet.

    Bu rəqəm optimizasiyamızın nə qədər qənaət
    etdiyini göstərmək üçün istifadə edilir.
    """

    # =========================
    # VEHICLE INFO
    # =========================

    vehicle_info = {}

    for _, row in vehicle_df.iterrows():

        vehicle_info[row["Araç Adı"]] = {
            "capacity": row["Kapasite (desi)"],
            "spot_daily": row["Spot Araç Sabit Günlük Maliyet (TL)"],
            "spot_km": row["Spot Kilometre Başına Maliyet (TL)"]
        }

    # =========================
    # DISTANCE LOOKUP
    # =========================

    distance_lookup = {}

    for _, row in distance_df.iterrows():

        key = (
            row["origin"],
            row["destination"]
        )

        distance_lookup[key] = row["distance_km"]

    # =========================
    # BASELINE: Ən böyük araç seç,
    # konsolidasiya yox, rental yox
    # =========================

    # Araçları kapasitəyə görə sırala
    sorted_vehicles = sorted(
        vehicle_info.items(),
        key=lambda x: x[1]["capacity"],
        reverse=True
    )

    baseline_cost = 0.0

    for _, row in forecast_df.iterrows():

        origin = row["Çıkış Transfer Merkezi"]

        destination = row["Varış Transfer Merkezi"]

        demand = row["Tahminlened Desi"] \
            if "Tahminlened Desi" in row \
            else row.get("Tahminlenen Desi", 0)

        distance = distance_lookup.get(
            (origin, destination),
            0
        )

        if demand <= 0:
            continue

        # Tələbatı tam daşıyacaq ən ucuz spot araç kombinasiyası
        # Baseline: heç bir optimizasiya yoxdur —
        # sadəcə ən böyük araçla bölüb yuxarı yuvarlaqlaşdır
        remaining = demand

        for v_name, v_info in sorted_vehicles:

            capacity = v_info["capacity"]

            if capacity <= 0:
                continue

            vehicles_needed = math.ceil(
                remaining / capacity
            )

            trip_cost = vehicles_needed * (
                v_info["spot_daily"]
                + distance * v_info["spot_km"]
            )

            baseline_cost += trip_cost

            break  # Yalnız ən böyük araç — naive baseline

    return round(baseline_cost, 2)


def generate_cost_savings_report(
    baseline_cost,
    optimized_cost
):
    """
    Baseline vs Optimized müqayisə hesabatı
    """

    savings = baseline_cost - optimized_cost

    savings_rate = round(
        (savings / baseline_cost * 100)
        if baseline_cost > 0 else 0,
        2
    )

    print(
        "\n====================================="
    )

    print(
        "AI COST OPTIMIZATION SAVINGS REPORT"
    )

    print(
        "====================================="
    )

    print(
        f"Baseline Cost (No Optimization): "
        f"{round(baseline_cost, 2):,.2f} TL"
    )

    print(
        f"Optimized Cost (NEURON AI):      "
        f"{round(optimized_cost, 2):,.2f} TL"
    )

    print(
        f"Total Savings:                   "
        f"{round(savings, 2):,.2f} TL"
    )

    print(
        f"Savings Rate:                    "
        f"{savings_rate}%"
    )

    print(
        "====================================="
    )

    return {
        "baseline_cost": baseline_cost,
        "optimized_cost": optimized_cost,
        "savings": round(savings, 2),
        "savings_rate": savings_rate
    }