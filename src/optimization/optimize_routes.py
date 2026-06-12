import math
import pandas as pd


def optimize_vehicle_plan(
    forecast_df,
    rental_df,
    vehicle_df,
    distance_df
):
    """
    Optimize logistics costs
    """

    results = []

    # Vehicle capacities
    vehicle_lookup = {}

    for _, row in vehicle_df.iterrows():

        vehicle_lookup[row["Araç Adı"]] = {
            "capacity": row["Kapasite (desi)"],
            "rental_daily": row["Kiralık Araç Günlük Kira (TL)"],
            "rental_km": row["Kiralık Araç Kilometre Başına Maliyet (TL)"],
            "spot_daily": row["Spot Araç Sabit Günlük Maliyet (TL)"],
            "spot_km": row["Spot Kilometre Başına Maliyet (TL)"],
        }

    # Iterate forecasts
    for _, row in forecast_df.iterrows():

        origin = row["Çıkış Transfer Merkezi"]
        destination = row["Varış Transfer Merkezi"]
        date = row["Tarih"]

        demand = row["Tahminlenen Desi"]

        remaining_demand = demand

        total_cost = 0

        # Distance lookup
        route_distance = distance_df[
            (distance_df["origin"] == origin)
            &
            (distance_df["destination"] == destination)
        ]

        if len(route_distance) == 0:
            continue

        distance_km = route_distance.iloc[0]["distance_km"]

        # RENTAL VEHICLES FIRST
        available_rentals = rental_df[
            (rental_df["Çıkış Transfer Merkezi"] == origin)
            &
            (rental_df["Varış Transfer Merkezi"] == destination)
        ]

        used_vehicles = []

        for _, rental in available_rentals.iterrows():

            vehicle_type = rental["Araç Türü"]
            vehicle_count = rental["Araç sayısı"]

            vehicle_info = vehicle_lookup[vehicle_type]

            capacity = vehicle_info["capacity"]

            total_capacity = capacity * vehicle_count

            # Rental Cost
            rental_cost = (
                vehicle_info["rental_daily"]
                +
                (distance_km * vehicle_info["rental_km"])
            ) * vehicle_count

            total_cost += rental_cost

            remaining_demand -= total_capacity

            used_vehicles.append(
                f"{vehicle_count}x Rental {vehicle_type}"
            )

        # SPOT VEHICLE OPTIMIZATION
        if remaining_demand > 0:

            best_vehicle = None
            best_cost_per_capacity = 999999

            for vehicle_name, info in vehicle_lookup.items():

                estimated_cost = (
                    info["spot_daily"]
                    +
                    (distance_km * info["spot_km"])
                )

                ratio = estimated_cost / info["capacity"]

                if ratio < best_cost_per_capacity:
                    best_cost_per_capacity = ratio
                    best_vehicle = vehicle_name

            best_info = vehicle_lookup[best_vehicle]

            needed_count = math.ceil(
                remaining_demand / best_info["capacity"]
            )

            spot_cost = (
                best_info["spot_daily"]
                +
                (distance_km * best_info["spot_km"])
            ) * needed_count

            total_cost += spot_cost

            used_vehicles.append(
                f"{needed_count}x Spot {best_vehicle}"
            )

        results.append({
            "Tarih": date,
            "Çıkış Transfer Merkezi": origin,
            "Varış Transfer Merkezi": destination,
            "Tahminlenen Desi": round(demand, 2),
            "Kullanılan Araçlar": ", ".join(used_vehicles),
            "Toplam Maliyet (TL)": round(total_cost, 2)
        })

    result_df = pd.DataFrame(results)

    print("\nOPTIMIZATION COMPLETED")

    print(result_df.head())

    return result_df