import math


def select_best_vehicle(
    demand,
    vehicle_df
):
    """
    Select cheapest vehicle combination
    """

    best_option = None
    lowest_cost = float("inf")

    for _, row in vehicle_df.iterrows():

        vehicle_type = row["Araç Adı"]
        capacity = row["Kapasite (desi)"]
        fixed_cost = row["Spot Araç Sabit Günlük Maliyet (TL)"]

        needed_vehicle_count = math.ceil(
            demand / capacity
        )

        total_cost = needed_vehicle_count * fixed_cost

        utilization = demand / (
            needed_vehicle_count * capacity
        )

        if total_cost < lowest_cost:

            lowest_cost = total_cost

            best_option = {
                "vehicle_type": vehicle_type,
                "vehicle_count": needed_vehicle_count,
                "total_cost": total_cost,
                "utilization": round(utilization, 2),
                "capacity": capacity
            }

    return best_option