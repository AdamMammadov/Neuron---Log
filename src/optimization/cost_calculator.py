def calculate_total_cost(
    distance_km,
    vehicle_count,
    fixed_cost,
    km_cost
):
    """
    Calculate shipment total cost
    """

    return (
        vehicle_count
        *
        (
            fixed_cost
            +
            (distance_km * km_cost)
        )
    )