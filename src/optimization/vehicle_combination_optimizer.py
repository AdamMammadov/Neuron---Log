import math
import itertools

# GLOBAL CACHE: Kombinasyonların təkrar-təkrar hesablanmasını əngəlləmək üçün
_COMBINATION_CACHE = None


def calculate_driver_penalty(
    distance
):

    if distance > 900:
        return 4000

    if distance > 450:
        return 2000

    return 0


def calculate_delay_penalty(
    distance
):

    if distance > 700:
        return 3000

    if distance > 400:
        return 1500

    return 0


def generate_vehicle_combinations(vehicle_info):
    global _COMBINATION_CACHE

    # Əgər kombinasiyalar əvvəlcədən hesablanıbsa, yeniden hesablama, birbaşa yaddaşdan oxu
    if _COMBINATION_CACHE is not None:
        return _COMBINATION_CACHE

    vehicle_types = list(vehicle_info.keys())
    combinations = []

    # CRITICAL FIX: range(1, 5) limiti range(1, 6) edilərək kombinasiya genişliyi artırıldı.
    for r in range(1, 6):
        combos = itertools.combinations_with_replacement(
            vehicle_types,
            r
        )
        combinations.extend(combos)

    # Hesablanan kombinasiyaları keşe yaz
    _COMBINATION_CACHE = combinations
    return combinations


def evaluate_combination(
    combo,
    demand,
    distance,
    vehicle_info
):

    total_capacity = 0

    total_cost = 0

    vehicles_used = []

    for vehicle_type in combo:

        info = vehicle_info[vehicle_type]

        capacity = info["capacity"]

        trip_cost = (
            info["spot_daily"] +
            (
                distance *
                info["spot_km"]
            )
        )

        total_capacity += capacity

        total_cost += trip_cost

        vehicles_used.append(vehicle_type)

    # DYNAMIC SCALING FOR ULTRA HIGH DEMAND
    # FIX: Əgər kombinasiyanın ümumi tutumu yükdən kiçikdirsə, 'None' qaytarıb sistemi fallback-ə
    # məcbur etmək əvəzinə, kombinasiyanı riyazi çarpanla çoxaldaraq yükə tam uyğunlaşdırırıq.
    if total_capacity < demand:
        multiplier = math.ceil(demand / total_capacity)
        total_capacity *= multiplier
        total_cost *= multiplier
        vehicles_used = vehicles_used * multiplier

    utilization = demand / total_capacity

    utilization = min(
        utilization,
        1
    )

    empty_penalty = (
        1 - utilization
    ) * 5000

    oversize_penalty = 0

    if total_capacity > (
        demand * 2
    ):
        oversize_penalty = 3000

    driver_penalty = (
        calculate_driver_penalty(
            distance
        )
    )

    delay_penalty = (
        calculate_delay_penalty(
            distance
        )
    )

    utilization_bonus = (
        utilization * 2500
    )

    # CRITICAL RIYAZI FIX: Virtual cəzaların real daşıma xərclərini (total_cost) manipulyasiya etməsi əngəlləndi.
    # Cəzalar qorundu, lakin çarpan təsiri 0.1-ə salınaraq sistem maliyyət baxımından optimallaşdırıldı.
    final_score = (

        total_cost

        +

        empty_penalty

        +

        oversize_penalty

        +

        (driver_penalty * 0.1)

        +

        (delay_penalty * 0.1)

        -

        utilization_bonus

    )

    return {
        "vehicles": vehicles_used,
        "capacity": total_capacity,
        "cost": total_cost,
        "utilization": utilization,
        "score": final_score,
        "driver_penalty": driver_penalty,
        "delay_penalty": delay_penalty,
        "oversize_penalty": oversize_penalty
    }


def find_best_vehicle_combination(
    demand,
    distance,
    vehicle_info
):

    combinations = generate_vehicle_combinations(
        vehicle_info
    )

    best_option = None

    best_score = float("inf")

    for combo in combinations:

        result = evaluate_combination(
            combo,
            demand,
            distance,
            vehicle_info
        )

        if result is None:
            continue

        if result["score"] < best_score:

            best_score = result["score"]

            best_option = result

    return best_option
