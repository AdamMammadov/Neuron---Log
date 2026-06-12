import random


def calculate_traffic_risk():

    traffic_score = random.randint(0, 100)

    if traffic_score >= 70:

        risk = "HIGH"

    elif traffic_score >= 40:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    return {
        "traffic_score": traffic_score,
        "traffic_risk": risk
    }