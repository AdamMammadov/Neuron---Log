import random


def calculate_weather_risk():

    risk_score = random.randint(0, 100)

    if risk_score >= 70:

        risk = "HIGH"

    elif risk_score >= 40:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    return {
        "weather_score": risk_score,
        "weather_risk": risk
    }