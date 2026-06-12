def predict_delay(
    weather_score,
    traffic_score,
    utilization
):

    delay_probability = (
        (weather_score * 0.4)
        +
        (traffic_score * 0.4)
        +
        ((1 - utilization) * 100 * 0.2)
    )

    if delay_probability >= 70:

        delay_level = "HIGH"

    elif delay_probability >= 40:

        delay_level = "MEDIUM"

    else:

        delay_level = "LOW"

    return {
        "delay_probability": round(
            delay_probability,
            2
        ),
        "delay_level": delay_level
    }