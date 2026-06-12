from src.realtime.weather_risk import (
    calculate_weather_risk
)

from src.realtime.traffic_risk import (
    calculate_traffic_risk
)

from src.realtime.delay_predictor import (
    predict_delay
)


def run_realtime_engine(plan_df):

    weather = []

    traffic = []

    delay_probability = []

    delay_level = []

    for _, row in plan_df.iterrows():

        weather_data = calculate_weather_risk()

        traffic_data = calculate_traffic_risk()

        delay_data = predict_delay(
            weather_data["weather_score"],
            traffic_data["traffic_score"],
            row["Doluluk Oranı"]
        )

        weather.append(
            weather_data["weather_risk"]
        )

        traffic.append(
            traffic_data["traffic_risk"]
        )

        delay_probability.append(
            delay_data["delay_probability"]
        )

        delay_level.append(
            delay_data["delay_level"]
        )

    plan_df["Weather Risk"] = weather

    plan_df["Traffic Risk"] = traffic

    plan_df["Delay Probability"] = delay_probability

    plan_df["Delay Level"] = delay_level

    print("\nREAL-TIME AI ENGINE COMPLETED")

    print(
        plan_df[
            [
                "Weather Risk",
                "Traffic Risk",
                "Delay Probability",
                "Delay Level"
            ]
        ].head()
    )

    return plan_df