import pandas as pd


def generate_ai_summary(
    forecast_df,
    plan_df
):
    """
    AI Decision Engine
    """

    total_desi = round(
        forecast_df["Tahminlenen Desi"].sum(),
        2
    )

    total_cost = round(
        plan_df["Toplam Maliyet"].sum(),
        2
    )

    avg_utilization = round(
        plan_df["Doluluk Oranı"].mean(),
        2
    )

    total_shipments = len(plan_df)

    high_risk = len(
        plan_df[
            plan_df["Risk Level"] == "HIGH"
        ]
    )

    medium_risk = len(
        plan_df[
            plan_df["Risk Level"] == "MEDIUM"
        ]
    )

    low_risk = len(
        plan_df[
            plan_df["Risk Level"] == "LOW"
        ]
    )

    anomaly_count = len(

        plan_df[

            plan_df["Anomaly"] == True

        ]

    )

    avg_cost_per_shipment = round(

        total_cost /
        total_shipments,

        2

    )

    avg_cost_per_desi = round(

        total_cost /
        total_desi,

        4

    )

    print("\n=====================================")
    print("NEURON AI DECISION ENGINE")
    print("=====================================")

    print(f"Total Forecasted Desi: {total_desi}")

    print(f"Total Logistics Cost: {total_cost} TL")

    print(f"Average Vehicle Utilization: {avg_utilization}")

    print(f"Total Shipments: {total_shipments}")

    print("\nRISK DISTRIBUTION")

    print(f"HIGH RISK: {high_risk}")

    print(f"MEDIUM RISK: {medium_risk}")

    print(f"LOW RISK: {low_risk}")

    print(
        f"\nDetected Anomalies: "
        f"{anomaly_count}"
    )

    print(
        f"Average Shipment Cost: "
        f"{avg_cost_per_shipment} TL"
    )

    print(
        f"Average Cost Per Desi: "
        f"{avg_cost_per_desi} TL"
    )

    # AI business evaluation
    print("\nAI BUSINESS INSIGHTS")

    if avg_utilization >= 0.85:
        print("Fleet efficiency is excellent.")

    elif avg_utilization >= 0.70:
        print("Fleet efficiency is acceptable.")

    else:
        print("Fleet efficiency needs improvement.")

    if high_risk < 20:
        print("Operational risk is under control.")

    else:
        print("High operational risk detected.")

    if total_cost < 9000000:
        print("Logistics cost optimization successful.")

    else:
        print("Transportation costs are high.")