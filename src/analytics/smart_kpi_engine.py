import pandas as pd


def generate_smart_kpis(
    forecast_df,
    plan_df
):

    print("\n=====================================")
    print("SMART KPI ENGINE")
    print("=====================================")

    total_forecast = round(
        forecast_df[
            "Tahminlenen Desi"
        ].sum(),
        2
    )

    total_cost = round(
        plan_df[
            "Toplam Maliyet"
        ].sum(),
        2
    )

    avg_utilization = round(
        plan_df[
            "Doluluk Oranı"
        ].mean(),
        2
    )

    shipment_count = len(
        plan_df
    )

    avg_cost_per_shipment = round(
        total_cost / shipment_count,
        2
    )

    avg_cost_per_desi = round(
        total_cost / total_forecast,
        4
    )

    high_risk_count = len(
        plan_df[
            plan_df["Risk Level"] == "HIGH"
        ]
    )

    anomaly_count = 0

    if "Anomaly" in plan_df.columns:

        anomaly_count = len(
            plan_df[
                plan_df["Anomaly"] == True
            ]
        )

    print(f"Total Forecasted Desi: {total_forecast}")

    print(f"Total Logistics Cost: {total_cost} TL")

    print(f"Average Vehicle Utilization: {avg_utilization}")

    print(f"Shipment Count: {shipment_count}")

    print(f"Average Cost Per Shipment: {avg_cost_per_shipment} TL")

    print(f"Average Cost Per Desi: {avg_cost_per_desi} TL")

    print(f"High Risk Shipment Count: {high_risk_count}")

    print(f"Detected Anomalies: {anomaly_count}")

    print("\nKPI ENGINE COMPLETED")

    return {
        "total_forecast": total_forecast,
        "total_cost": total_cost,
        "avg_utilization": avg_utilization,
        "shipment_count": shipment_count,
        "avg_cost_per_shipment": avg_cost_per_shipment,
        "avg_cost_per_desi": avg_cost_per_desi,
        "high_risk_count": high_risk_count,
        "anomaly_count": anomaly_count
    }