import pandas as pd


def generate_ai_recommendations(
    plan_df
):

    recommendations = []

    # ==========================================
    # HIGH RISK SHIPMENTS
    # ==========================================

    high_risk_df = plan_df[
        plan_df["Risk Level"] == "HIGH"
    ]

    if len(high_risk_df) > 0:

        recommendations.append(
            {
                "Priority": "HIGH",
                "Category": "Risk",
                "Recommendation":
                "Increase operational monitoring for high-risk shipments."
            }
        )

    # ==========================================
    # LOW UTILIZATION
    # ==========================================

    low_util_df = plan_df[
        plan_df["Doluluk Oranı"] < 0.60
    ]

    if len(low_util_df) > 50:

        recommendations.append(
            {
                "Priority": "MEDIUM",
                "Category": "Fleet Efficiency",
                "Recommendation":
                "Optimize fleet allocation to reduce empty vehicle usage."
            }
        )

    # ==========================================
    # HIGH COST
    # ==========================================

    expensive_df = plan_df[
        plan_df["Toplam Maliyet"] > 20000
    ]

    if len(expensive_df) > 20:

        recommendations.append(
            {
                "Priority": "HIGH",
                "Category": "Cost Optimization",
                "Recommendation":
                "Review expensive shipment routes for consolidation opportunities."
            }
        )

    # ==========================================
    # DELAY RISK
    # ==========================================

    if "Delay Level" in plan_df.columns:

        delay_df = plan_df[
            plan_df["Delay Level"] == "HIGH"
        ]

        if len(delay_df) > 30:

            recommendations.append(
                {
                    "Priority": "HIGH",
                    "Category": "Delay Prevention",
                    "Recommendation":
                    "Activate proactive rerouting for delay-prone shipments."
                }
            )

    # ==========================================
    # ANOMALY DETECTION
    # ==========================================

    if "Anomaly" in plan_df.columns:

        anomaly_df = plan_df[
            plan_df["Anomaly"] == True
        ]

        if len(anomaly_df) > 10:

            recommendations.append(
                {
                    "Priority": "HIGH",
                    "Category": "AI Monitoring",
                    "Recommendation":
                    "Investigate operational anomalies detected by AI engine."
                }
            )

    # ==========================================
    # FINAL DATAFRAME
    # ==========================================

    recommendation_df = pd.DataFrame(
        recommendations
    )

    print("\nAI RECOMMENDATION ENGINE COMPLETED")

    print(recommendation_df)

    return recommendation_df