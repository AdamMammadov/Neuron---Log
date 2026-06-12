import pandas as pd


def generate_ai_recommendations(plan_df):

    recommendations = []

    # =====================================
    # HIGH COST ROUTES
    # =====================================

    expensive_routes = (
        plan_df
        .groupby(
            [
                "Çıkış Transfer Merkezi",
                "Varış Transfer Merkezi"
            ]
        )["Toplam Maliyet"]
        .sum()
        .reset_index()
    )

    expensive_routes = expensive_routes.sort_values(
        by="Toplam Maliyet",
        ascending=False
    ).head(3)

    for _, row in expensive_routes.iterrows():

        recommendations.append(
            f"""
High transportation cost detected:
{row['Çıkış Transfer Merkezi']}
→
{row['Varış Transfer Merkezi']}

Recommendation:
Use regional consolidation strategy.
"""
        )

    # =====================================
    # LOW UTILIZATION
    # =====================================

    low_util = plan_df[
        plan_df["Doluluk Oranı"] < 0.50
    ]

    if len(low_util) > 0:

        recommendations.append(
            f"""
{len(low_util)} shipments detected with low
vehicle utilization.

Recommendation:
Merge low-volume shipments into shared routes.
"""
        )

    # =====================================
    # HIGH RISK
    # =====================================

    high_risk = plan_df[
        plan_df["Risk Level"] == "HIGH"
    ]

    if len(high_risk) > 20:

        recommendations.append(
            """
Operational risk level is high.

Recommendation:
Increase rental fleet allocation
for unstable shipment corridors.
"""
        )

    # =====================================
    # COST EFFICIENCY
    # =====================================

    avg_cost = plan_df[
        "Toplam Maliyet"
    ].mean()

    if avg_cost < 15000:

        recommendations.append(
            """
AI detected strong logistics
cost efficiency performance.
"""
        )

    return recommendations