import pandas as pd


def detect_operational_anomalies(plan_df):
    """
    AI Anomaly Detection Engine — Gelişmiş Çözüm.

    Dizayn Qərarı — Anomaly vs Delay/Risk Ayrılığı:
    - OPERASIONAL anomaliya: aşağı utilization + yüksək xərc + risk
    - SLA cezası: ayrıca xarici risk göstəricisi
    - Delay: real-time hava/trafik faktoru

    Anomaly=0 + SLA=0 → sistem mükəmməl işləyir.
    """

    anomaly_flags   = []
    anomaly_scores  = []
    anomaly_reasons = []

    for _, row in plan_df.iterrows():

        score   = 0
        reasons = []

        utilization = row.get("Doluluk Oranı", 0)
        cost        = row.get("Toplam Maliyet", 0)
        risk        = row.get("Risk Level", "LOW")
        sla_penalty = row.get("SLA cezası", 0)

        # Low utilization
        if utilization < 0.35:
            score += 35
            reasons.append("Low vehicle utilization")

        # High cost
        if cost > 32000:
            score += 25
            reasons.append("High transportation cost")

        # High risk
        if risk == "HIGH":
            score += 18
            reasons.append("High operational risk")

        # SLA cezası — Gelişmiş Çözüm yeniliyi
        if sla_penalty > 0:
            score += 30
            reasons.append(f"SLA violation: {sla_penalty} TL")

        # Combined critical
        if utilization < 0.35 and cost > 45000:
            score += 20
            reasons.append("Expensive underutilized shipment")

        # Anomaly threshold
        anomaly = score >= 60

        if anomaly and score >= 80:
            reasons.append("Immediate operational intervention required")

        anomaly_flags.append(anomaly)
        anomaly_scores.append(score)
        anomaly_reasons.append(
            ", ".join(reasons) if reasons else "Operationally normal"
        )

    plan_df["Anomaly"]        = anomaly_flags
    plan_df["Anomaly Score"]  = anomaly_scores
    plan_df["Anomaly Reason"] = anomaly_reasons

    total     = len(plan_df[plan_df["Anomaly"] == True])
    avg_score = round(plan_df["Anomaly Score"].mean(), 2)
    critical  = len(plan_df[plan_df["Anomaly Score"] >= 80])

    print("\nAI ANOMALY DETECTION COMPLETED")
    print(f"  TOTAL ANOMALIES : {total}")
    print(f"  CRITICAL        : {critical}")
    print(f"  AVG SCORE       : {avg_score}")

    # Anomaly vs SLA/Delay izahı
    if total == 0:
        print("\n  ANOMALY vs SLA/DELAY DESIGN NOTE:")
        print("  Anomaly = internal ops failure (utilization+cost+risk)")
        print("  SLA     = external constraint (time-based delivery)")
        print("  Both 0  → System operating at peak efficiency")

    return plan_df