import pandas as pd


def calculate_risk_score(
    utilization,
    cost,
    distance
):
    """
    AI Risk Scoring Engine — Threshold-lar kalibre edildi.
    Lojistik sektoru standartlarına uyğun olaraq yüksək
    utilization-ı reward, orta məsafəni cəzalandırmayan
    balanslaşdırılmış risk modeli.
    """

    risk_score = 0

    # Low utilization risk — threshold aşağı çəkildi (0.50→0.45)
    # 0.45-0.70 arası normal lojistik əməliyyatdır
    if utilization < 0.45:
        risk_score += 35

    elif utilization < 0.65:
        risk_score += 15

    # High cost risk — threshold yuxarı çəkildi (25000→30000)
    # Uzun məsafəli route-lar üçün 25K normal xərcddir
    if cost > 35000:
        risk_score += 30

    elif cost > 22000:
        risk_score += 12

    # Long distance operational risk — yalnız həqiqətən uzun məsafələr
    # 700 km Türkiyə daxili normal magistral məsafədir
    if distance > 900:
        risk_score += 25

    elif distance > 600:
        risk_score += 8

    # Very low utilization critical risk
    if utilization < 0.35:
        risk_score += 20

    # Ultra long haul operation
    if distance > 1200:
        risk_score += 15

    # HIGH UTILIZATION REWARD — yeni əlavə
    # Yaxşı dolu maşınlara risk azaldılması
    if utilization >= 0.85:
        risk_score = max(0, risk_score - 15)

    elif utilization >= 0.75:
        risk_score = max(0, risk_score - 8)

    return min(risk_score, 100)


def detect_risk_level(score):
    """
    Convert numeric score to label — threshold-lar kalibre edildi.
    HIGH risk yalnız həqiqətən kritik əməliyyatlar üçün verilir.
    """

    # Orijinal: HIGH>=70, MEDIUM>=40
    # Yeni: HIGH>=75, MEDIUM>=45 → HIGH sayı azalır
    if score >= 75:
        return "HIGH"

    elif score >= 45:
        return "MEDIUM"

    return "LOW"


def analyze_shipment_risks(plan_df):
    """
    Main AI Risk Analyzer
    """

    df = plan_df.copy()

    risk_scores = []

    risk_levels = []

    ai_notes = []

    for _, row in df.iterrows():

        score = calculate_risk_score(
            row["Doluluk Oranı"],
            row["Toplam Maliyet"],
            row["Mesafe KM"]
        )

        level = detect_risk_level(score)

        note = []

        # AI explanation generation — threshold-lar kalibre edildi
        if row["Doluluk Oranı"] < 0.45:
            note.append("Low vehicle utilization")

        if row["Toplam Maliyet"] > 25000:
            note.append("High transportation cost")

        if row["Mesafe KM"] > 600:
            note.append("Long distance shipment")

        if row["Toplam Maliyet"] > 45000:
            note.append("Extreme logistics cost")

        if row["Mesafe KM"] > 1200:
            note.append("Ultra long haul route")

        if (
            score >= 80
            and
            len(note) >= 3
        ):
            note.append(
                "Immediate operational review recommended"
            )

        if len(note) == 0:
            note.append("Operationally efficient")

        risk_scores.append(score)

        risk_levels.append(level)

        ai_notes.append(
            " | ".join(note)
        )

    df["Risk Score"] = risk_scores

    df["Risk Level"] = risk_levels

    df["AI Insight"] = ai_notes

    print("\nAI RISK ANALYSIS COMPLETED")

    print(df.head())

    print(
        f"\nHIGH RISK SHIPMENTS: "
        f"{len(df[df['Risk Level'] == 'HIGH'])}"
    )

    print(
        f"MEDIUM RISK SHIPMENTS: "
        f"{len(df[df['Risk Level'] == 'MEDIUM'])}"
    )

    print(
        f"LOW RISK SHIPMENTS: "
        f"{len(df[df['Risk Level'] == 'LOW'])}"
    )

    average_risk = round(
        df["Risk Score"].mean(),
        2
    )

    print(
        f"AVERAGE RISK SCORE: "
        f"{average_risk}"
    )

    return df