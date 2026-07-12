import pandas as pd


def calculate_risk_score(utilization, cost, distance):
    """
    AI Risk Scoring Engine — Gelişmiş Çözüm versiyonu v2.
    Threshold-lar sistemin real ortalama utilization-ına
    (≈0.69, sərbəst SLA-bazlı konsolidasiyanın nəticəsi)
    uyğun kalibre edildi. Bug fix: cost sütun adı düzəldildi
    ("Toplam maliyet" → "Toplam Maliyet") — əvvəllər maliyet
    riski heç vaxt hesablanmırdı.
    """
    risk_score = 0

    # Low utilization risk — 0.45/0.65 idi, sistemin real
    # ortalaması (0.69) nəzərə alınaraq aşağı çəkildi
    if utilization < 0.35:
        risk_score += 30
    elif utilization < 0.50:
        risk_score += 12

    # High cost risk — bug fix sonrası real dəyərlərlə işləyir
    if cost > 40000:
        risk_score += 25
    elif cost > 25000:
        risk_score += 10

    # Long distance risk
    if distance > 900:
        risk_score += 20
    elif distance > 600:
        risk_score += 6

    # Very low utilization critical
    if utilization < 0.25:
        risk_score += 15

    # Ultra long haul
    if distance > 1200:
        risk_score += 12

    # High utilization reward — daha geniş aralıqda mükafat
    if utilization >= 0.80:
        risk_score = max(0, risk_score - 15)
    elif utilization >= 0.65:
        risk_score = max(0, risk_score - 8)

    return min(risk_score, 100)


def detect_risk_level(score):
    if score >= 80:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    return "LOW"


def analyze_shipment_risks(plan_df):
    """
    Main AI Risk Analyzer — Gelişmiş Çözüm.
    Yeni sütunlar: Araç ID, SLA cezası, Yolculuk süresi
    """
    df = plan_df.copy()

    risk_scores = []
    risk_levels = []
    ai_notes    = []

    for _, row in df.iterrows():

        score = calculate_risk_score(
            row.get("Doluluk Oranı", 0),
            row.get("Toplam Maliyet", 0),
            row.get("Mesafe KM", 0)
        )

        level = detect_risk_level(score)
        note  = []

        if row.get("Doluluk Oranı", 0) < 0.35:
            note.append("Low vehicle utilization")

        if row.get("Toplam Maliyet", 0) > 30000:
            note.append("High transportation cost")

        if row.get("Mesafe KM", 0) > 600:
            note.append("Long distance shipment")

        if row.get("Toplam Maliyet", 0) > 50000:
            note.append("Extreme logistics cost")

        if row.get("Mesafe KM", 0) > 1200:
            note.append("Ultra long haul route")

        # SLA cezası varsa risk notu əlavə et
        if row.get("SLA cezası", 0) > 0:
            note.append("SLA violation detected")

        if score >= 80 and len(note) >= 3:
            note.append("Immediate operational review recommended")

        if len(note) == 0:
            note.append("Operationally efficient")

        risk_scores.append(score)
        risk_levels.append(level)
        ai_notes.append(" | ".join(note))

    df["Risk Score"] = risk_scores
    df["Risk Level"] = risk_levels
    df["AI Insight"] = ai_notes

    high   = len(df[df["Risk Level"] == "HIGH"])
    medium = len(df[df["Risk Level"] == "MEDIUM"])
    low    = len(df[df["Risk Level"] == "LOW"])
    avg    = round(df["Risk Score"].mean(), 2)

    print("\nAI RISK ANALYSIS COMPLETED")
    print(f"  HIGH RISK    : {high}")
    print(f"  MEDIUM RISK  : {medium}")
    print(f"  LOW RISK     : {low}")
    print(f"  AVERAGE SCORE: {avg}")

    return df