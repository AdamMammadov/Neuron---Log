import pandas as pd


def generate_smart_kpis(forecast_df, plan_df):
    """
    Smart KPI Engine — Gelişmiş Çözüm.
    Yeni KPI-lar: SLA compliance, route coverage,
    saat bazlı analiz, araç tipi breakdown.
    """

    print("\n=====================================")
    print("SMART KPI ENGINE")
    print("=====================================")

    # =========================================
    # FORECAST KPI-LAR
    # =========================================

    desi_col = "Tahmin Edilen Desi" \
        if "Tahmin Edilen Desi" in forecast_df.columns \
        else "Tahminlenen Desi"

    total_forecast = round(forecast_df[desi_col].sum(), 2) \
        if desi_col in forecast_df.columns else 0

    # Saat bazlı breakdown
    morning_forecast = 0
    evening_forecast = 0
    if "Talep Tamamlama Saati" in forecast_df.columns and desi_col in forecast_df.columns:
        morning_mask     = forecast_df["Talep Tamamlama Saati"].astype(str).str.strip().isin(["9:00", "09:00"])
        morning_forecast = round(forecast_df.loc[morning_mask, desi_col].sum(), 2)
        evening_forecast = round(forecast_df.loc[~morning_mask, desi_col].sum(), 2)

    # =========================================
    # OPTİMİZASİYA KPI-LAR
    # =========================================

    total_cost      = round(plan_df["Toplam Maliyet"].sum(), 2) \
        if "Toplam Maliyet" in plan_df.columns else 0

    total_sla       = round(plan_df["SLA cezası"].sum(), 2) \
        if "SLA cezası" in plan_df.columns else 0

    total_combined  = round(total_cost + total_sla, 2)

    avg_utilization = round(plan_df["Doluluk Oranı"].mean(), 2) \
        if "Doluluk Oranı" in plan_df.columns else 0

    shipment_count  = len(plan_df)

    avg_cost_per_shipment = round(total_combined / shipment_count, 2) \
        if shipment_count > 0 else 0

    avg_cost_per_desi = round(total_combined / total_forecast, 4) \
        if total_forecast > 0 else 0

    # SLA compliance
    sla_violations  = len(plan_df[plan_df["SLA cezası"] > 0]) \
        if "SLA cezası" in plan_df.columns else 0

    sla_compliance  = round(
        (1 - sla_violations / shipment_count) * 100, 2
    ) if shipment_count > 0 else 100.0

    # Risk breakdown
    high_risk   = len(plan_df[plan_df["Risk Level"] == "HIGH"]) \
        if "Risk Level" in plan_df.columns else 0
    medium_risk = len(plan_df[plan_df["Risk Level"] == "MEDIUM"]) \
        if "Risk Level" in plan_df.columns else 0
    low_risk    = len(plan_df[plan_df["Risk Level"] == "LOW"]) \
        if "Risk Level" in plan_df.columns else 0

    # Anomaly
    anomaly_count = len(plan_df[plan_df["Anomaly"] == True]) \
        if "Anomaly" in plan_df.columns else 0

    # Araç tipi breakdown
    kiralik_count = len(plan_df[plan_df["Araç Tipi"] == "Kiralık"]) \
        if "Araç Tipi" in plan_df.columns else 0
    spot_count    = len(plan_df[plan_df["Araç Tipi"] == "Spot"]) \
        if "Araç Tipi" in plan_df.columns else 0

    # =========================================
    # CARBON / CO2 OPTIMIZATION — DÜZƏLDİLDİ
    # KÖK PROBLEM: Gelişmiş Çözüm-də bir Araç ID artıq
    # BİRDƏN ÇOX sətirdə görünə bilər (hər konsolidə
    # edilmiş Talep ID öz sətrini alır — bax:
    # optimize_shipments_advanced.py, Talep ID format
    # düzəlişi). plan_df.iterrows() bunu nəzərə almadan
    # HƏR SƏTIR üçün km×factor əlavə edirdi — nəticədə
    # eyni fiziki aracın CO2-si 5-7 dəfə TƏKRAR sayılırdı
    # (sınaqda: 142.66 ton → 1165 ton, ~8x səhv artım).
    # FIX: hər Araç ID YALNIZ 1 dəfə sayılsın deyə,
    # əvvəlcə araç-səviyyəli (unique) alt-cədvəl yaradılır.
    # =========================================

    CO2_FACTOR_G_PER_KM = {
        "Tır":          900,   # böyük yük tırı
        "Kamyon":       650,
        "Hafif Kamyon": 450,
        "Kamyonet":     280,
    }

    if "Araç ID" in plan_df.columns:
        vehicle_level_df = plan_df.drop_duplicates(subset="Araç ID")
    else:
        vehicle_level_df = plan_df

    total_co2_kg = 0.0

    if "Araç türü" in vehicle_level_df.columns and "Mesafe KM" in vehicle_level_df.columns:
        for _, row in vehicle_level_df.iterrows():
            vtype   = row.get("Araç türü", "")
            km      = row.get("Mesafe KM", 0)
            n_araç  = row.get("Araç Sayısı", 1) if "Araç Sayısı" in vehicle_level_df.columns else 1
            factor  = CO2_FACTOR_G_PER_KM.get(vtype, 600)
            total_co2_kg += (km * factor * n_araç) / 1000

    total_co2_tons = round(total_co2_kg / 1000, 3)

    # Consolidation Rate — DÜZƏLDİLDİ
    # Əvvəllər shipment_count (indi Talep-ID-səviyyəli sətir
    # sayı) istifadə olunurdu — bu, forecast sətir sayından
    # (4046) BÖYÜK ola bildiyi üçün mənfi nəticə verirdi
    # (sınaqda: -56.52%). Düzgün metrik: unikal ARAÇ sayı
    # (neçə fiziki vasitə işlədildi) forecast sətir sayına
    # qarşı — "neçə az avtomobillə eyni işi gördük" mənasını
    # verir.
    unique_vehicle_count = (
        vehicle_level_df["Araç ID"].nunique()
        if "Araç ID" in plan_df.columns else shipment_count
    )
    baseline_shipment_estimate = len(forecast_df) if not forecast_df.empty else unique_vehicle_count
    consolidation_ratio = 1 - (unique_vehicle_count / baseline_shipment_estimate) \
        if baseline_shipment_estimate > 0 else 0
    co2_reduction_pct = round(consolidation_ratio * 100, 2)

    # =========================================
    # PRINT
    # =========================================

    print(f"\nFORECAST KPIs")
    print(f"  Total Forecasted Desi : {total_forecast}")
    print(f"  09:00 Desi            : {morning_forecast}")
    print(f"  17:00 Desi            : {evening_forecast}")

    print(f"\nOPTIMIZATION KPIs")
    print(f"  Vehicle Cost          : {total_cost} TL")
    print(f"  SLA Penalty           : {total_sla} TL")
    print(f"  Total Combined Cost   : {total_combined} TL")
    print(f"  Avg Utilization       : {avg_utilization}")
    print(f"  Shipment Count (rows) : {shipment_count}  (Talep-ID səviyyəli sətir sayı)")
    print(f"  Unique Vehicles       : {unique_vehicle_count}  (fiziki araç sayı)")
    print(f"  Avg Cost/Shipment     : {avg_cost_per_shipment} TL")
    print(f"  Avg Cost/Desi         : {avg_cost_per_desi} TL")

    print(f"\nSLA & COMPLIANCE")
    print(f"  SLA Violations        : {sla_violations}")
    print(f"  SLA Compliance Rate   : {sla_compliance}%")

    print(f"\nRISK BREAKDOWN")
    print(f"  HIGH Risk             : {high_risk}")
    print(f"  MEDIUM Risk           : {medium_risk}")
    print(f"  LOW Risk              : {low_risk}")
    print(f"  Anomalies             : {anomaly_count}")

    print(f"\nVEHICLE BREAKDOWN")
    print(f"  Kiralık               : {kiralik_count}")
    print(f"  Spot                  : {spot_count}")

    print(f"\nCARBON / CO2 OPTIMIZATION")
    print(f"  Total CO2 Emission    : {total_co2_tons} tons")
    print(f"  Consolidation Rate    : {round(consolidation_ratio*100, 2)}%")
    print(f"  Est. CO2 Reduction    : {co2_reduction_pct}% (vs. no consolidation)")

    print("\nKPI ENGINE COMPLETED")

    return {
        "total_forecast":       total_forecast,
        "morning_forecast":     morning_forecast,
        "evening_forecast":     evening_forecast,
        "total_cost":           total_cost,
        "total_sla":            total_sla,
        "total_combined":       total_combined,
        "avg_utilization":      avg_utilization,
        "shipment_count":       shipment_count,
        "avg_cost_per_shipment": avg_cost_per_shipment,
        "avg_cost_per_desi":    avg_cost_per_desi,
        "sla_violations":       sla_violations,
        "sla_compliance":       sla_compliance,
        "high_risk":            high_risk,
        "medium_risk":          medium_risk,
        "low_risk":             low_risk,
        "anomaly_count":        anomaly_count,
        "kiralik_count":        kiralik_count,
        "spot_count":           spot_count,
        "unique_vehicle_count": unique_vehicle_count,
        "total_co2_tons":       total_co2_tons,
        "co2_reduction_pct":    co2_reduction_pct,
    }