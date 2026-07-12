"""
DIGITAL TWIN DEMO - "What-If" Scenario Simulation

BONUS / DEMO SCRIPT - resmi teslimatin bir parcasi DEGIL.
main.py, Talep-tahmini.xlsx, Tasima-plani.xlsx dosyalarina
HICBIR sekilde dokunmaz.

Senaryo: Bilecik transfer merkezi (elleceme + tir kapasitesi)
tamamen kapatilirsa ne olur? Sistem, ONCEDEN HESAPLANMIS
forecast'i tekrar KULLANARAK (model egitimi TEKRARLANMAZ,
yalnizca optimizasyon adimi yeniden calisir - saniyeler
suruyor), yeni bir plan uretir ve ORIJINAL plan ile
karsilastirir.

GUVENLIK KURALLARI (ihlal edilmez):
  1. Orijinal data dosyalari (data/raw/*) SADECE OKUNUR,
     hicbir zaman diske geri yazilmaz.
  2. outputs/Talep-tahmini.xlsx ve outputs/Tasima-plani.xlsx
     ASLA ezilmez - demo ciktisi ayrica dosya adina yazilir.
  3. main.py'a dokunulmaz - bu tamamen bagimsiz bir scripttir.

Calistirmak icin: python digital_twin_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import time
import pandas as pd

from src.data_loader.load_data import load_all_data
from src.optimization.optimize_shipments_advanced import optimize_shipments_advanced

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)

# Resmi teslimat dosyalarindan FARKLI, acikca "demo" etiketli isimler
DEMO_OUTPUT_FILE = OUTPUT_PATH / "demo_Bilecik_kapali_plan.xlsx"

CLOSED_TM_NAME = "Bilecik"


def run_digital_twin_demo():

    print("=" * 60)
    print("DIGITAL TWIN DEMO - 'What-If' Scenario Simulation")
    print("=" * 60)
    print(
        "BU BIR DEMO SCRIPTIDIR - resmi teslimatin parcasi degildir.\n"
        "main.py, Talep-tahmini.xlsx, Tasima-plani.xlsx dosyalarina\n"
        "dokunulmaz. Orijinal veri dosyalari yalnizca okunur."
    )

    # =========================================
    # 1. MEVCUT (ONCEDEN URETILMIS) FORECAST'I OKU
    # Model egitimi TEKRARLANMAZ - bu, demo'nun hizli
    # olmasinin (saniyeler) sebebidir.
    # =========================================

    forecast_file = OUTPUT_PATH / "Talep-tahmini.xlsx"

    if not forecast_file.exists():
        print(
            "\nHATA: outputs/Talep-tahmini.xlsx bulunamadi. "
            "Once 'python main.py' calistirip resmi forecast'i "
            "uretmelisiniz - bu demo o ciktiyi yeniden kullanir."
        )
        return None

    print(f"\nMevcut forecast okunuyor: {forecast_file}")
    forecast_df = pd.read_excel(forecast_file)

    # export_results.py formati sadece 6 sutun icerir
    # (Talep ID, Tarih, Talep Tamamlama Saati, Cikis TM,
    #  Varis TM, Tahmin Edilen Desi) - optimizer'in
    # bekledigi sutun isimleriyle birebir uyumludur.

    print(f"  Toplam Forecast Satiri: {len(forecast_df)}")

    # =========================================
    # 2. RAW DATA YUKLE (SADECE OKUNUR)
    # =========================================

    print("\nOrijinal veri dosyalari yukleniyor (yalnizca okuma)...")
    datasets = load_all_data()

    rental_df       = datasets["kiralik_araclar"].copy()
    vehicle_df      = datasets["arac_kapasite"].copy()
    handling_df     = datasets["ellecleme_kapasite"].copy()
    distance_df     = datasets["sehirler_arasi"].copy()
    tir_kapasite_df = datasets["tir_kapasite"].copy()

    # =========================================
    # 3. BASELINE (ORIJINAL) OPTIMIZASYON — MEVCUT KISITLARLA
    # =========================================

    print("\n" + "-" * 60)
    print("BASELINE: Mevcut kisitlarla optimizasyon calistiriliyor...")
    t0 = time.time()

    baseline_plan = optimize_shipments_advanced(
        forecast_df=forecast_df,
        rental_df=rental_df,
        vehicle_df=vehicle_df,
        distance_df=distance_df,
        handling_df=handling_df,
        tir_kapasite_df=tir_kapasite_df,
    )

    baseline_time = time.time() - t0
    baseline_cost = baseline_plan["Toplam Maliyet"].sum() if not baseline_plan.empty else 0
    baseline_sla  = baseline_plan["SLA cezası"].sum() if not baseline_plan.empty else 0
    baseline_util = baseline_plan["Doluluk Oranı"].mean() if not baseline_plan.empty else 0

    print(f"  Sure           : {baseline_time:.2f} saniye")
    print(f"  Toplam Sevkiyat: {len(baseline_plan)}")
    print(f"  Arac Maliyeti  : {baseline_cost:,.2f} TL")
    print(f"  SLA Cezasi     : {baseline_sla:,.2f} TL")
    print(f"  Ort. Doluluk   : {baseline_util:.2f}")

    # =========================================
    # 4. "WHAT-IF": BILECIK TM'I TAMAMEN KAPAT
    # Sadece BELLEKTEKI kopya degistirilir - diske
    # HICBIR ZAMAN geri yazilmaz.
    # =========================================

    print("\n" + "-" * 60)
    print(f"SENARYO: '{CLOSED_TM_NAME}' transfer merkezi KAPATILIYOR")
    print(f"  (elleçleme kapasitesi -> 0, tir kapasitesi -> 0)")
    print("  (Bu degisiklik yalnizca bellekte yapilir, hicbir")
    print("   orijinal dosyaya YAZILMAZ)")

    handling_df_demo = handling_df.copy()
    tir_kapasite_df_demo = tir_kapasite_df.copy()

    if "transfer_merkezi" in handling_df_demo.columns:
        mask = handling_df_demo["transfer_merkezi"] == CLOSED_TM_NAME
        if mask.sum() == 0:
            print(
                f"  UYARI: '{CLOSED_TM_NAME}' handling_df icinde bulunamadi "
                "- isim eslesmesini kontrol edin."
            )
        handling_df_demo.loc[mask, "ellecleme_kapasite"] = 0

    if "transfer_merkezi" in tir_kapasite_df_demo.columns:
        mask = tir_kapasite_df_demo["transfer_merkezi"] == CLOSED_TM_NAME
        tir_kapasite_df_demo.loc[mask, "tir_kapasitesi"] = 0

    # =========================================
    # 5. YENI KISITLARLA OPTIMIZASYONU TEKRAR CALISTIR
    # (model egitimi YOK - yalnizca optimizasyon, hizli)
    # =========================================

    print(f"\nYeni kisitlarla optimizasyon yeniden calistiriliyor...")
    t0 = time.time()

    demo_plan = optimize_shipments_advanced(
        forecast_df=forecast_df,
        rental_df=rental_df,
        vehicle_df=vehicle_df,
        distance_df=distance_df,
        handling_df=handling_df_demo,
        tir_kapasite_df=tir_kapasite_df_demo,
    )

    demo_time = time.time() - t0
    demo_cost = demo_plan["Toplam Maliyet"].sum() if not demo_plan.empty else 0
    demo_sla  = demo_plan["SLA cezası"].sum() if not demo_plan.empty else 0
    demo_util = demo_plan["Doluluk Oranı"].mean() if not demo_plan.empty else 0

    print(f"  Sure           : {demo_time:.2f} saniye")
    print(f"  Toplam Sevkiyat: {len(demo_plan)}")
    print(f"  Arac Maliyeti  : {demo_cost:,.2f} TL")
    print(f"  SLA Cezasi     : {demo_sla:,.2f} TL")
    print(f"  Ort. Doluluk   : {demo_util:.2f}")

    # =========================================
    # 6. KARSILASTIRMA
    # =========================================

    print("\n" + "=" * 60)
    print("KARSILASTIRMA: Baseline vs 'Bilecik Kapali' Senaryosu")
    print("=" * 60)

    cost_diff = demo_cost - baseline_cost
    sla_diff  = demo_sla  - baseline_sla
    cost_pct  = (cost_diff / baseline_cost * 100) if baseline_cost else 0

    print(f"{'Metrik':<25}{'Baseline':>15}{'Bilecik Kapali':>18}{'Fark':>15}")
    print(f"{'Arac Maliyeti (TL)':<25}{baseline_cost:>15,.2f}{demo_cost:>18,.2f}{cost_diff:>+15,.2f}")
    print(f"{'SLA Cezasi (TL)':<25}{baseline_sla:>15,.2f}{demo_sla:>18,.2f}{sla_diff:>+15,.2f}")
    print(f"{'Ort. Doluluk':<25}{baseline_util:>15.2f}{demo_util:>18.2f}{demo_util-baseline_util:>+15.2f}")
    print(f"{'Sevkiyat Sayisi':<25}{len(baseline_plan):>15}{len(demo_plan):>18}{len(demo_plan)-len(baseline_plan):>+15}")

    print(f"\nMaliyet degisimi: {cost_pct:+.2f}%")
    print(f"Optimizasyon suresi: {demo_time:.2f} saniye (model egitimi olmadan)")

    # =========================================
    # 7. DEMO CIKTISINI AYRI DOSYAYA KAYDET
    # ASLA resmi Tasima-plani.xlsx uzerine yazilmaz.
    # =========================================

    demo_plan.to_excel(DEMO_OUTPUT_FILE, index=False)
    print(f"\nDemo plani kaydedildi: {DEMO_OUTPUT_FILE}")
    print("(Bu dosya resmi Tasima-plani.xlsx DEGILDIR - ayri bir demo ciktisidir)")

    print("=" * 60)

    return {
        "baseline_cost": round(baseline_cost, 2),
        "demo_cost": round(demo_cost, 2),
        "cost_diff": round(cost_diff, 2),
        "cost_pct": round(cost_pct, 2),
        "baseline_sla": round(baseline_sla, 2),
        "demo_sla": round(demo_sla, 2),
        "demo_runtime_seconds": round(demo_time, 2),
    }


if __name__ == "__main__":
    run_digital_twin_demo()