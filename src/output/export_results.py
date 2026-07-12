from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_PATH = BASE_DIR / "outputs"

OUTPUT_PATH.mkdir(exist_ok=True)


def export_forecasts(df):
    """
    Export forecast results

    Gelişmiş Çözüm Aşaması — Yeni Format:
    - Fayl adı: Talep-tahmini.xlsx (yarışma tələbi)
    - Sütunlar: Talep ID, Tarih, Talep Tamamlama Saati,
                Çıkış, Varış, Tahmin Edilen Desi
    - Köhnə format da dəstəklənir (Tahminlenen Desi)
    """

    # =========================================
    # FAYL ADI — yarışma tələbinə uyğun
    # =========================================

    output_file = OUTPUT_PATH / "Talep-tahmini.xlsx"

    df = df.copy()

    # =========================================
    # TARİH FORMATI
    # =========================================

    if "Tarih" in df.columns:
        df["Tarih"] = pd.to_datetime(
            df["Tarih"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    # =========================================
    # SÜTUN SIRASI — yarışma nümunəsinə uyğun
    # Talep ID | Tarih | Talep Tamamlama Saati |
    # Çıkış Transfer Merkezi | Varış Transfer Merkezi |
    # Tahmin Edilen Desi
    # =========================================

    # SPESİFİKASİYAYA TAM UYĞUN — yalnız PDF-də açıq
    # göstərilən 6 sütun. Əlavə (icazəsiz) sütunlar
    # diskvalifikasiya riski daşıdığı üçün saxlanmır.
    preferred_columns = [
        "Talep ID",
        "Tarih",
        "Talep Tamamlama Saati",
        "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi",
        "Tahmin Edilen Desi",
    ]

    # Yalnız mövcud sütunları saxla, sıraya görə
    final_columns = [
        c for c in preferred_columns if c in df.columns
    ]

    # FORMAT TƏHLÜKƏSİZLİYİ: "Model Type" kimi daxili
    # debug/analiz sütunları ƏSAS fayla YAZILMIR — yarışma
    # spesifikasiyası dəqiq format tələb edir, əlavə sütunlar
    # (deklarə edilməyən) diskvalifikasiya riski yaradır.
    # Yalnız açıq şəkildə icazə verilən sütunlar ixrac olunur.
    df = df[final_columns]

    # Export Excel
    df.to_excel(output_file, index=False)

    print(f"\nForecast exported: {output_file}")

    # =========================================
    # EXPORT SUMMARY
    # =========================================

    print("\nFORECAST EXPORT SUMMARY")
    print(f"Total Rows: {len(df)}")

    # Yeni format
    desi_col = None
    if "Tahmin Edilen Desi" in df.columns:
        desi_col = "Tahmin Edilen Desi"
    elif "Tahminlenen Desi" in df.columns:
        desi_col = "Tahminlenen Desi"

    if desi_col:
        print(
            f"Total Forecasted Desi: "
            f"{round(df[desi_col].sum(), 2)}"
        )
        print(
            f"Average Forecasted Desi: "
            f"{round(df[desi_col].mean(), 2)}"
        )
        print(
            f"Max Forecasted Desi: "
            f"{round(df[desi_col].max(), 2)}"
        )
        print(
            f"Min Forecasted Desi: "
            f"{round(df[desi_col].min(), 2)}"
        )

    # Saat bazlı statistika
    if "Talep Tamamlama Saati" in df.columns:
        morning = df[df["Talep Tamamlama Saati"].astype(str).str.strip().isin(["9:00", "09:00"])]
        evening = df[~df["Talep Tamamlama Saati"].astype(str).str.strip().isin(["9:00", "09:00"])]
        print(f"09:00 Forecasts: {len(morning)} rows")
        print(f"17:00 Forecasts: {len(evening)} rows")

    if "Talep ID" in df.columns:
        print(
            f"Talep ID Range: "
            f"{df['Talep ID'].iloc[0]} → {df['Talep ID'].iloc[-1]}"
        )

    print("\nEXPORT COMPLETED SUCCESSFULLY.")