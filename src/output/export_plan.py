from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_PATH = BASE_DIR / "outputs"

OUTPUT_PATH.mkdir(exist_ok=True)


def export_plan(df):
    """
    Export optimization plan

    Gelişmiş Çözüm Aşaması — Yeni Format:
    - Fayl adı: Tasima-plani.xlsx (yarışma tələbi)
    - Sütunlar: Araç ID, Araç Tipi, Araç türü,
                Çıkış/Varış Transfer Merkezi,
                Çıkış/Varış Tarihi + Saati,
                Talep ID, Taşınan Desi,
                Yolculuk süresi,
                Varış/Çıkış Elleçleme süresi,
                SLA cezası, Toplam Maliyet
    """

    # =========================================
    # FAYL ADI — yarışma tələbinə uyğun
    # =========================================

    output_file = OUTPUT_PATH / "Tasima-plani.xlsx"

    df = df.copy()

    # =========================================
    # TARİH FORMATI
    # =========================================

    for date_col in ["Çıkış Tarihi", "Varış Tarihi", "Tarih"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(
                df[date_col], errors="coerce"
            ).dt.strftime("%Y-%m-%d")

    # =========================================
    # SÜTUN SIRASI — yarışma nümunəsinə uyğun
    # TAŞIMA PLANI.xlsx formatı
    # =========================================

    # SPESİFİKASİYAYA TAM UYĞUN — yalnız PDF-in nümunə
    # TAŞIMA PLANI.xlsx faylında göstərilən 16 sütun.
    # "Doluluk Oranı" və "Mesafe KM" bizim daxili analiz
    # üçün əlavə etdiyimiz sütunlardır, nümunə formatda
    # yoxdur — diskvalifikasiya riski daşımamaq üçün
    # əsas fayldan çıxarılıb (kod GitHub-da hələ də var).
    preferred_columns = [
        "Araç ID",
        "Araç Tipi",
        "Araç türü",
        "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi",
        "Çıkış Tarihi",
        "Çıkış Saati",
        "Varış Tarihi",
        "Varış Saati",
        "Talep ID",
        "Taşınan Desi",
        "Yolculuk süresi",
        "Varış elleçleme süresi",
        "Çıkış Elleçleme süresi",
        "SLA cezası",
        "Toplam Maliyet",
    ]

    # Yalnız mövcud sütunları saxla, sıraya görə —
    # extra_columns qəsdən ƏLAVƏ EDİLMİR (format təhlükəsizliyi)
    final_columns = [
        c for c in preferred_columns if c in df.columns
    ]

    df = df[final_columns]

    df.to_excel(output_file, index=False)

    print(f"\nPLAN EXPORTED: {output_file}")

    # =========================================
    # EXPORT SUMMARY
    # =========================================

    print("\nTASIMA PLANI EXPORT SUMMARY")
    print(f"Total Rows     : {len(df)}")

    if "Toplam Maliyet" in df.columns:
        print(
            f"Vehicle Cost   : "
            f"{round(df['Toplam Maliyet'].sum(), 2)} TL"
        )

    if "SLA cezası" in df.columns:
        total_sla = df["SLA cezası"].sum()
        sla_count = len(df[df["SLA cezası"] > 0])
        print(f"SLA Penalty    : {round(total_sla, 2)} TL")
        print(f"SLA Violations : {sla_count} shipments")

    if "Doluluk Oranı" in df.columns:
        print(
            f"Avg Utilization: "
            f"{round(df['Doluluk Oranı'].mean(), 2)}"
        )

    if "Araç ID" in df.columns:
        print(
            f"Araç ID Range  : "
            f"{df['Araç ID'].iloc[0]} → {df['Araç ID'].iloc[-1]}"
        )

    if "Araç Tipi" in df.columns:
        kiralik = len(df[df["Araç Tipi"] == "Kiralık"])
        spot = len(df[df["Araç Tipi"] == "Spot"])
        print(f"Kiralık Araç   : {kiralik} rows")
        print(f"Spot Araç      : {spot} rows")