from pathlib import Path
import unicodedata
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw"


def _normalize_filename(name):
    normalized = unicodedata.normalize("NFKD", name.lower())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    normalized = normalized.strip()
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("-", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def resolve_data_file(file_name):
    """
    SECURITY FIX: substring matching çıxarıldı.
    Əvvəlki versiya "requested_name in candidate_name" yoxlaması
    ilə səhv (oxşar adlı, köhnə mərhələdən qalan) faylı seçə
    bilirdi — məsələn "Arac_Kapasite_Maliyet.xlsx" (MVP, günlük
    maliyət) "Arac__Kapasite_Maliyet_Saat.xlsx" (Gelişmiş, saatlik
    maliyət) əvəzinə sakitcə seçilə bilərdi. İndi yalnız TAM
    normallaşdırılmış ad bərabərliyi qəbul edilir — bu, boşluq/
    tire/böyük-kiçik hərf fərqlərini hələ də idarə edir, amma
    səhv fayl seçimini qeyri-mümkün edir.
    """
    file_path = RAW_DATA_PATH / file_name
    if file_path.exists():
        return file_path

    requested_name = _normalize_filename(file_name)
    matches = []

    for candidate in RAW_DATA_PATH.iterdir():
        if not candidate.is_file():
            continue

        candidate_name = _normalize_filename(candidate.name)
        if candidate_name == requested_name:
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise FileNotFoundError(
            f"Ambiguous file match for '{file_name}': "
            f"{[m.name for m in matches]} — please remove duplicates."
        )

    raise FileNotFoundError(f"File not found: {file_path}")


def load_excel_file(file_name):
    """
    Generic Excel loader
    """

    file_path = resolve_data_file(file_name)

    df = pd.read_excel(file_path)

    print(f"Loaded: {file_name}")
    print(f"Shape: {df.shape}")

    return df


def load_all_data():
    """
    Load all project datasets

    Gelişmiş Çözüm Aşaması — Yeni Fayllar:
    - teknofest26_gelismis.xlsx  → əsas talep datası (saat bazlı)
    - Arac_Kapasite_Maliyet_Saat.xlsx → saatlik maliyet
    - Ellecleme-kapasite.xlsx  → elleçleme kapasitesi
    - sehirler_arasi_lojistik.xlsx → mesafe + SLA matrissi
    - Kiralik_Araclar.xlsx     → kiralık araç listesi
    """

    # =========================================
    # ƏSAS TALEP DATASI — YENİ
    # Saat bazlı: 09:00 və 17:00
    # =========================================

    demand_df = load_excel_file(
        "teknofest26_gelismis.xlsx"
    )

    # =========================================
    # ARAÇ KAPASİTE VƏ MALİYƏT — SAATLİK
    # Əvvəlki günlük maliyet → saatlik maliyet
    # =========================================

    vehicle_df = load_excel_file(
        "Arac__Kapasite_Maliyet_Saat.xlsx"
    )

    # =========================================
    # ELLEÇLƏMƏ KAPASİTESİ — YENİ
    # Hər transfer mərkəzi üçün günlük limit
    # =========================================

    handling_df = load_excel_file(
        "Ellecleme-kapasite.xlsx"
    )

    # =========================================
    # ŞƏHİRLƏRARASI LOJİSTİK MATRİSİ — YENİ
    # Mesafe + araç tipinə görə seyahat saatı + SLA
    # =========================================

    distance_df = load_excel_file(
        "sehirler_arasi_lojistik.xlsx"
    )

    # =========================================
    # KİRALIQ ARAÇLAR
    # Əvvəlki ilə eyni format
    # =========================================

    rental_df = load_excel_file(
        "Kiralik_Araclar.xlsx"
    )

    # =========================================
    # TIR KAPASİTESİ — YENİ (v2 güncellenmiş)
    # Hər TM üçün günlük max tır sayı
    # =========================================

    tir_kapasite_df = load_excel_file(
        "tir_kapasiteleri_v2.xlsx"
    )

    return {
        "desi_talep": demand_df,
        "arac_kapasite": vehicle_df,
        "ellecleme_kapasite": handling_df,
        "sehirler_arasi": distance_df,
        "kiralik_araclar": rental_df,
        "tir_kapasite": tir_kapasite_df,
    }


if __name__ == "__main__":

    data = load_all_data()

    print("\nDATA SUCCESSFULLY LOADED")