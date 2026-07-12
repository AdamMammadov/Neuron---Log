import pandas as pd


def preprocess_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temiz ve düzenli bir talep veri seti oluşturur.

    Gelişmiş Çözüm Aşaması — Yeni Özellikler:
    - Saat sütunu (talep_tamamlanma_saati) işlenir
    - Sütun adları yeni data formatına uyarlandı
    - is_morning feature əlavə edildi (09:00=1, 17:00=0)
    - Talep ID sütunu saxlanılır

    Args:
        df (pd.DataFrame): İşlenecek ham veri seti.

    Returns:
        pd.DataFrame: Ön işlemesi tamamlanmış temiz veri seti.
    """

    df = df.copy()

    # =========================================
    # 1. SÜTUN ADLARINI NORMALLAŞDIR
    # Yeni data: tarih, cikis, varis, talep_id,
    # toplam_desi, talep_tamamlanma_saati
    # Köhnə sistem ilə uyğunluq üçün adlar
    # standart formata çevrilir
    # =========================================

    column_map = {
        "tarih": "Tarih",
        "cikis": "Çıkış Transfer Merkezi",
        "varis": "Varış Transfer Merkezi",
        "talep_id": "Talep ID",
        "toplam_desi": "Toplam Desi",
        "talep_tamamlanma_saati": "Saat",

        # Köhnə format da dəstəklənir
        "Tarih": "Tarih",
        "Çıkış Transfer Merkezi": "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi": "Varış Transfer Merkezi",
        "Toplam Desi": "Toplam Desi",
    }

    df = df.rename(columns={
        k: v for k, v in column_map.items()
        if k in df.columns
    })

    # =========================================
    # 2. TARİH DÖNÜŞÜMİ
    # =========================================

    df["Tarih"] = pd.to_datetime(
        df["Tarih"], errors="coerce"
    )

    # =========================================
    # 3. SAAT SÜTUNU — YENİ
    # talep_tamamlanma_saati: "9:00" → Saat
    # is_morning: 09:00=1, 17:00=0
    # =========================================

    if "Saat" in df.columns:

        df["Saat"] = df["Saat"].astype(str).str.strip()

        df["is_morning"] = (
            df["Saat"]
            .str.startswith("9")
            .astype(int)
        )

    else:

        # Əgər saat sütunu yoxdursa default 09:00
        df["Saat"] = "9:00"
        df["is_morning"] = 1

    # =========================================
    # 4. KRİTİK SÜTUNLARDAKI BOŞ DEĞERLERİ SİL
    # =========================================

    essential_columns = [
        "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi",
        "Toplam Desi",
        "Tarih",
    ]

    df = df.dropna(subset=essential_columns)

    # =========================================
    # 5. NEGATİF TALEP DEĞERLERİNİ FİLTRELE
    # =========================================

    df = df[df["Toplam Desi"] >= 0]

    # =========================================
    # 6. MÜKERRER KAYITLARI TEKİLLEŞTİR
    # Saat sütunu da duplicate key-ə daxildir —
    # eyni gün + eyni route + eyni saat → tekil
    # =========================================

    dup_columns = [
        "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi",
        "Tarih",
        "Saat",
    ]

    # Yalnız mövcud sütunlarla dedup
    dup_columns = [c for c in dup_columns if c in df.columns]

    df = df.drop_duplicates(
        subset=dup_columns,
        keep="last"
    )

    # =========================================
    # 7. SIRALAMA VƏ İNDEKS YENİLƏMƏ
    # =========================================

    sort_columns = [
        "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi",
        "Tarih",
        "is_morning",
    ]

    sort_columns = [c for c in sort_columns if c in df.columns]

    df = df.sort_values(
        by=sort_columns
    ).reset_index(drop=True)

    print(
        f"Preprocessed demand data: "
        f"{len(df)} rows"
    )

    if "Saat" in df.columns:

        morning_count = len(df[df["is_morning"] == 1])
        evening_count = len(df[df["is_morning"] == 0])

        print(
            f"  09:00 records: {morning_count}"
        )

        print(
            f"  17:00 records: {evening_count}"
        )

    return df