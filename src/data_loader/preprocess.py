import pandas as pd


def preprocess_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Temiz ve düzenli bir talep veri seti (demand dataset) oluşturur.

    Args:
        df (pd.DataFrame): İşlenecek ham veri seti.

    Returns:
        pd.DataFrame: Ön işlemesi tamamlanmış temiz veri seti.
    """
    df = df.copy()

    # 1. Tarih Dönüşümü
    df["Tarih"] = pd.to_datetime(df["Tarih"], errors="coerce")

    # 2. Kritik Sütunlardaki Boş (Null) Değerlerin Temizlenmesi
    essential_columns = [
        "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi",
        "Toplam Desi",
        "Tarih",
    ]
    df = df.dropna(subset=essential_columns)

    # 3. Negatif Talep / Desi Değerlerinin Filtrelenmesi
    df = df[df["Toplam Desi"] >= 0]

    # 4. Mükerrer (Çift) Kayıtların Tekilleştirilmesi (Son kayıt tutulur)
    dup_columns = ["Çıkış Transfer Merkezi", "Varış Transfer Merkezi", "Tarih"]
    df = df.drop_duplicates(subset=dup_columns, keep="last")

    # 5. Verinin Sıralanması ve İndeksin Yeniden Kurulması
    df = df.sort_values(by=dup_columns).reset_index(drop=True)

    return df
