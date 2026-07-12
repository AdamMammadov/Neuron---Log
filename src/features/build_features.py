import numpy as np
import pandas as pd


def create_time_features(df):
    """
    Create time based features
    """

    df = df.copy()

    # Basic Time Features
    df["month"] = df["Tarih"].dt.month
    df["day"] = df["Tarih"].dt.day
    df["day_of_week"] = df["Tarih"].dt.dayofweek
    df["week_of_year"] = df["Tarih"].dt.isocalendar().week.astype(int)

    # Weekend
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical Encoding
    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def create_lag_features(df):
    """
    Create lag & rolling features
    """

    df = df.copy()

    df = df.sort_values(
        [
            "Çıkış Transfer Merkezi",
            "Varış Transfer Merkezi",
            "Tarih"
        ]
    )

    group_cols = [
        "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi"
    ]

    target = "Toplam Desi"

    # Lag Features
    df["lag_1"] = (
        df.groupby(group_cols)[target]
        .shift(1)
    )

    df["lag_2"] = (
        df.groupby(group_cols)[target]
        .shift(2)
    )

    df["lag_7"] = (
        df.groupby(group_cols)[target]
        .shift(7)
    )

    # Rolling Means
    df["rolling_mean_3"] = (
        df.groupby(group_cols)[target]
        .transform(
            lambda x: x.rolling(3).mean()
        )
    )

    df["rolling_mean_7"] = (
        df.groupby(group_cols)[target]
        .transform(
            lambda x: x.rolling(7).mean()
        )
    )

    # Rolling Std
    df["rolling_std_7"] = (
        df.groupby(group_cols)[target]
        .transform(
            lambda x: x.rolling(7).std()
        )
    )

    return df


# =====================================================
# ADVANCED FEATURE ENGINEERING
# =====================================================

def create_advanced_features(df):
    """
    Advanced AI feature engineering
    for lower MAE forecasting

    Gelişmiş Çözüm Aşaması — Yeni Özellikler:
    - is_morning: 09:00=1, 17:00=0
    - Saat bazlı lag features: hər saat ayrıca qruplanır
    - Route+Saat target encoding əlavə edildi
    """

    df = df.copy()

    df["Tarih"] = pd.to_datetime(
        df["Tarih"]
    )

    # =========================================
    # SAAT FEATURE — YENİ
    # 09:00 tələbi ilə 17:00 tələbi fərqlidir.
    # is_morning=1 → sabah, is_morning=0 → axşam
    # =========================================

    if "Saat" in df.columns:
        df["is_morning"] = (
            df["Saat"]
            .astype(str)
            .str.startswith("9")
            .astype(int)
        )
    else:
        df["is_morning"] = 1

    # Saat bazlı sort — saat feature-ı lag-lara düzgün tətbiq olunsun
    df = df.sort_values(
        by=[
            "Çıkış Transfer Merkezi",
            "Varış Transfer Merkezi",
            "Tarih",
            "is_morning"
        ]
    )

    # =========================================
    # ADVANCED TIME FEATURES
    # =========================================

    df["ay"] = (
        df["Tarih"].dt.month
    )

    df["gun_hafta"] = (
        df["Tarih"].dt.dayofweek
    )

    df["gun_ay"] = (
        df["Tarih"].dt.day
    )

    df["hafta_yil"] = (
        df["Tarih"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    df["haftasonu"] = (
        df["gun_hafta"]
        .isin([5, 6])
        .astype(int)
    )

    # =========================================
    # SEASONALITY FEATURES
    # =========================================

    df["ay_sin"] = np.sin(
        2 * np.pi * df["ay"] / 12
    )

    df["ay_cos"] = np.cos(
        2 * np.pi * df["ay"] / 12
    )

    # =========================================
    # HOLIDAY DISTANCE FEATURE
    # =========================================

    official_holidays = pd.to_datetime([

        # NEW YEAR
        "2026-01-01",

        # NATIONAL HOLIDAYS
        "2026-04-23",
        "2026-05-01",
        "2026-05-19",
        "2026-07-15",
        "2026-08-30",
        "2026-10-29",

        # RAMADAN
        "2026-03-20",
        "2026-03-21",
        "2026-03-22",

        # KURBAN
        "2026-05-27",
        "2026-05-28",
        "2026-05-29",
        "2026-05-30"
    ])

    def calc_days_to_holiday(current_date):

        return min(
            abs(
                (
                    holiday - current_date
                ).days
            )
            for holiday in official_holidays
        )

    df["days_to_holiday"] = (
        df["Tarih"]
        .apply(calc_days_to_holiday)
    )

    # Month edge features
    df["is_month_start"] = (
        df["Tarih"]
        .dt
        .is_month_start
    ).astype(int)

    df["is_month_end"] = (
        df["Tarih"]
        .dt
        .is_month_end
    ).astype(int)

    # Ramadan effect
    df["is_ramadan_week"] = (
        (df["Tarih"] >= pd.Timestamp("2026-03-16"))
        &
        (df["Tarih"] <= pd.Timestamp("2026-03-26"))
    ).astype(int)

    # =========================================
    # ROUTE ENCODING
    # =========================================

    df["Rota"] = (
        df["Çıkış Transfer Merkezi"]
        .astype(str)
        +
        "_"
        +
        df["Varış Transfer Merkezi"]
        .astype(str)
    )

    df["Rota_Code"] = (
        df["Rota"]
        .astype("category")
        .cat
        .codes
    )

    # =========================================
    # TARGET ENCODING FEATURES
    # =========================================

    df["Origin_Target_Enc"] = (
        df.groupby(
            "Çıkış Transfer Merkezi"
        )[
            "Toplam Desi"
        ].transform(
            "mean"
        )
    )

    df["Destination_Target_Enc"] = (
        df.groupby(
            "Varış Transfer Merkezi"
        )[
            "Toplam Desi"
        ].transform(
            "mean"
        )
    )

    df["Route_Target_Enc"] = (
        df.groupby(
            [
                "Çıkış Transfer Merkezi",
                "Varış Transfer Merkezi"
            ]
        )[
            "Toplam Desi"
        ].transform(
            "mean"
        )
    )

    # =========================================
    # GROUP DEFINITIONS
    # Saat bazlı qruplandırma — YENİ
    # Hər route + saat kombinasiyası ayrıca qruplanır
    # Bu 09:00 lag-larının 17:00-a qarışmasının
    # qarşısını alır
    # =========================================

    # Saat bazlı group (is_morning ilə)
    group_cols_saat = [
        "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi",
        "is_morning"
    ]

    group_saat = df.groupby(group_cols_saat)["Toplam Desi"]

    # Köhnə route group (geriyə uyğunluq üçün saxlanır)
    group = df.groupby(
        [
            "Çıkış Transfer Merkezi",
            "Varış Transfer Merkezi"
        ]
    )["Toplam Desi"]

    # =========================================
    # ADVANCED LAG FEATURES — SAAT BAZLI
    # lag_1 = əvvəlki gün eyni saat tələbi
    # =========================================

    for lag in [1, 2, 3, 7, 14]:

        df[f"lag_{lag}"] = (
            group_saat.shift(lag)
        )

    # =========================================
    # ADVANCED ROLLING FEATURES — SAAT BAZLI
    # =========================================

    for window in [3, 7, 14]:

        df[f"rolling_mean_{window}"] = (
            group_saat
            .shift(1)
            .groupby(
                [
                    df["Çıkış Transfer Merkezi"],
                    df["Varış Transfer Merkezi"],
                    df["is_morning"]
                ]
            )
            .rolling(window)
            .mean()
            .reset_index(level=[0, 1, 2], drop=True)
        )

        df[f"rolling_std_{window}"] = (
            group_saat
            .shift(1)
            .groupby(
                [
                    df["Çıkış Transfer Merkezi"],
                    df["Varış Transfer Merkezi"],
                    df["is_morning"]
                ]
            )
            .rolling(window)
            .std()
            .reset_index(level=[0, 1, 2], drop=True)
        )

        df[f"rolling_max_{window}"] = (
            group_saat
            .shift(1)
            .groupby(
                [
                    df["Çıkış Transfer Merkezi"],
                    df["Varış Transfer Merkezi"],
                    df["is_morning"]
                ]
            )
            .rolling(window)
            .max()
            .reset_index(level=[0, 1, 2], drop=True)
        )

    # =========================================
    # DEMAND MOMENTUM FEATURES
    # =========================================

    df["lag_diff_1_7"] = (
        df.get("lag_1")
        -
        df.get("lag_7")
    )

    df["lag_ratio_1_7"] = (
        df.get("lag_1")
        /
        (
            df.get("lag_7")
            + 1
        )
    )

    df["rolling_ratio"] = (
        df.get("rolling_mean_3")
        /
        (
            df.get("rolling_mean_14")
            + 1
        )
    )

    # =========================================
    # NAN CLEANING
    # =========================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.ffill()

    df = df.fillna(0)

    print(
        "Advanced feature engineering completed."
    )

    return df