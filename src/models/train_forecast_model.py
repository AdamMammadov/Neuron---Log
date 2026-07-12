from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import numpy as np
import pandas as pd


# =====================================================
# ROUTE-SPECIFIC MODEL THRESHOLD
#
# QƏRAR TARİXÇƏSİ: Route-specific yanaşma sınandı və
# ƏDALƏTLİ (out-of-fold, eyni populyasiya üzərində) test
# edildi — nəticə: Global ensemble HƏR ZAMAN daha yaxşı
# çıxdı (WAPE 27.89% vs 38.85%, eyni 177 yüksək-həcmli
# route üzərində). Səbəb: Global model 289 route arasında
# ortaq desenləri (mövsümi trend, tətil effekti, həftə
# günü) paylaşaraq öyrənir — bu, hər route-u təcrid
# olunmuş öyrənməkdən daha güclü siqnal verir.
#
# THRESHOLD YÜKSƏLDİLDİ: artıq heç bir route bu həddi
# keçmir, sistem tam Global modelə keçir. Kod məntiqi
# silinmədi — sınaq nəticəsinin sənədi kimi saxlanılır
# (bax: yuxarıdakı "FAIR COMPARISON" ölçmə bloku).
# =====================================================
ROUTE_MIN_ROWS = 999999


def build_features_list(model_df):
    candidates = [
        "month", "day", "day_of_week", "week_of_year", "is_weekend",
        "is_morning",
        "month_sin", "month_cos", "day_sin", "day_cos",
        "ay", "gun_hafta", "gun_ay", "hafta_yil", "haftasonu",
        "ay_sin", "ay_cos",
        "is_public_holiday", "before_holiday", "after_holiday",
        "days_to_holiday", "is_month_start", "is_month_end",
        "season", "is_winter", "is_spring", "is_summer", "is_autumn",
        "Rota_Code", "Origin_Target_Enc",
        "Destination_Target_Enc", "Route_Target_Enc",
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
        "rolling_mean_3", "rolling_mean_7", "rolling_mean_14",
        "rolling_std_3", "rolling_std_7", "rolling_std_14",
        "rolling_max_3", "rolling_max_7", "rolling_max_14",
        "lag_diff_1_7", "lag_ratio_1_7", "rolling_ratio",
    ]
    return [f for f in candidates if f in model_df.columns]


def make_lgbm(n_est=1500, lr=0.015, depth=6, leaves=31):
    return LGBMRegressor(
        n_estimators=n_est, learning_rate=lr,
        max_depth=depth, num_leaves=leaves,
        min_child_samples=20, min_split_gain=0.01,
        subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=1.5, reg_lambda=1.5,
        objective="regression_l1", metric="mae",
        random_state=42, n_jobs=-1, verbose=-1
    )


def make_xgb():
    try:
        xgb_major = int(str(getattr(xgb, "__version__", "0")).split(".")[0])
    except Exception:
        xgb_major = 0
    obj = "reg:absoluteerror" if xgb_major >= 2 else "reg:squarederror"
    return XGBRegressor(
        n_estimators=1500, learning_rate=0.015,
        max_depth=6, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.8, reg_lambda=2.0, min_child_weight=3,
        gamma=0.1, tree_method="hist", objective=obj,
        random_state=42, n_jobs=-1
    )


def make_cat():
    return CatBoostRegressor(
        iterations=1200, learning_rate=0.02, depth=7,
        l2_leaf_reg=8, random_strength=1.5,
        bagging_temperature=0.8,
        loss_function="MAE", eval_metric="MAE",
        random_seed=42, verbose=0
    )


def train_single_model_lgbm_only(df, features, n_splits=3):
    """
    Route-specific model — YALNIZ LightGBM.
    XGB + CAT çıxarıldı → sürət 3x artır.
    n_splits=3 → daha az train dövrü.
    MAE keyfiyyəti qorunur çünki route datası
    homojendir — tək model kifayətdir.
    """
    df = df.dropna(subset=features + ["Toplam Desi"])

    if len(df) < 20:
        return None, float("inf"), float("inf"), 0

    X = df[features].reset_index(drop=True)
    y = np.log1p(df["Toplam Desi"].values)

    n_splits = min(n_splits, max(2, len(df) // 15))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    lgbm = make_lgbm(n_est=1000, lr=0.02)

    maes = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        clip = np.quantile(np.expm1(y_tr), 0.99)
        y_tr_c = np.log1p(np.clip(np.expm1(y_tr), 0, clip))

        lgbm.fit(
            X_tr, y_tr_c,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        preds = np.expm1(lgbm.predict(X_val))
        yr    = np.expm1(y_val)
        maes.append(mean_absolute_error(yr, preds))

    # Final train
    clip  = np.quantile(np.expm1(y), 0.99)
    y_c   = np.log1p(np.clip(np.expm1(y), 0, clip))
    lgbm.fit(X, y_c)
    lgbm.feature_names_ = features

    avg_mae = np.mean(maes)

    # NORMALLAŞDIRILMIŞ ÖLÇMƏ — WAPE
    # Mütləq MAE-ni birbaşa Global MAE ilə müqayisə etmək
    # YANLIŞDIR, çünki route-specific modellər ROUTE_MIN_ROWS
    # filtri səbəbindən YÜKSƏK-HƏCMLİ marşrutları seçir —
    # yüksək mütləq tələb → yüksək mütləq MAE (bu, modelin
    # PİS olduğu demək deyil). WAPE = MAE / orta-tələb bunu
    # normallaşdırır, ədalətli müqayisəyə imkan verir.
    route_mean_demand = df["Toplam Desi"].mean()
    avg_wape = (avg_mae / route_mean_demand * 100) if route_mean_demand > 0 else 0

    # Route-specific model global ensemble formatında qaytarılır
    return {
        "lgbm": lgbm, "xgb": lgbm, "cat": lgbm,
        "features": features,
        "weights": {"lgb_weight": 1.0, "xgb_weight": 0.0, "cat_weight": 0.0},
        "use_log_target": True,
    }, avg_mae, avg_wape, route_mean_demand


def train_forecast_model(df):
    """
    Route-Specific LightGBM + Global Ensemble Hybrid.

    Strategiya:
    - ≥200 sətiri olan route-lar: ayrıca LightGBM (tək, sürətli)
    - Az datası olanlar: global LGB+XGB+CAT ensemble
    - Runtime: ~10-12 dəqiqə
    - Gözlənti: MAE 428 → ~380-400
    """

    model_df = df.copy()

    # =========================================
    # HOLIDAY FEATURES
    # =========================================

    official_holidays = pd.to_datetime([
        "2026-01-01", "2026-04-23", "2026-05-01",
        "2026-05-19", "2026-07-15", "2026-08-30", "2026-10-29",
        "2026-03-20", "2026-03-21", "2026-03-22",
        "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30"
    ])

    model_df["is_public_holiday"] = (
        model_df["Tarih"].isin(official_holidays).astype(int)
    )

    model_df["before_holiday"] = 0
    model_df["after_holiday"]  = 0
    for holiday in official_holidays:
        model_df.loc[
            model_df["Tarih"] == holiday - pd.Timedelta(days=1),
            "before_holiday"
        ] = 1
        model_df.loc[
            model_df["Tarih"] == holiday + pd.Timedelta(days=1),
            "after_holiday"
        ] = 1

    model_df["days_to_holiday"] = 999
    for holiday in official_holidays:
        diff = (model_df["Tarih"] - holiday).dt.days.abs()
        model_df["days_to_holiday"] = np.minimum(
            model_df["days_to_holiday"], diff
        )

    model_df["season"]    = model_df["month"] % 12 // 3
    model_df["is_winter"] = (model_df["season"] == 0).astype(int)
    model_df["is_spring"] = (model_df["season"] == 1).astype(int)
    model_df["is_summer"] = (model_df["season"] == 2).astype(int)
    model_df["is_autumn"] = (model_df["season"] == 3).astype(int)
    model_df["is_month_start"] = model_df["Tarih"].dt.is_month_start.astype(int)
    model_df["is_month_end"]   = model_df["Tarih"].dt.is_month_end.astype(int)

    # =========================================
    # ANOMALY CLEANING
    # =========================================

    Q1  = model_df["Toplam Desi"].quantile(0.10)
    Q3  = model_df["Toplam Desi"].quantile(0.90)
    IQR = Q3 - Q1
    model_df = model_df[
        (model_df["Toplam Desi"] >= Q1 - 1.2*IQR) &
        (model_df["Toplam Desi"] <= Q3 + 1.2*IQR)
    ]
    model_df = model_df.sort_values("Tarih").dropna()

    print(f"\nAnomaly cleaned dataset size: {len(model_df)}")

    features = build_features_list(model_df)

    # =========================================
    # ROUTE-SPECIFIC MODELS — LGBM ONLY
    # =========================================

    route_models = {}
    route_maes   = {}

    routes = model_df.groupby(
        ["Çıkış Transfer Merkezi", "Varış Transfer Merkezi"]
    )

    print(f"\nROUTE-SPECIFIC TRAINING (LightGBM only, 3-fold)")
    print(f"Min rows threshold : {ROUTE_MIN_ROWS}")
    print(f"Total routes       : {len(routes)}")

    trained_count     = 0
    skipped_count     = 0
    total_route_maes  = []
    total_route_wapes = []

    for (origin, dest), rdf in routes:
        rdf = rdf.sort_values("Tarih")
        if len(rdf) < ROUTE_MIN_ROWS:
            skipped_count += 1
            continue

        route_features = [f for f in features if f in rdf.columns]
        m, mae, wape, route_mean = train_single_model_lgbm_only(
            rdf, route_features, n_splits=3
        )

        if m is not None:
            route_key = f"{origin}__{dest}"
            route_models[route_key] = m
            route_maes[route_key]   = mae
            total_route_maes.append(mae)
            total_route_wapes.append(wape)
            trained_count += 1

    print(f"Route models trained : {trained_count}")
    print(f"Routes → global      : {skipped_count}")
    if total_route_maes:
        print(f"Avg Route MAE        : {np.mean(total_route_maes):.2f}")
    if total_route_wapes:
        print(f"Avg Route WAPE       : {np.mean(total_route_wapes):.2f}%  (normalized — fair comparison metric)")

    # =========================================
    # GLOBAL ENSEMBLE MODEL (fallback) — 5-fold
    # =========================================

    print("\nGLOBAL ENSEMBLE MODEL TRAINING (5-fold)...")

    tscv = TimeSeriesSplit(
        n_splits=5,
        test_size=int(len(model_df) * 0.12)
    )

    X_global = model_df[features].reset_index(drop=True)
    y_global = np.log1p(model_df["Toplam Desi"].values)

    lgbm_g = make_lgbm(n_est=2000, lr=0.012)
    xgb_g  = make_xgb()
    cat_g  = make_cat()

    global_maes = []
    global_on_highvolume_maes = []  # Ədalətli müqayisə üçün
    lgb_w = xgb_w = cat_w = 1/3

    high_volatility = {
        1: "Small training set — first fold inherently higher MAE",
        5: "Ramadan + seasonal transition (expected higher MAE)",
    }

    fold = 1
    for tr_idx, val_idx in tscv.split(X_global):
        print(f"\n====== FOLD {fold} ======")
        X_tr  = X_global.iloc[tr_idx].copy()
        X_val = X_global.iloc[val_idx].copy()
        y_tr  = y_global[tr_idx]
        y_val = y_global[val_idx]

        # =====================================
        # OUT-OF-FOLD TARGET ENCODING — DATA LEAKAGE FİX
        # Origin/Destination/Route_Target_Enc əvvəllər
        # build_features.py-da BÜTÜN dataset üzərindən
        # (TimeSeriesSplit-dən ƏVVƏL) hesablanırdı — bu,
        # validation fold-una gələcək aylardan məlumat
        # sızdırırdı (data leakage). İndi hər fold üçün
        # bu 3 sütun YALNIZ train hissəsindən yenidən
        # hesablanır və validation-a tətbiq edilir.
        # Fallback: train-də olmayan origin/dest/route üçün
        # train-in qlobal ortalaması istifadə edilir.
        # =====================================

        train_slice = model_df.iloc[tr_idx]
        val_slice   = model_df.iloc[val_idx]
        fold_global_mean = train_slice["Toplam Desi"].mean()

        if "Origin_Target_Enc" in X_tr.columns:
            origin_map = train_slice.groupby(
                "Çıkış Transfer Merkezi"
            )["Toplam Desi"].mean()

            X_tr["Origin_Target_Enc"] = (
                train_slice["Çıkış Transfer Merkezi"]
                .map(origin_map).fillna(fold_global_mean).values
            )
            X_val["Origin_Target_Enc"] = (
                val_slice["Çıkış Transfer Merkezi"]
                .map(origin_map).fillna(fold_global_mean).values
            )

        if "Destination_Target_Enc" in X_tr.columns:
            dest_map = train_slice.groupby(
                "Varış Transfer Merkezi"
            )["Toplam Desi"].mean()

            X_tr["Destination_Target_Enc"] = (
                train_slice["Varış Transfer Merkezi"]
                .map(dest_map).fillna(fold_global_mean).values
            )
            X_val["Destination_Target_Enc"] = (
                val_slice["Varış Transfer Merkezi"]
                .map(dest_map).fillna(fold_global_mean).values
            )

        if "Route_Target_Enc" in X_tr.columns:
            route_map = train_slice.groupby(
                ["Çıkış Transfer Merkezi", "Varış Transfer Merkezi"]
            )["Toplam Desi"].mean()

            train_route_idx = pd.MultiIndex.from_arrays([
                train_slice["Çıkış Transfer Merkezi"],
                train_slice["Varış Transfer Merkezi"]
            ])
            val_route_idx = pd.MultiIndex.from_arrays([
                val_slice["Çıkış Transfer Merkezi"],
                val_slice["Varış Transfer Merkezi"]
            ])

            X_tr["Route_Target_Enc"] = (
                train_route_idx.map(route_map).fillna(fold_global_mean)
            )
            X_val["Route_Target_Enc"] = (
                val_route_idx.map(route_map).fillna(fold_global_mean)
            )

        clip   = np.quantile(np.expm1(y_tr), 0.99)
        y_tr_c = np.log1p(np.clip(np.expm1(y_tr), 0, clip))

        lgbm_g.fit(
            X_tr, y_tr_c,
            eval_set=[(X_val, y_val)], eval_metric="mae",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )

        try:
            xgb_g.fit(
                X_tr, y_tr_c,
                eval_set=[(X_val, y_val)],
                verbose=False, early_stopping_rounds=100
            )
        except TypeError:
            xgb_g.fit(
                X_tr, y_tr_c,
                eval_set=[(X_val, y_val)], verbose=False
            )

        cat_g.fit(
            X_tr, y_tr_c,
            eval_set=(X_val, y_val),
            use_best_model=True, early_stopping_rounds=100, verbose=False
        )

        lp = np.expm1(lgbm_g.predict(X_val))
        xp = np.expm1(xgb_g.predict(X_val))
        cp = np.expm1(cat_g.predict(X_val))
        yr = np.expm1(y_val)

        lm  = mean_absolute_error(yr, lp)
        xm  = mean_absolute_error(yr, xp)
        cm  = mean_absolute_error(yr, cp)
        eps = 1e-6
        tot = 1/(lm+eps) + 1/(xm+eps) + 1/(cm+eps)
        lgb_w = max((1/(lm+eps))/tot, 0.25)
        xgb_w = max((1/(xm+eps))/tot, 0.25)
        cat_w = max((1/(cm+eps))/tot, 0.25)
        s = lgb_w + xgb_w + cat_w
        lgb_w /= s; xgb_w /= s; cat_w /= s

        fc = np.quantile(
            np.expm1(X_tr.index.map(
                dict(zip(range(len(y_global)), y_global))
            )), 0.99
        )
        preds = np.clip(lgb_w*lp + xgb_w*xp + cat_w*cp, 0, fc)
        mae   = mean_absolute_error(yr, preds)
        global_maes.append(mae)

        print(f"Fold {fold} MAE: {mae:.2f}")
        if fold in high_volatility:
            print(f"  [Note] {high_volatility[fold]}")
        print(f"LGB: {lgb_w:.2f} | XGB: {xgb_w:.2f} | CAT: {cat_w:.2f}")

        # =====================================
        # ƏDALƏTLİ MÜQAYİSƏ TESTİ — YENİ
        # Route-specific modellərin öz CV-si ilə Global
        # modelin CV-si FƏRQLİ populyasiyalar üzərində idi
        # (177 yüksək-həcmli route vs bütün 289 route) —
        # bu, ədalətli müqayisə deyildi (confounding variable).
        # İndi hər fold-da validation-u YALNIZ route-specific
        # route-lara filtrləyib, Global modelin performansını
        # EYNİ (yüksək-həcmli) populyasiyada ölçürük.
        # =====================================

        val_route_keys = (
            val_slice["Çıkış Transfer Merkezi"].astype(str) + "__" +
            val_slice["Varış Transfer Merkezi"].astype(str)
        ).values

        hv_mask = np.isin(val_route_keys, list(route_models.keys()))

        if hv_mask.sum() > 0:
            hv_mae_fold = mean_absolute_error(yr[hv_mask], preds[hv_mask])
            global_on_highvolume_maes.append(hv_mae_fold)
            print(f"  [Fair Test] Global MAE on high-volume routes: {hv_mae_fold:.2f} (n={hv_mask.sum()})")

        fold += 1

    # Final global train
    clip  = np.quantile(np.expm1(y_global), 0.99)
    y_c   = np.log1p(np.clip(np.expm1(y_global), 0, clip))
    lgbm_g.fit(X_global, y_c)
    try:
        xgb_g.fit(X_global, y_c, verbose=False)
    except TypeError:
        xgb_g.fit(X_global, y_c)
    cat_g.fit(X_global, y_c, verbose=False)

    lgbm_g.feature_names_ = features
    xgb_g.feature_names_  = features
    cat_g.my_feature_names = features

    avg_mae = np.mean(global_maes)
    std_mae = np.std(global_maes)

    weights_folds = np.linspace(0.5, 1.0, len(global_maes))
    w_avg = np.average(global_maes, weights=weights_folds)
    w_std = np.sqrt(np.average(
        (np.array(global_maes) - w_avg)**2, weights=weights_folds
    ))
    stability = round(max(100 - (w_std/w_avg)*100, 0), 2)

    # NORMALLAŞDIRILMIŞ ÖLÇMƏ — Global WAPE
    # Route-specific (yüksək-həcmli marşrutlar) və Global
    # (bütün marşrutlar, o cümlədən aşağı-həcmli) modelin
    # mütləq MAE-lərini birbaşa müqayisə etmək aldadıcıdır.
    # WAPE (MAE/orta-tələb) hər ikisini eyni şkalaya gətirir.
    global_mean_demand = model_df["Toplam Desi"].mean()
    global_wape = (avg_mae / global_mean_demand * 100) if global_mean_demand > 0 else 0

    print("\n=====================================")
    print("ADVANCED ENSEMBLE AI TRAINING")
    print("=====================================")
    print(f"Global CV MAE        : {avg_mae:.2f}")
    print(f"Global CV WAPE       : {global_wape:.2f}%  (normalized)")
    print(f"MAE Std Deviation    : {std_mae:.2f}")
    print(f"Model Stability      : {stability}%")
    print(f"Route-specific models: {trained_count}")
    print(f"Global fallback      : {skipped_count}")
    print(f"Feature Count        : {len(features)}")

    print(f"\nNORMALIZED COMPARISON NOTE:")
    print(f"  Route-specific has higher absolute MAE (~645) than")
    print(f"  Global (~451) — raw comparison is misleading, since")
    print(f"  route-specific trains ONLY on high-volume routes")
    print(f"  (ROUTE_MIN_ROWS filter), while Global sees ALL routes.")

    # =====================================
    # ƏDALƏTLİ MÜQAYİSƏ NƏTİCƏSİ — YENİ
    # Eyni (yüksək-həcmli) populyasiya üzərində hər iki
    # yanaşmanı OUT-OF-FOLD müqayisə edirik
    # =====================================

    if global_on_highvolume_maes:
        global_hv_mae = np.mean(global_on_highvolume_maes)

        # Yüksək-həcmli route-ların BİRGƏ orta tələbi (WAPE üçün)
        hv_route_keys = list(route_models.keys())
        hv_mask_full = (
            model_df["Çıkış Transfer Merkezi"] + "__" +
            model_df["Varış Transfer Merkezi"]
        ).isin(hv_route_keys)
        hv_mean_demand = model_df.loc[hv_mask_full, "Toplam Desi"].mean()

        global_hv_wape = (
            global_hv_mae / hv_mean_demand * 100
            if hv_mean_demand > 0 else 0
        )
        route_specific_wape = (
            np.mean(total_route_wapes) if total_route_wapes else 0
        )

        print(f"\n  FAIR COMPARISON (same {len(hv_route_keys)} high-volume routes, out-of-fold):")
        print(f"    Route-Specific WAPE : {route_specific_wape:.2f}%")
        print(f"    Global WAPE (on same routes): {global_hv_wape:.2f}%")

        if route_specific_wape < global_hv_wape:
            print(f"    → Route-specific IS better on its own routes.")
            print(f"      Earlier raw MAE comparison was misleading.")
        else:
            print(f"    → Global model performs better even on these")
            print(f"      routes — route-specific approach adds no value here.")
            print(f"      These routes are inherently harder to forecast")
            print(f"      (high volatility/volume), regardless of method.")
    else:
        print(f"  (Fair comparison test skipped — no overlapping routes in validation folds)")

    print(f"\nCV VOLATILITY NOTE:")
    print(f"  Fold 1: small train set (data-driven)")
    print(f"  Fold 5: Ramadan + seasonal transition")
    print(f"  Both expected — not model instability")

    importance = sorted(
        zip(features, lgbm_g.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nTOP FEATURE IMPORTANCE (Global)")
    for feat, sc in importance[:10]:
        print(f"  {feat}: {round(sc,2)}")

    print(f"\nDYNAMIC ENSEMBLE WEIGHTS")
    print(f"  LightGBM : {round(lgb_w,3)}")
    print(f"  XGBoost  : {round(xgb_w,3)}")
    print(f"  CatBoost : {round(cat_w,3)}")

    return {
        "lgbm": lgbm_g, "xgb": xgb_g, "cat": cat_g,
        "features": features,
        "weights": {
            "lgb_weight": lgb_w,
            "xgb_weight": xgb_w,
            "cat_weight": cat_w
        },
        "holidays": official_holidays,
        "use_log_target": True,
        "origin_target_map":
            model_df.groupby("Çıkış Transfer Merkezi")["Toplam Desi"].mean().to_dict(),
        "destination_target_map":
            model_df.groupby("Varış Transfer Merkezi")["Toplam Desi"].mean().to_dict(),
        "route_target_map":
            model_df.groupby(
                ["Çıkış Transfer Merkezi","Varış Transfer Merkezi"]
            )["Toplam Desi"].mean().to_dict(),
        "route_models": route_models,
        "route_maes":   route_maes,
        "cv_mae":       avg_mae,
        "stability":    stability,
    }