# NEURON - Log Logistics AI Engine
### Teknofest 2026 — Yapay Zeka Destekli Lojistik Anahat Optimizasyonu

---

## 🎯 Problem Statement

Lojistik operasyonlarda yanlış talep tahmini ve verimsiz araç planlama;

- yüksek operasyon maliyetlerine,
- düşük araç doluluk oranlarına,
- gereksiz sevkiyatlara,
- sürücü ve kaynak israfına

neden olmaktadır.

**NEURON - Log Logistics AI Engine**, transfer merkezleri arasındaki gelecekteki desi talebini tahmin ederek araç planlamasını optimize eder, maliyetleri azaltır ve operasyonel karar desteği sağlar.

---

## 📊 Results

| Metric | Value |
|---|---|
| **Optimized Logistics Cost** | **8,323,113.51 TL** |
| Baseline Cost (No Optimization) | 14,080,448.25 TL |
| **Total AI Savings** | **5,757,334.74 TL (40.89%)** |
| Average Cost Per Desi | 1.2869 TL |
| Vehicle Utilization | 78% |
| Hard Constraint Success | 100% |
| Forecast Accuracy (WAPE) | 80.95% |
| CV MAE | 2,210.80 Desi |
| Mean Historical Demand | 11,603.35 Desi |

> **Formula:** Accuracy = 100 × (1 − MAE / MeanDemand)

> The proposed system reduced logistics costs by **40.89%** while satisfying all operational constraints and maintaining **100% shipment feasibility**.

---

## 🏆 Key Achievements

- **40.89%** logistics cost reduction vs baseline
- **100%** operational feasibility — all hard constraints satisfied
- **16.21%** shipment consolidation rate (101 trips saved)
- **80.95%** forecasting accuracy (WAPE based)
- **0** critical operational anomalies detected

---

## 📁 Dataset Overview

Sistem aşağıdaki veri kaynaklarını kullanmaktadır:

| Dataset | Description |
|---|---|
| `Desi_talep.xlsx` | Tarihsel desi talep verileri (Ocak–Nisan 2026) |
| `Koordinatlar v2.xlsx` | Transfer merkezi koordinatları |
| `Kiralik_Araclar.xlsx` | Mevcut kiralık araç bilgileri |
| `Arac_Kapasite_Maliyet.xlsx` | Araç kapasite ve maliyet bilgileri |

**Toplam Veri Boyutu:**
- 10,770 talep kaydı
- 89 rota
- 18 transfer merkezi

---

## 🗂️ Repository Structure

```
neuron-logistics-ai/
│
├── data/
│   ├── Desi_talep.xlsx
│   ├── Koordinatlar v2.xlsx
│   ├── Kiralik_Araclar.xlsx
│   └── Arac_Kapasite_Maliyet.xlsx
│
├── outputs/
│   ├── Tahminlenen_Talep.xlsx
│   └── Arac_Planlama.xlsx
│
├── src/
│   ├── data_loader/
│   ├── features/
│   ├── models/
│   ├── optimization/
│   ├── risk_engine/
│   ├── analytics/
│   └── realtime/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🏗️ System Architecture

```
Raw Data (4 Excel files)
        ↓
Feature Engineering (48 features)
        ↓
Ensemble Forecast Model
  ├── LightGBM  (weight: 0.355)
  ├── XGBoost   (weight: 0.341)
  └── CatBoost  (weight: 0.303)
        ↓
Recursive 7-Day Forecast (May 11–17)
        ↓
AI Load Consolidation Engine
        ↓
Shipment Optimization
  ├── Step 1: Rental Vehicle Priority
  └── Step 2: Spot Hybrid AI Engine
        ↓
Risk Analysis + Anomaly Detection
        ↓
Output: Tahminlenen_Talep.xlsx
        Arac_Planlama.xlsx
```

---

## 🔧 Feature Engineering

Model toplam **48 özellik** kullanmaktadır. Önemli özellik grupları:

| Grup | Özellikler | Amaç |
|---|---|---|
| **Lag Features** | lag_1, lag_2, lag_7, lag_14 | Kısa/orta vadeli hafıza |
| **Rolling Statistics** | rolling_mean_3/7/14, rolling_std, rolling_max | Trend ve volatilite |
| **Holiday Distance** | days_to_holiday, is_public_holiday, before/after_holiday | Tatil etkisi |
| **Route Encoding** | Rota_Code, Route_Target_Enc, Origin/Dest_Target_Enc | Rota bazlı davranış |
| **Seasonal Features** | season, is_winter/spring/summer/autumn, ay_sin/cos | Mevsimsellik |
| **Ratio Features** | lag_ratio_1_7, rolling_ratio | Momentum göstergeleri |
| **Cyclical Features** | month_sin/cos, day_sin/cos | Döngüsel zaman |

### Top 10 Feature Importance (LightGBM)

```
lag_1            ████████████████████  3609
day              ████████████████████  3605
Rota_Code        ████████████████      2968
rolling_ratio    ███████████████       2821
lag_7            ███████████████       2752
days_to_holiday  ██████████████        2599
lag_ratio_1_7    ██████████████        2579
Route_Target_Enc █████████████         2476
lag_14           █████████████         2356
lag_2            ████████████          2285
```

Bu özellikler sayesinde rota bazlı davranış ve sezon etkileri modellenmiştir.

---

## 🤖 Forecast Model

**Method:** Recursive (1-step-ahead) Ensemble Forecasting
**Validation:** 8-Fold Time Series Cross Validation
**Horizon:** 7 days (May 11–17, 2026)

### Cross-Validation Results

| Fold | Period | MAE | Note |
|---|---|---|---|
| 1 | Jan 06–23 | 2885.77 | Small train set — expected |
| 2 | Jan 24 – Feb 09 | 1980.47 | |
| 3 | Feb 09–23 | 1538.44 | Best fold |
| 4 | Feb 23 – Mar 11 | 1739.90 | |
| 5 | Mar 11–25 | 2773.52 | Ramadan transition — expected |
| 6 | Mar 25 – Apr 10 | 2542.62 | |
| 7 | Apr 10–24 | 1925.20 | |
| 8 | Apr 24 – May 10 | 2300.44 | |
| **Avg** | | **2210.80** | |

> Fold 1 ve Fold 5'teki yüksek MAE, model kararsızlığından değil, veri kaynaklı
> nedenlerden (küçük eğitim seti ve Ramazan geçiş dönemi) kaynaklanmaktadır.

### Error Control (3-Layer Smoothing)
- **Layer 1:** lag_1 = 70% prediction + 30% historical mean
- **Layer 2:** Outlier clipping (±2.5 std)
- **Layer 3:** Memory smoothing (80/20)

---

## ⚙️ Optimization Engine

### Vehicle Assignment Strategy
1. **Rental Priority** — mevcut kiralık araçlar önce kullanılır
2. **Spot Hybrid AI** — kalan talep spot araçlarla karşılanır
3. **Fallback Guarantee** — hard constraint %100 sağlanır

### Shipment Breakdown
```
Consolidation Output : 522 routes
After Vehicle Split  : 539 plan rows
Split Rows Added     : 17 (multi-vehicle type assignments)
```

### Driver Assignment
```
Single Driver Routes : 422  (< 700 km)
Double Driver Routes : 117  (> 700 km, yasal zorunluluk)
Total Drivers        : 656
```
> Türkiye Karayolları mevzuatı gereği 700 km+ rotalar için 2 sürücü zorunludur.
> Bu nedenle TOTAL DRIVERS > TOTAL SHIPMENTS normaldir.

---

## 🚗 Vehicle Utilization Note

Average utilization is **78%**.

The optimization engine prioritizes cost minimization while maintaining
hard operational constraints (capacity, driver availability, route compatibility).

Increasing utilization beyond 80% was tested but resulted in higher logistics
costs (+147K TL), therefore the current configuration was selected as the
optimal business trade-off between utilization rate and total cost.

---

## 🛡️ Risk & Anomaly Design

Sistem iki ayrı risk modülü içerir:

| Modül | Ölçüt | Kaynak |
|---|---|---|
| **Operational Anomaly** | Utilization + Cost + Risk birleşimi | İç operasyon |
| **Delay Risk** | Hava + Trafik olasılığı | Dış faktör |

> Yüksek gecikme ihtimali (HIGH Delay) tek başına operasyonel anomali sayılmaz.
> Bu tasarım kararıdır — dış riskler ayrı modülde izlenir.

---

## 📄 Output Files

| File | Description |
|---|---|
| `outputs/Tahminlenen_Talep.xlsx` | 11–17 Mayıs 2026 haftalık talep tahmini (623 satır) |
| `outputs/Arac_Planlama.xlsx` | Araç atama ve maliyet planı (539 satır) |

---

## 🔁 Reproducibility

Running `python main.py` with the provided datasets reproduces all reported
results and generates the submission files automatically.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

**Runtime:** ~130–140 seconds
**Output:** outputs/ klasörüne otomatik kaydedilir

---

## 📦 Tech Stack

- Python 3.10+
- LightGBM, XGBoost, CatBoost
- Pandas, NumPy, Scikit-learn
- OpenPyXL (Excel export)

---

## 🔮 Future Work

- **Probabilistic Forecasting** — Quantile regression ile tahmin aralıkları
- **Route-Specific Models** — Her rota için ayrı model (89 route)
- **Real-Time GPS Integration** — Canlı araç takibi ile optimizasyon
- **Dynamic Traffic APIs** — Gerçek zamanlı trafik verisi entegrasyonu
- **Reinforcement Learning Fleet Optimization** — RL tabanlı filo yönetimi
- **Carbon Emission Optimization** — Çevre dostu rota planlaması

---

## 📬 Submission Files

This repository generates the two official competition deliverables:

1. `Tahminlenen_Talep.xlsx` — 11–17 Mayıs 2026 demand forecast
2. `Arac_Planlama.xlsx` — Vehicle assignment and cost plan

Both files are automatically produced after running:

```bash
python main.py
```

---

## 👥 Team

**NEURON - Log** — Teknofest 2026