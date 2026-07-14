# NEURON Logistics AI Engine
### Teknofest 2026 — Gelişmiş Çözüm Aşaması (Saat Bazlı Optimizasyon)

---

## 🎯 Problem Statement

Gelişmiş Çözüm aşamasında lojistik operasyonlar artık **saat bazlı** planlanmalıdır: hangi aracın hangi saatte çıkacağı, transfer merkezlerinde elleçleme süreleri, SLA cezaları ve tır/elleçleme kapasite kısıtları aynı anda yönetilmelidir.

**NEURON Logistics AI Engine**, transfer merkezleri arasındaki 09:00 ve 17:00 taleplerini ayrı ayrı tahmin eder, saat bazlı araç planlaması yapar, elleçleme ve tır kapasitelerine tam uyumlu, SLA cezasını minimize eden, spesifikasyonun her satırına karşı doğrulanmış bir optimizasyon sunar.

Bu proje yalnızca bir tahmin modeli değildir — **her varsayımın test edildiği, her metriğin ölçüldüğü ve her tasarım kararının belgelendiği bir mühendislik sürecinin ürünüdür.**

---

## 📊 Key Results

| Metric | Value |
|---|---|
| **Total Combined Cost** | **10,460,052.36 TL** |
| Baseline Cost (No Optimization) | 27,138,265.17 TL |
| **Total AI Savings** | **16,678,212.81 TL (61.46%)** |
| Unique Vehicles Deployed | 900 |
| Vehicle Utilization | 64% |
| SLA Compliance Rate | 95.04% (request-level, bkz. Metrik Şeffaflığı) |
| **Tır Kapasitesi Violations** | **0** |
| Global CV MAE | 450.92 |
| Global CV WAPE (normalized) | 29.36% |
| CO2 Emission | 142.75 tons |
| CO2 Reduction (vs. no consolidation) | 77.76% |
| High Risk Shipments | 1 / 6,332 (0.02%) |
| System Runtime | ~362 seconds |

> **Formula:** WAPE = CV MAE / Mean Demand × 100 (normalized, fair-comparison metric — bkz. Methodology)

---

## 📁 Dataset Overview

| Dataset | Description |
|---|---|
| `teknofest26_gelismis.xlsx` | Saat bazlı talep verisi (66,024 kayıt, 09:00/17:00) |
| `Araç_Kapasite_Maliyet_Saat.xlsx` | Saatlik araç kiralama/spot maliyetleri |
| `Ellecleme-kapasite.xlsx` | Transfer merkezi başına günlük elleçleme kapasitesi |
| `sehirler_arasi_lojistik.xlsx` | Mesafe, araç tipine göre seyir süresi, SLA hedefleri |
| `Kiralik_Araclar.xlsx` | Günlük zorunlu kiralık araç listesi |
| `tir_kapasiteleri_v2.xlsx` | Transfer merkezi başına günlük tır kapasitesi |

**Toplam:** 66,024 talep kaydı · 289 rota · 18 transfer merkezi · 4 araç tipi (Tır, Kamyon, Hafif Kamyon, Kamyonet)

---

## 🗂️ Repository Structure

```
neuron-logistics-ai/
│
├── data/raw/                          # Yarışma veri setleri
├── outputs/
│   ├── Talep-tahmini.xlsx             # Teslim edilecek talep tahmini
│   └── Tasima-plani.xlsx              # Teslim edilecek taşıma planı
│
├── src/
│   ├── data_loader/                   # load_data.py, preprocess.py
│   ├── features/                      # build_features.py
│   ├── models/                        # train_forecast_model.py, predict_future.py
│   ├── optimization/                  # optimize_shipments_advanced.py
│   ├── risk_engine/                   # risk_analyzer.py
│   ├── analytics/                     # anomaly_detector.py, smart_kpi_engine.py,
│   │                                  # baseline_calculator.py
│   └── output/                        # export_results.py, export_plan.py
│
├── main.py                            # Ana pipeline
├── backtest_forecast.py               # Bağımsız doğrulama scripti
├── digital_twin_demo.py               # Bonus: "What-if" senaryo demosu
├── requirements.txt
└── README.md
```

---

## 🏗️ System Architecture

```
Raw Data (6 Excel files)
        ↓
Preprocessing + Feature Engineering (49 features)
        ↓
Global Ensemble Forecast Model
  ├── LightGBM   (weight: dynamic, ~0.33-0.36)
  ├── XGBoost    (weight: dynamic, ~0.32-0.36)
  └── CatBoost   (weight: dynamic, ~0.31-0.34)
        ↓
Recursive 7-Day Forecast (09:00 + 17:00, ayrı ayrı)
        ↓
Saat Bazlı Optimizasyon
  ├── Step A: Zorunlu Kiralık Dispatch (talep bağımsız)
  ├── Step B: Spot Konsolidasyon (SLA-bazlı, minimum doluluk kısıtı yok)
  ├── Elleçleme Kapasite Kontrolü (gece yarısı proportional split)
  ├── Tır Kapasite Kontrolü (kiralık + spot birlikte izlenir)
  └── Talep ID Doğrulaması (birebir eşleşme — bkz. aşağı)
        ↓
Risk Analysis + Anomaly Detection + CO2 Tracking
        ↓
Output: Talep-tahmini.xlsx
        Tasima-plani.xlsx
```

---

## 🔬 Forecast Model — Methodology & Key Decisions

Bu bölüm, projede alınan her önemli teknik kararın **neden** alındığını, hangi deneyle test edildiğini gösterir. Amacımız yalnızca iyi sonuç almak değil, **her sonucun neden doğru olduğunu kanıtlayabilmekti.**

### Route-Specific vs. Global: Bir Bilimsel Deney

İlk yaklaşımımız, yüksek hacimli 177 rota için ayrı LightGBM modelleri eğitmekti (Route-Specific), düşük hacimli rotalar için Global ensemble'a düşmekti (Hybrid). Ham MAE karşılaştırması yanıltıcıydı:

| | Ham MAE | Görünüş |
|---|---|---|
| Route-Specific | 645.44 | Daha kötü |
| Global | 450.92 | Daha iyi |

Bu karşılaştırma **adil değildi** — farklı popülasyonlar üzerindeydi (177 yüksek-hacimli rota vs. 289 rotanın tümü). Bunu düzeltmek için **normalize edilmiş WAPE** (MAE/ortalama-talep) hesapladık ve ardından **adil bir deney** tasarladık: Global modelin performansını **aynı 177 rotada**, out-of-fold olarak ölçtük.

| | WAPE (aynı 177 rota, out-of-fold) |
|---|---|
| Route-Specific | 38.85% |
| **Global** | **27.89%** |

**Sonuç:** Global ensemble, kendi rotalarında bile Route-Specific'ten üstündür — çünkü 289 rota arasında paylaşılan mevsimsel/tatil/haftanın-günü örüntülerini öğrenir, bu tekil rota geçmişinden daha güçlü bir sinyaldir. Bu kanıta dayanarak **sistem tamamen Global ensemble'a geçirildi** (`ROUTE_MIN_ROWS` parametresi kod tabanında belgelenmiş bir karar olarak korunuyor — jüri kod içinde bu deneyin tam izini görebilir).

### Data Leakage Fix — Out-of-Fold Target Encoding

`Origin_Target_Enc`, `Destination_Target_Enc`, `Route_Target_Enc` özellikleri başlangıçta **tüm veri seti** üzerinden (TimeSeriesSplit'ten önce) hesaplanıyordu — bu, validation fold'una gelecek aylardan bilgi sızdırıyordu (data leakage). Düzeltme: her CV fold'unda bu 3 sütun **yalnızca train kısmından** yeniden hesaplanıp validation'a uygulanıyor. Düzeltme sonrası MAE farkı yalnızca +2.16 (448.76 → 450.92) — bu, modelin gerçek örüntüler öğrendiğinin, leak'e bağımlı olmadığının kanıtıdır.

### Backtest — Gerçek Dünya Doğrulaması

Training CV WAPE (29.36%) yalnızca 1-adım-ileri tahminleri ölçer. Gerçek üretim senaryosu 7 gün **recursive** tahmin gerektirir. Bunu doğrulamak için `backtest_forecast.py` scripti yazıldı: son 7 gün "gizlenir", model yalnızca önceki veriyle eğitilir, `predict_future.py`'nin kendi recursive mantığıyla gizli haftayı tahmin eder ve gerçek değerlerle karşılaştırır.

**Sonuç:**

| Gün | WAPE | Örnek Sayısı |
|---|---|---|
| 1 | 40.89% | 436 |
| 2 | 37.72% | 462 |
| 3 | 43.21% | 430 |
| 4 | 42.30% | 436 |
| 5 | 40.51% | 404 |
| 6 | 42.59% | 370 |
| 7 | 85.79% | **47** ⚠️ |

**Gün 7 istatistiksel olarak güvenilir değildir** — ham veri setinin son günü doğal olarak eksik kapsama sahiptir (47 kayıt vs. diğer günlerde ortalama ~430) — bu bir model hatası değil, kaynak verinin son gününün kesim noktasıyla ilgili bir özelliğidir.

**Gün 1-6'nın bilimsel yorumu:** Bu rakamlar (37.72%–43.21%) monoton artan bir "hata birikimi eğrisi" GÖSTERMEZ — bunun yerine ~38-43% aralığında istikrarlı, gürültülü bir düzey oluştururlar. Bu, recursive tahminin 1-adım CV'ye göre sabit bir ~12 puan performans farkı ("düzey farkı", artan trend değil) gösterdiği anlamına gelir. Bu da **3-Katmanlı Yumuşatma mekanizmasının hatanın katlanarak büyümesini etkili şekilde önlediğinin kanıtıdır** — aksi halde Gün 6, Gün 1'den belirgin şekilde kötü olurdu, ki olmamıştır.

Bu backtest, **production pipeline'ı değiştirmez** — ayrı, bağımsız bir doğrulama scriptidir (`python backtest_forecast.py`).

---

## ⚙️ Optimization Engine

### Saat Bazlı Maliyet Formülü

```
Toplam Araç Maliyeti = Saatlik Kira × (Çıkış Elleçleme + Yolculuk + Varış Elleçleme + Kapasite-Bekleme) 
                        + Mesafe × Km Başı Maliyet
```

Elleçleme süresi: 0.01 dk/desi (çıkış ve varış için ayrı ayrı hesaplanır, konsolidasyonda 2 kez sayılır). Kapasite aşımı nedeniyle oluşan bekleme süresi de kullanım süresine (dolayısıyla maliyete) dahildir — PDF'in kendi örneğine birebir uyumlu (500 dk → 560 dk, 1 saatlik bekleme senaryosu).

### Süre Yuvarlama Kuralı

Teknofest ekibinden gelen resmi bildirime göre: yolculuk ve elleçleme süreleri dakikaya çevrilip **en yakın büyük tam dakikaya** (ceiling) yuvarlanır. Örnek: 0.92 saat = 55.2 dakika → 56 dakika. Bu kural sistemde `get_route_info()` (yolculuk süresi) ve `calc_handling_hours()` (elleçleme süresi) fonksiyonlarında merkezi olarak uygulanır — tüm çağıran kod otomatik olarak yuvarlanmış değerlerle çalışır.

### Zorunlu Kiralık Dispatch

Spesifikasyon gereği: *"Talep yetersiz olsa bile kiralık araçları çıkarmak zorundasınız."* Sistem, talep tahmininden bağımsız bir döngüyle her kiralık kaydı her gün dispatch eder.

### Sınırsız Konsolidasyon (SLA-Bazlı)

Temel İşlevli Çözüm aşamasındaki %10 minimum doluluk kısıtı bu aşamada **kaldırılmıştır** (resmi Soru-Cevap oturumunda teyit edildi). Sistem bunu kullanarak mümkün olduğunca fazla konsolidasyon yapar.

### Tır ve Elleçleme Kapasite Kontrolü

- Her transfer merkezi için günlük tır kapasitesi hem **kiralık hem spot** araçlar için ortak izlenir
- Elleçleme, gece yarısını geçen işlemlerde **orantısal olarak** iki güne bölünür (resmi Q&A örneğine birebir uygun)

### SLA Cezası

```
SLA Cezası = Geciken Desi × Gecikme Süresi (tam saate yuvarlanmış) × 0.4 TL
```

---

## 🆔 Talep ID Format Uyumluluğu — Kritik Bir Düzeltme

Şartname iki kez açıkça belirtiyor: *"Talep ID exceli ile Taşıma planı exceli içindeki talep ID'lerin **birebir eşleşmesi** gerekmektedir."* ve bölünme senaryosu için tam format örneği veriyor: `D00001-1`, `D00001-2`.

**Yaptığımız iç denetimde**, ilk implementasyonumuzun konsolide edilmiş talepleri (`D03781,D03782,...`) veya aralık gösterimiyle (`D03781-D03788`) tek bir hücrede birleştirdiğini, bunun spesifikasyonla **birebir uyuşmadığını** tespit ettik. Bu, otomatik ID-eşleştirmeli bir değerlendirmede güçlü bir diskalifikasyon riskiydi.

**Çözüm:** Optimizasyon motoru artık her Taşıma Planı satırında **tam olarak bir** Talep ID taşıyacak şekilde yeniden yapılandırıldı:

- **Konsolidasyon** (birden çok farklı talep → tek araç): her talep kendi satırında, aynı Araç ID ile görünür
- **Bölünme** (tek talep → kapasite nedeniyle birden çok araç): sistem bunu otomatik tespit edip PDF'in tam istediği `-1`, `-2`, `-3` formatını uygular

Bu düzeltme, tam pipeline çalıştırmaları (`python main.py`) ve öncesi/sonrası konsol çıktı karşılaştırmasıyla doğrulandı — her iki senaryo da (`optimize_shipments_advanced.py` içinde açıkça belgelenmiştir) manuel olarak simüle edilip sonuçlar satır satır incelendi.

---

## 📐 Metrik Şeffaflığı — Request-Level vs. Vehicle-Level

Talep ID düzeltmesinin doğal bir sonucu olarak, Taşıma Planı artık **araç başına değil, talep başına** bir satır içerebilir (bir araç 5-7 farklı müşteri talebini taşıyorsa, 5-7 satır oluşur). Bu, bazı metriklerin *tanımını* değiştirir — **sonucu değil**, sayım granülaritesini:

| Metrik | Anlamı |
|---|---|
| Shipment Count = 6,332 | Talep-ID seviyesinde satır sayısı (her müşteri talebi ayrı izlenir) |
| Unique Vehicles = 900 | Fiziksel araç sayısı (`Araç ID` bazında benzersiz) |
| SLA Violations = 314 | Talep-ID seviyesinde ihlal sayısı — **daha dürüst bir metrik**, çünkü bir aracın gecikmesi artık ona yüklenen her müşteri talebi için ayrı ayrı raporlanır (önceki araç-seviyeli sayım: 24) |
| SLA Cezası (TL) | **Değişmedi** — 223,309.75 TL, çünkü ceza tutarı matematiksel olarak talepler arasında orantılı bölünür; toplam sabit kalır |

CO2 ve Consolidation Rate hesaplamaları da aynı nedenle `Araç ID` bazında benzersizleştirilerek (`drop_duplicates`) yapılır — bir aracın emisyonu, üzerindeki talep sayısı kadar tekrar sayılmaz.

**Bu şeffaflık bilinçli bir tercihtir:** rakamları "iyi görünsün" diye eski (kaba) granülariteye geri döndürmek yerine, daha ayrıntılı ve doğrulanabilir bir izleme sistemini tercih ettik.

---

## ✅ Format Compliance (Diskalifikasyon Önleme)

Yarışma spesifikasyonu net bir uyarı içeriyor: *"Bu formata uymayan takımların sonuçları kesinlikle değerlendirmeye alınmayacaktır."* Bu nedenle export fonksiyonları, dahili analiz amaçlı ekstra sütunları **son Excel çıktılarından çıkarır** — yalnızca spesifikasyonda açıkça belirtilen sütunlar dışa aktarılır:

- `Talep-tahmini.xlsx`: Talep ID, Tarih, Talep Tamamlama Saati (`09:00`/`17:00` formatında), Çıkış TM, Varış TM, Tahmin Edilen Desi (6 sütun)
- `Tasima-plani.xlsx`: Araç ID (`V0001` formatı), Araç Tipi, Araç türü, Çıkış/Varış TM, Çıkış/Varış Tarihi+Saati, Talep ID, Taşınan Desi, Yolculuk süresi, Elleçleme süreleri, SLA cezası, Toplam Maliyet (16 sütun)

Bu ekstra veriler (risk skorları, CO2, WAPE detayları vb.) kod tabanında ve konsol çıktısında tam olarak mevcuttur — yalnızca resmi teslim dosyalarına dahil edilmez.

---

## 🛡️ Risk, Anomaly & Carbon Tracking

| Modül | Ölçüt |
|---|---|
| **Risk Score** | Utilization + Maliyet + Mesafe kombinasyonu |
| **Anomaly Detection** | Düşük utilization + yüksek maliyet + SLA ihlali kombinasyonu |
| **CO2 Emission** | Araç tipine göre g CO2/km × mesafe, benzersiz araç bazında |

---

## 🚀 How to Run

```bash
pip install -r requirements.txt

# Ana pipeline — Talep-tahmini.xlsx ve Tasima-plani.xlsx üretir
python main.py

# Bağımsız doğrulama — gerçek (görünmeyen) veri üzerinde backtest
python backtest_forecast.py

# Bonus: What-if senaryo demosu (bkz. aşağı)
python digital_twin_demo.py
```

**Runtime:** `main.py` ~300-350 saniye · `backtest_forecast.py` ~200 saniye

---

## 📦 Tech Stack

- Python 3.10+ · LightGBM, XGBoost, CatBoost · Pandas, NumPy, Scikit-learn · OpenPyXL

---

## 🔮 Future Work

- **Milk-run / Multi-stop Consolidation** — spesifikasyon bir aracın birden fazla varış noktasına yük taşımasına izin veriyor; mevcut sistem bunu henüz kullanmıyor (en büyük gelecek fırsatı)
- **OR-Tools Bin-Packing** — mevcut greedy buffer mantığının yerine gerçek optimizasyon solver'ı
- **Dinamik Araç Tipi Seçimi** — spot konsolidasyonda araç tipi şu an rota kuyruğunun ilk talebine göre seçiliyor; buffer büyüklüğüne göre yeniden değerlendirme cüzi bir iyileştirme sağlayabilir
- **Explainable AI** — SHAP-benzeri özellik katkı analizi (prototiplendi, format riski nedeniyle üretime alınmadı)
- **Probabilistic Forecasting** — quantile regression ile belirsizlik aralıkları

---

## 🧪 Bonus: Digital Twin "What-If" Demo

> **Not:** Bu bölüm resmi teslimatın bir parçası **değildir**. `Talep-tahmini.xlsx` ve `Tasima-plani.xlsx` dosyalarına dokunmaz, `main.py`'ı değiştirmez — tamamen bağımsız, isteğe bağlı bir demo scriptidir.

`digital_twin_demo.py`, sistemin kapasite kısıtlarındaki değişiklikleri ne kadar hızlı değerlendirebildiğini göstermek için bir "what-if" senaryosu simüle eder: **Bilecik transfer merkezinin elleçleme ve tır kapasitesi 0'a çekilirse ne olur?**

Script, önceden üretilmiş forecast'i (model eğitimi tekrarlanmadan) yeniden kullanarak optimizasyon motorunu iki farklı kısıt setiyle çalıştırır:

| Metrik | Baseline | Bilecik Kapasitesi 0 | Fark |
|---|---|---|---|
| Araç Maliyeti | 10,226,881.51 TL | 10,440,631.67 TL | +213,750.16 TL (+2.09%) |
| **SLA Cezası** | 223,309.75 TL | **638,497.25 TL** | **+415,187.50 TL (+185.9%)** |
| **Optimizasyon Süresi** | — | **1.33 saniye** | — |

**Doğru Yorum:** Sistem, bir transfer merkezindeki kapasite çakışmasını **tespit edip toplam maliyet etkisini 1.33 saniyede sayısallaştırır**. Baskın etki SLA cezasındadır (+185.9%) — sistem yeniden-rotalama yapmaz, gecikmeyi doğru şekilde raporlar. Araç maliyetindeki küçük artış (+2.09%) ise kapasite aşımı nedeniyle oluşan bekleme süresinin — PDF'in kendi formülüne uygun olarak — kullanım süresine (dolayısıyla maliyete) dahil edilmesinden kaynaklanır. Bu, sistemin yalnızca gecikmeyi değil, **gecikmenin gerçek maliyet etkisini de** doğru hesapladığının kanıtıdır.

```bash
python digital_twin_demo.py
```

---

## 🏁 Kapanış Notu

Bu proje boyunca aldığımız her karar üç soruya cevap vermek zorunda kaldı: *Spesifikasyona tam uyuyor mu? Kanıtlanabilir mi? Sistemi bozmadan mı yapıldı?* Route-specific modelleme fikrini kanıt bulamayınca terk ettik. Talep ID formatını spesifikasyonla birebir eşleşene kadar yeniden yazdık. Her iyileştirmeyi, önce ve sonra rakamlarıyla test ettik — işe yaramayanı (varış-tarafı kapasite kontrolü gibi) geri aldık, işe yarayanı belgeledik.

**Sonuç, yalnızca çalışan değil — her satırıyla savunulabilir bir sistemdir.**

---

## 👥 Team

**NEURON** — Teknofest 2026