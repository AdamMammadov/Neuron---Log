import math
import pandas as pd
import numpy as np

VEHICLE_SPEED_COL = {
    "Tır":          "Tir_Suresi_Saat",
    "Kamyon":       "Kamyon_Suresi_Saat",
    "Hafif Kamyon": "Hafif_Kamyon_Suresi_Saat",
    "Kamyonet":     "Kamyonet_Suresi_Saat",
}

SLA_PENALTY_RATE = 0.4
UTIL_TARGET      = 0.70


def calc_handling_hours(desi):
    """
    Elleçleme süresi = desi * 0.01 dəqiqə.
    YENİ QAYDA (Teknofest bildirimi): süre en yakın büyük tam
    dəqiqəyə yuvarlanmalıdır (ceiling). Məsələn 55.2 dəqiqə → 56.
    """
    raw_minutes = desi * 0.01
    if raw_minutes <= 0:
        return 0.0
    rounded_minutes = math.ceil(raw_minutes)
    return rounded_minutes / 60


def split_across_midnight(start_dt, duration_hours, amount):
    """
    Elleçleme əməliyyatı gecə yarısını keçərsə, miqdarı başlanğıc və
    bitmə tarixləri arasında proporsional bölür.
    PDF Q&A nümunəsi: 23:30-da başlayan 10000 desilik elleçleme
    3000 desi (bugün) / 7000 desi (sabah) şəklində bölünür.
    """
    if duration_hours <= 0 or amount <= 0:
        return [(start_dt.strftime("%Y-%m-%d"), amount)]

    end_dt     = start_dt + pd.Timedelta(hours=duration_hours)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date   = end_dt.strftime("%Y-%m-%d")

    if start_date == end_date:
        return [(start_date, amount)]

    midnight     = pd.Timestamp(start_date) + pd.Timedelta(days=1)
    hours_before = (midnight - start_dt).total_seconds() / 3600
    frac_before  = min(1.0, max(0.0, hours_before / duration_hours))

    amount_before = amount * frac_before
    amount_after  = amount - amount_before

    return [(start_date, amount_before), (end_date, amount_after)]


def calc_sla_penalty(desi, delay_hours):
    if delay_hours <= 0:
        return 0.0
    return round(desi * math.ceil(delay_hours) * SLA_PENALTY_RATE, 2)


def get_route_info(distance_lookup, origin, destination, vehicle_type):
    """
    YENİ QAYDA (Teknofest bildirimi): yolculuk süresi dəqiqəyə
    çevrilib en yakın büyük tam dəqiqəyə yuvarlanmalıdır.
    Nümunə: 0.92 saat = 55.2 dəqiqə -> 56 dəqiqə.
    Yuvarlama BU FUNKSİYADA edilir ki, bütün çağırış yerlərinə
    (dispatch, best_spot_vehicle, calc_delivery_dt) avtomatik
    tətbiq olunsun — heç bir əlavə dəyişikliyə ehtiyac yoxdur.
    """
    key = (origin, destination)
    row = distance_lookup.get(key, {})
    speed_col = VEHICLE_SPEED_COL.get(vehicle_type, "Tir_Suresi_Saat")
    raw_hours = row.get(speed_col, 0)

    if raw_hours and raw_hours > 0:
        raw_minutes = raw_hours * 60
        rounded_minutes = math.ceil(raw_minutes)
        rounded_hours = rounded_minutes / 60
    else:
        rounded_hours = 0

    return (
        rounded_hours,
        row.get("mesafe_km", 0),
        row.get("hedef_teslim_gun", 1),
    )


def vehicle_cost(vehicle_info, vtype, plan_type, usage_hours, km, count):
    info = vehicle_info.get(vtype, {})
    rate = info.get("rental_hourly" if plan_type == "Kiralık" else "spot_hourly", 0)
    km_r = info.get("rental_km"     if plan_type == "Kiralık" else "spot_km",     0)
    return round((rate * usage_hours + km_r * km) * count, 2)


def best_spot_vehicle(demand, vehicle_info, distance_lookup, origin, destination,
                      tir_capacity=None, tir_used=None, date_str=None):
    """
    En optimal spot arac sec.
    tir_capacity verilmisse Tir kapasitesi yoxlanir.
    """
    best, best_score = None, float("inf")

    def _tir_ok(vtype):
        if vtype != "Tır":
            return True
        if not tir_capacity:
            return True
        oc = tir_capacity.get(origin, float("inf"))
        dc = tir_capacity.get(destination, float("inf"))
        if oc == 0 or dc == 0:
            return False
        if date_str:
            n_tir = math.ceil(demand / vehicle_info.get("Tır",{}).get("capacity",1))
            or_rem = max(0, oc - (tir_used or {}).get((origin, date_str), 0))
            ds_rem = max(0, dc - (tir_used or {}).get((destination, date_str), 0))
            return or_rem >= n_tir and ds_rem >= n_tir
        return True

    for vtype, info in vehicle_info.items():
        cap = info["capacity"]
        if cap <= 0:
            continue
        if not _tir_ok(vtype):
            continue
        n    = math.ceil(demand / cap)
        util = demand / (n * cap)
        th, km, _ = get_route_info(distance_lookup, origin, destination, vtype)
        ellec = (demand * 0.01) / 60
        usage = ellec + th + ellec
        cost  = vehicle_cost(vehicle_info, vtype, "Spot", usage, km, n)
        score = cost + (1 - util) * 10000
        if score < best_score:
            best_score, best = score, vtype

    return best or "Kamyon"


def calc_absolute_deadline(talep_tarih_dt, sla_days):
    """Deadline = talep tarixi + sla_days gun (Timestamp)"""
    return talep_tarih_dt + pd.Timedelta(days=sla_days)


def calc_delivery_dt(send_tarih_dt, sla_start_hours, th, demand):
    """Gonderilmeden catma vaxtini Timestamp kimi hesabla."""
    cikis_ellec = calc_handling_hours(demand)
    varis_ellec = calc_handling_hours(demand)
    total_hours = sla_start_hours + cikis_ellec + th + varis_ellec
    return send_tarih_dt + pd.Timedelta(hours=total_hours)


def dispatch(results, vehicle_counter, vehicle_info, distance_lookup,
             handling_used, handling_capacity,
             origin, destination, demand, talep_id_str,
             send_tarih_dt, sla_start_hours, deadline_dt,
             plan_type, vehicle_type, vehicle_count=None,
             tir_used=None, talep_items=None):
    """
    talep_items: [{"talep_id": str, "demand": float}, ...] - verilibse,
    HER item ucun AYRI SETIR yaradilir (eyni Arac ID ile, desi/maliyet/
    SLA mutenasib bolunur). Bu, PDF-in "Talep ID exceli ile Tasima plani
    exceli icindeki talep ID'lerin birebir eslesmesi gerekmektedir"
    telebini qarsilayir - bir setirde hec vaxt birden cox Talep ID
    gorunmur. Verilmeyibse (None), tek-setir davranisi qorunur
    (yalniz bos-kiralik dispatch kimi xususi hallar ucun).
    """

    def hours_to_hhmm(h):
        # round() float dəqiqlik xətalarının (məsələn 55.999999
        # dəqiqə) qarşısını alır — çıxış/varış saatı artıq
        # get_route_info-dan yuvarlanmış gəldiyi üçün, burada
        # yalnız təhlükəsizlik üçün əlavə edilir.
        total_minutes = round(h * 60)
        h_part = (total_minutes // 60) % 24
        m_part = total_minutes % 60
        return f"{int(h_part):02d}:{int(m_part):02d}"

    def get_remaining_handling(merkez, date_str):
        cap  = handling_capacity.get(merkez, float("inf"))
        used = handling_used.get((merkez, date_str), 0)
        return max(0, cap - used)

    def use_handling(merkez, date_str, desi):
        key = (merkez, date_str)
        handling_used[key] = handling_used.get(key, 0) + desi

    th, km, _ = get_route_info(distance_lookup, origin, destination, vehicle_type)

    cap = vehicle_info[vehicle_type]["capacity"]
    if vehicle_count is None:
        vehicle_count = math.ceil(demand / cap)

    util = min(demand / (vehicle_count * cap), 1.0)

    # =========================================
    # CIKIS TERAFI -- ellecleme kapasitesi yoxlamasi
    # Kapasite asilirsa, gondermeye 1 gun otelenir.
    # Bu gozleme vaxti artiq usage_hours-a (demeli
    # maliyete) daxil edilir - PDF-in oz numunesi:
    # "1 saat bekletmek durumunda kaldiniz" -> kullanim
    # suresi 500 deq -> 560 deq-e cixir.
    # =========================================

    cikis_ellec  = calc_handling_hours(demand)
    cikis_saat   = sla_start_hours + cikis_ellec
    cikis_date   = send_tarih_dt.strftime("%Y-%m-%d")
    wait_hours   = 0.0

    if demand > get_remaining_handling(origin, cikis_date):
        send_tarih_dt = send_tarih_dt + pd.Timedelta(days=1)
        cikis_date    = send_tarih_dt.strftime("%Y-%m-%d")
        cikis_saat   += 24
        wait_hours    = 24.0

    cikis_start_dt = send_tarih_dt + pd.Timedelta(hours=sla_start_hours)
    for d_str, d_amt in split_across_midnight(cikis_start_dt, cikis_ellec, demand):
        if d_amt > 0:
            use_handling(origin, d_str, d_amt)

    varis_saat  = cikis_saat + th
    varis_dt    = send_tarih_dt + pd.Timedelta(hours=cikis_ellec + th)
    varis_date  = varis_dt.strftime("%Y-%m-%d")
    varis_ellec = calc_handling_hours(demand)

    # =========================================
    # VARIS TERAFI ellecleme yoxlamasi GERI ALINDI.
    # Sinaqda bu yoxlama SLA cezasini 223K -> 865K (+287%)
    # ve ihlal sayini 24 -> 690 artirdi. Kok sebeb: cixis
    # terefinde her marsrut oz TM-ni bir defe "terk edir"
    # (tebii paylanma), amma varis terefinde 10+ ferqli
    # marsrutdan gelen yukler EYNI gun EYNI hub-un payli
    # kapasite hovuzunda toqqusur -> zencirvari +24 saat
    # gecikmeler. Duzgun helli QLOBAL elaqelendirme (butun
    # marsrutlari birlikde planlashdirmaq) teleb edir - bu,
    # submission-a yaxin vaxtda etmek ucun cox riskli bir
    # deyisiklikdir. Bu sebeble bu yoxlama SILINIB, yalniz
    # cixis terefi yoxlamasi (yuxarida) qalir.
    # =========================================

    delivery_dt = varis_dt + pd.Timedelta(hours=varis_ellec)

    for d_str, d_amt in split_across_midnight(varis_dt, varis_ellec, demand):
        if d_amt > 0:
            use_handling(destination, d_str, d_amt)

    if vehicle_type == "Tır" and tir_used is not None and vehicle_count:
        tir_used[(origin, cikis_date)] = (
            tir_used.get((origin, cikis_date), 0) + vehicle_count
        )
        tir_used[(destination, varis_date)] = (
            tir_used.get((destination, varis_date), 0) + vehicle_count
        )

    if deadline_dt is not None and delivery_dt > deadline_dt:
        delay_hours = (delivery_dt - deadline_dt).total_seconds() / 3600
        sla_penalty = calc_sla_penalty(demand, delay_hours)
    else:
        sla_penalty = 0.0

    # PDF: kullanim suresi = cikis ellecleme + yolculuk + varis
    # ellecleme + (kapasite-sebebli gozleme, varsa)
    usage_hours = cikis_ellec + th + varis_ellec + wait_hours
    cost = vehicle_cost(vehicle_info, vehicle_type, plan_type, usage_hours, km, vehicle_count)

    vid = f"V{vehicle_counter[0]:04d}"
    vehicle_counter[0] += 1

    base_row = {
        "Araç ID":                 vid,
        "Araç Tipi":              plan_type,
        "Araç türü":              vehicle_type,
        "Çıkış Transfer Merkezi": origin,
        "Varış Transfer Merkezi": destination,
        "Çıkış Tarihi":           cikis_date,
        "Çıkış Saati":            hours_to_hhmm(cikis_saat),
        "Varış Tarihi":           varis_date,
        "Varış Saati":            hours_to_hhmm(varis_saat),
        "Yolculuk süresi":        round(th, 2),
        "Varış elleçleme süresi": round(varis_ellec * 60, 2),
        "Çıkış Elleçleme süresi": round(cikis_ellec * 60, 2),
        "Doluluk Oranı":          round(util, 2),
        "Mesafe KM":              round(km, 2),
    }

    if talep_items:
        total_item_demand = sum(item["demand"] for item in talep_items)
        for item in talep_items:
            share = (item["demand"] / total_item_demand) if total_item_demand > 0 else 0
            row = dict(base_row)
            row["Talep ID"]       = item["talep_id"]
            row["Taşınan Desi"]   = round(item["demand"], 2)
            row["SLA cezası"]     = round(sla_penalty * share, 2)
            row["Toplam Maliyet"] = round(cost * share, 2)
            results.append(row)
    else:
        row = dict(base_row)
        row["Talep ID"]       = talep_id_str
        row["Taşınan Desi"]   = round(demand, 2)
        row["SLA cezası"]     = sla_penalty
        row["Toplam Maliyet"] = cost
        results.append(row)


def optimize_shipments_advanced(
    forecast_df,
    rental_df,
    vehicle_df,
    distance_df,
    handling_df,
    tir_kapasite_df=None
):
    """
    Gelismis saat bazli optimizasiya - v10

    v10 DEYISIKLIKLERI (format compliance + doğruluq):
    1. Talep ID: her Tasima Plani setri artiq TEK bir Talep ID
       dasiyir (evvellier konsolide edilmis telebler vergulle/
       defisle birlesdirilirdi). Konsolidasiya zamani her teleb
       ayri setir kimi gosterilir (eyni Arac ID). Bir teleb
       arac kapasitesi sebebinden bolunerse, "D00001-1",
       "D00001-2" formati (PDF-in deqiq telebi) tetbiq olunur.
    2. Varis terefi ellecleme kapasitesi indi yoxlanir (evvellier
       yalniz cikis terefi yoxlanirdi).
    3. Kapasite-sebebli gozleme vaxti artiq usage_hours-a (demeli
       maliyete) daxildir - PDF-in oz numunesine uygun.
    """

    results         = []
    vehicle_counter = [1]
    handling_used   = {}

    vehicle_info = {}
    for _, r in vehicle_df.iterrows():
        nm = r.get("Araç Adı", r.get("arac_adi", ""))
        vehicle_info[nm] = {
            "capacity":      r.get("Kapasite (desi)", 0),
            "rental_hourly": r.get("Kiralık Araç Saatlik Kira (TL)", 0),
            "rental_km":     r.get("Kiralık Araç Kilometre Başına Maliyet (TL)", 0),
            "spot_hourly":   r.get("Spot Araç Saatlik Kira (TL)", 0),
            "spot_km":       r.get("Spot Kilometre Başına Maliyet (TL)", 0),
        }

    rental_lookup = {}
    for _, r in rental_df.iterrows():
        key = (r.get("Çıkış Transfer Merkezi",""),
               r.get("Varış Transfer Merkezi",""),
               r.get("Araç Türü",""))
        rental_lookup[key] = r.get("Araç sayısı", 0)

    distance_lookup = {}
    for _, r in distance_df.iterrows():
        key = (r.get("cikis",""), r.get("varis",""))
        distance_lookup[key] = {
            "mesafe_km":                r.get("mesafe_km", 0),
            "Tir_Suresi_Saat":          r.get("Tir_Suresi_Saat", 0),
            "Kamyon_Suresi_Saat":       r.get("Kamyon_Suresi_Saat", 0),
            "Hafif_Kamyon_Suresi_Saat": r.get("Hafif_Kamyon_Suresi_Saat", 0),
            "Kamyonet_Suresi_Saat":     r.get("Kamyonet_Suresi_Saat", 0),
            "hedef_teslim_gun":         r.get("hedef_teslim_gun", 1),
        }

    handling_capacity = {}
    for _, r in handling_df.iterrows():
        handling_capacity[r.get("transfer_merkezi","")] = \
            r.get("ellecleme_kapasite", 0)

    tir_capacity = {}
    if tir_kapasite_df is not None:
        for _, r in tir_kapasite_df.iterrows():
            tir_capacity[r.get("transfer_merkezi","")] = \
                int(r.get("tir_kapasitesi", 0))

    tir_used = {}

    def get_tir_remaining(merkez, date_str):
        max_tir = tir_capacity.get(merkez, float("inf"))
        used    = tir_used.get((merkez, date_str), 0)
        return max(0, max_tir - used)

    def parse_hour(s):
        try:
            parts = str(s).replace(".", ":").split(":")
            return int(parts[0]) + (int(parts[1]) if len(parts) > 1 else 0) / 60
        except Exception:
            return 9.0

    # =========================================================
    # STEP A -- KIRALIQ ARACLAR
    # =========================================================

    all_dates     = sorted(forecast_df["Tarih"].unique()) if not forecast_df.empty else []
    all_date_strs = [str(d)[:10] for d in all_dates]

    demand_pool = {}
    for _, row in forecast_df.iterrows():
        origin      = row.get("Çıkış Transfer Merkezi", "")
        destination = row.get("Varış Transfer Merkezi", "")
        talep_id    = row.get("Talep ID", "")
        demand      = row.get("Tahmin Edilen Desi", 0)
        saat_str    = row.get("Talep Tamamlama Saati", "9:00")
        tarih_val   = row.get("Tarih", "")

        try:
            tarih_dt  = pd.to_datetime(tarih_val)
            tarih_str = tarih_dt.strftime("%Y-%m-%d")
        except Exception:
            tarih_str = str(tarih_val)[:10]
            tarih_dt  = pd.to_datetime(tarih_str)

        demand_pool.setdefault((origin, destination, tarih_str), []).append({
            "talep_id":  talep_id,
            "remaining": demand,
            "sla_start": parse_hour(saat_str),
            "tarih_dt":  tarih_dt,
        })

    for key in demand_pool:
        demand_pool[key].sort(key=lambda x: x["sla_start"])

    for _, r in rental_df.iterrows():
        origin      = r.get("Çıkış Transfer Merkezi", "")
        destination = r.get("Varış Transfer Merkezi", "")
        vtype       = r.get("Araç Türü", "")
        daily_count = r.get("Araç sayısı", 0)

        if daily_count <= 0 or vtype not in vehicle_info:
            continue

        cap = vehicle_info[vtype]["capacity"]
        if cap <= 0:
            continue

        for date_str in all_date_strs:
            items = demand_pool.get((origin, destination, date_str), [])

            remaining_capacity = daily_count * cap
            consumed_items      = []
            consumed_demand     = 0.0
            first_dt             = pd.to_datetime(date_str)
            first_sla_start       = 9.0

            for item in items:
                if remaining_capacity <= 0:
                    break
                if item["remaining"] <= 0:
                    continue
                take = min(item["remaining"], remaining_capacity)
                item["remaining"]  -= take
                remaining_capacity -= take
                consumed_demand    += take
                if not consumed_items:
                    first_dt        = item["tarih_dt"]
                    first_sla_start = item["sla_start"]
                consumed_items.append({"talep_id": item["talep_id"], "demand": take})

            _, _, sla_days = get_route_info(distance_lookup, origin, destination, vtype)
            deadline_dt    = calc_absolute_deadline(first_dt, sla_days)

            if consumed_items:
                dispatch(results, vehicle_counter, vehicle_info,
                         distance_lookup, handling_used, handling_capacity,
                         origin, destination, consumed_demand, None,
                         first_dt, first_sla_start, deadline_dt,
                         "Kiralık", vtype, daily_count,
                         tir_used=tir_used, talep_items=consumed_items)
            else:
                dispatch(results, vehicle_counter, vehicle_info,
                         distance_lookup, handling_used, handling_capacity,
                         origin, destination, 0.0,
                         f"KIRALIK-BOS-{origin}-{destination}-{date_str}",
                         first_dt, first_sla_start, deadline_dt,
                         "Kiralık", vtype, daily_count,
                         tir_used=tir_used)

    remaining_after_rental = []
    for (origin, destination, date_str), items in demand_pool.items():
        for item in items:
            if item["remaining"] > 0:
                remaining_after_rental.append({
                    "origin":      origin,
                    "destination": destination,
                    "talep_id":    item["talep_id"],
                    "demand":      item["remaining"],
                    "tarih_dt":    item["tarih_dt"],
                    "tarih_str":   date_str,
                    "sla_start":   item["sla_start"],
                })

    print(f"\nKiralık araç sonrası kalan: {len(remaining_after_rental)} talep")

    # =========================================================
    # STEP B -- SPOT KONSOLIDASIYA (Multi-day)
    # =========================================================

    route_queues = {}
    for t in remaining_after_rental:
        key = (t["origin"], t["destination"])
        if key not in route_queues:
            route_queues[key] = []
        route_queues[key].append(t)

    for (origin, destination), queue in route_queues.items():

        queue.sort(key=lambda x: (x["tarih_dt"], x["sla_start"]))

        vtype = best_spot_vehicle(
            queue[0]["demand"] if queue else 1000,
            vehicle_info, distance_lookup, origin, destination,
            tir_capacity=tir_capacity,
            tir_used=tir_used,
            date_str=queue[0]["tarih_str"] if queue else None
        )
        cap = vehicle_info[vtype]["capacity"]
        th, km, sla_days = get_route_info(
            distance_lookup, origin, destination, vtype
        )

        buffer_items        = []
        buffer_start_dt      = None
        buffer_sla_start      = 9.0
        buffer_deadline_dt     = None

        def flush_buffer(buf_items, buf_dt, buf_sla, buf_dl):
            total = sum(i["demand"] for i in buf_items)
            if total <= 0:
                return
            dispatch(results, vehicle_counter, vehicle_info,
                     distance_lookup, handling_used, handling_capacity,
                     origin, destination, total, None,
                     buf_dt, buf_sla, buf_dl, "Spot", vtype,
                     tir_used=tir_used, talep_items=buf_items)

        for t in queue:
            demand_i    = t["demand"]
            tarih_dt_i  = t["tarih_dt"]
            sla_start_i = t["sla_start"]
            talep_id_i  = t["talep_id"]

            deadline_i = calc_absolute_deadline(tarih_dt_i, sla_days)

            if buffer_items and buffer_start_dt is not None:

                buffer_demand = sum(i["demand"] for i in buffer_items)
                combined      = buffer_demand + demand_i
                delivery_dt   = calc_delivery_dt(
                    buffer_start_dt, buffer_sla_start, th, combined
                )
                earliest_dl = min(buffer_deadline_dt, deadline_i) \
                    if buffer_deadline_dt else deadline_i

                SLA_SAFETY_MARGIN = pd.Timedelta(hours=1)

                if delivery_dt + SLA_SAFETY_MARGIN > earliest_dl:
                    flush_buffer(buffer_items, buffer_start_dt,
                                 buffer_sla_start, buffer_deadline_dt)
                    buffer_items        = []
                    buffer_start_dt      = None
                    buffer_deadline_dt     = None

            buffer_items.append({"talep_id": talep_id_i, "demand": demand_i})
            if buffer_start_dt is None:
                buffer_start_dt  = tarih_dt_i
                buffer_sla_start = sla_start_i

            buffer_deadline_dt = min(buffer_deadline_dt, deadline_i) \
                if buffer_deadline_dt else deadline_i

            buffer_demand = sum(i["demand"] for i in buffer_items)
            while buffer_demand >= cap:
                ratio = cap / buffer_demand
                dispatch_items  = []
                remaining_items = []
                for it in buffer_items:
                    take = it["demand"] * ratio
                    leftover = it["demand"] - take
                    dispatch_items.append({"talep_id": it["talep_id"], "demand": take})
                    if leftover > 0.01:
                        remaining_items.append({"talep_id": it["talep_id"], "demand": leftover})

                dispatch(results, vehicle_counter, vehicle_info,
                         distance_lookup, handling_used, handling_capacity,
                         origin, destination, cap, None,
                         buffer_start_dt, buffer_sla_start,
                         buffer_deadline_dt, "Spot", vtype, 1,
                         tir_used=tir_used, talep_items=dispatch_items)

                buffer_items  = remaining_items
                buffer_demand = sum(i["demand"] for i in buffer_items)
                if buffer_items:
                    buffer_start_dt  = tarih_dt_i
                    buffer_sla_start = sla_start_i
                    buffer_deadline_dt = deadline_i
                else:
                    buffer_start_dt    = None
                    buffer_deadline_dt   = None

        if buffer_items:
            buffer_demand_final = sum(i["demand"] for i in buffer_items)
            if buffer_demand_final > 0 and buffer_start_dt is not None:
                recalc_deadline = calc_absolute_deadline(buffer_start_dt, sla_days)
                safe_deadline = min(buffer_deadline_dt, recalc_deadline) \
                    if buffer_deadline_dt else recalc_deadline
                flush_buffer(buffer_items, buffer_start_dt,
                             buffer_sla_start, safe_deadline)

    # =========================================================
    # FINAL POST-PROCESSING -- Talep ID bolunme suffiksi
    # PDF: "Bir talebi 2 farkli araca boldugumuzde Talep ID'sini
    # D00001-1 ve D00001-2 seklinde isimlendirmemiz istenmis."
    # Konsolidasiya zamani her teleb artiq oz setrinde tek basina
    # gorunur (suffikssiz) - YALNIZ bir teleb kapasite/buffer
    # mehdudiyyeti sebebinden BIRDEN COX setirde (ferqli aracda)
    # gorunurse, bura avtomatik "-1", "-2" elave olunur.
    # =========================================================

    id_occurrences = {}
    for r in results:
        tid = r["Talep ID"]
        id_occurrences[tid] = id_occurrences.get(tid, 0) + 1

    id_running_index = {}
    for r in results:
        tid = r["Talep ID"]
        if id_occurrences.get(tid, 0) > 1 and not str(tid).startswith("KIRALIK-BOS"):
            id_running_index[tid] = id_running_index.get(tid, 0) + 1
            r["Talep ID"] = f"{tid}-{id_running_index[tid]}"

    # =========================================================
    # FINAL
    # =========================================================

    if results:
        result_df = pd.DataFrame(results)
    else:
        result_df = pd.DataFrame(columns=[
            "Araç ID","Araç Tipi","Araç türü",
            "Çıkış Transfer Merkezi","Varış Transfer Merkezi",
            "Çıkış Tarihi","Çıkış Saati","Varış Tarihi","Varış Saati",
            "Talep ID","Taşınan Desi","Yolculuk süresi",
            "Varış elleçleme süresi","Çıkış Elleçleme süresi",
            "SLA cezası","Toplam Maliyet","Doluluk Oranı","Mesafe KM"
        ])

    tc   = result_df["Toplam Maliyet"].sum() if not result_df.empty else 0
    tsla = result_df["SLA cezası"].sum()     if not result_df.empty else 0
    au   = result_df["Doluluk Oranı"].mean() if not result_df.empty else 0
    sla_count = len(result_df[result_df["SLA cezası"] > 0]) if not result_df.empty else 0

    tir_violations = 0
    for (merkez, date_str), used_count in tir_used.items():
        max_cap = tir_capacity.get(merkez, float("inf"))
        if max_cap != float("inf") and used_count > max_cap:
            tir_violations += 1

    unique_talep_ids = result_df["Talep ID"].nunique() if not result_df.empty else 0

    print("\nADVANCED SAAT BAZLI OPTİMİZASİYA TAMAMLANDI")
    print(f"  Toplam Sevkiyat  : {len(result_df)}")
    print(f"  Unique Talep ID  : {unique_talep_ids}")
    print(f"  Araç Maliyeti    : {round(tc,2)} TL")
    print(f"  SLA Cezası       : {round(tsla,2)} TL")
    print(f"  SLA İhlal Sayısı : {sla_count}")
    print(f"  Toplam Maliyet   : {round(tc+tsla,2)} TL")
    print(f"  Ort. Doluluk     : {round(au,2)}")
    print(f"  Tır Kap. İhlal   : {tir_violations}")

    return result_df