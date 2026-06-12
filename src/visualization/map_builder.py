import pandas as pd


def build_route_map_data(plan_df):

    city_coordinates = {
        "İstanbul": [41.0082, 28.9784],
        "Ankara": [39.9334, 32.8597],
        "İzmir": [38.4237, 27.1428],
        "Bursa": [40.1885, 29.0610],
        "Kocaeli": [40.8533, 29.8815],
        "Tekirdağ": [40.9781, 27.5117],
        "Balıkesir": [39.6484, 27.8826],
        "Manisa": [38.6191, 27.4289],
        "Eskişehir": [39.7667, 30.5256],
        "Kütahya": [39.4242, 29.9833],
        "Isparta": [37.7648, 30.5566],
        "Mersin": [36.8121, 34.6415],
        "Sivas": [39.7477, 37.0179],
        "Bilecik": [40.1500, 29.9833]
    }

    map_rows = []

    for _, row in plan_df.iterrows():

        origin = row["Çıkış Transfer Merkezi"]

        destination = row["Varış Transfer Merkezi"]

        if (
            origin not in city_coordinates
            or
            destination not in city_coordinates
        ):
            continue

        origin_lat, origin_lon = city_coordinates[origin]

        dest_lat, dest_lon = city_coordinates[destination]

        risk = row["Risk Level"]

        if risk == "HIGH":
            color = [255, 0, 0]

        elif risk == "MEDIUM":
            color = [255, 165, 0]

        else:
            color = [0, 200, 0]

        map_rows.append({
            "origin": origin,
            "destination": destination,

            "from_lat": origin_lat,
            "from_lon": origin_lon,

            "to_lat": dest_lat,
            "to_lon": dest_lon,

            "risk": risk,

            "desi": row["Taşınan Desi"],

            "color": color
        })

    return pd.DataFrame(map_rows)