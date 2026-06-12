import pandas as pd

from src.utils.distance_calculator import haversine


def build_distance_matrix(coordinates_df):

    distances = []

    for _, row1 in coordinates_df.iterrows():

        for _, row2 in coordinates_df.iterrows():

            origin = row1["Transfer Merkezi"]
            destination = row2["Transfer Merkezi"]

            if origin == destination:
                continue

            distance = haversine(
                row1["Enlem"],
                row1["Boylam"],
                row2["Enlem"],
                row2["Boylam"],
            )

            distances.append({
                "origin": origin,
                "destination": destination,
                "distance_km": round(distance, 2)
            })

    distance_df = pd.DataFrame(distances)

    print("\nDistance matrix created.")

    return distance_df