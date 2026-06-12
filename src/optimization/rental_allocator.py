def find_rental_option(
    origin,
    destination,
    rental_df
):
    """
    Check rental availability
    """

    filtered = rental_df[
        (rental_df["Çıkış Transfer Merkezi"] == origin)
        &
        (rental_df["Varış Transfer Merkezi"] == destination)
    ]

    if len(filtered) == 0:
        return None

    return filtered.iloc[0]