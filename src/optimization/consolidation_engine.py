import pandas as pd


def apply_consolidation(
    forecast_df,
    utilization_target=0.85,
    max_wait_days=1
):
    """
    AI Load Consolidation Engine

    Small loads are intelligently merged with
    future shipments to improve fleet utilization.

    Existing logic is preserved.
    """

    df = forecast_df.copy()

    df = df.sort_values(
        [
            "Çıkış Transfer Merkezi",
            "Varış Transfer Merkezi",
            "Tarih"
        ]
    )

    consolidated_rows = []

    processed = set()

    for idx, row in df.iterrows():

        if idx in processed:
            continue

        current_load = row[
            "Tahminlenen Desi"
        ]

        current_date = row[
            "Tarih"
        ]

        origin = row[
            "Çıkış Transfer Merkezi"
        ]

        destination = row[
            "Varış Transfer Merkezi"
        ]

        combined_load = current_load

        # Dynamic threshold
        min_target_load = (
            8000 * utilization_target
        )

        if current_load < min_target_load:

            next_rows = df[
                (
                    df[
                        "Çıkış Transfer Merkezi"
                    ]
                    ==
                    origin
                )
                &
                (
                    df[
                        "Varış Transfer Merkezi"
                    ]
                    ==
                    destination
                )
                &
                (
                    df["Tarih"]
                    >
                    current_date
                )
            ].head(
                max_wait_days
            )

            for next_idx, next_row in next_rows.iterrows():

                if next_idx in processed:
                    continue

                combined_load += next_row[
                    "Tahminlenen Desi"
                ]

                processed.add(
                    next_idx
                )

                if (
                    combined_load
                    >=
                    min_target_load
                ):
                    break

        new_row = row.copy()

        new_row[
            "Tahminlenen Desi"
        ] = round(
            combined_load,
            2
        )

        consolidated_rows.append(
            new_row
        )

    result = pd.DataFrame(
        consolidated_rows
    )

    original_shipments = len(df)

    consolidated_shipments = len(
        result
    )

    saved_shipments = (
        original_shipments
        -
        consolidated_shipments
    )

    consolidation_rate = round(
        (
            saved_shipments
            /
            max(original_shipments, 1)
        )
        *
        100,
        2
    )

    print(
        "\nAI LOAD CONSOLIDATION COMPLETED"
    )

    print(
        f"Original Shipments: "
        f"{original_shipments}"
    )

    print(
        f"Consolidated Shipments: "
        f"{consolidated_shipments}"
    )

    print(
        f"Saved Trips: "
        f"{saved_shipments}"
    )

    print(
        f"Consolidation Rate: "
        f"{consolidation_rate}%"
    )

    return result


# ====================================================
# COMPATIBILITY LAYER
# ====================================================
# main.py imports this function:
#
# from consolidation_engine import
# apply_shipment_consolidation
#
# Therefore we expose a wrapper without
# changing the original implementation.
# ====================================================

def apply_shipment_consolidation(
    forecast_df,
    utilization_target=0.85,
    max_wait_days=1
):
    """
    Wrapper for backward compatibility.

    Existing imports continue to work
    without modifying main.py.
    """

    return apply_consolidation(
        forecast_df=forecast_df,
        utilization_target=utilization_target,
        max_wait_days=max_wait_days
    )