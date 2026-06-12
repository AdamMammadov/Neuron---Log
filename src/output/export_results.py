from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_PATH = BASE_DIR / "outputs"

OUTPUT_PATH.mkdir(exist_ok=True)


def export_forecasts(df):
    """
    Export forecast results
    """

    output_file = OUTPUT_PATH / "Tahminlenen_Talep.xlsx"

    # Tarih sütununu datetime formatına çevir — Excel serial nömrəsi görünməsin
    df = df.copy()
    if "Tarih" in df.columns:
        df["Tarih"] = pd.to_datetime(df["Tarih"]).dt.strftime("%Y-%m-%d")

    # Export Excel
    df.to_excel(output_file, index=False)

    print(f"\nForecast exported: {output_file}")

    # Extra validation info
    print("\nFORECAST EXPORT SUMMARY")
    print(f"Total Rows: {len(df)}")

    if "Tahminlenen Desi" in df.columns:

        print(
            f"Total Forecasted Desi: "
            f"{round(df['Tahminlenen Desi'].sum(), 2)}"
        )

        print(
            f"Average Forecasted Desi: "
            f"{round(df['Tahminlenen Desi'].mean(), 2)}"
        )

        print(
            f"Max Forecasted Desi: "
            f"{round(df['Tahminlenen Desi'].max(), 2)}"
        )

        print(
            f"Min Forecasted Desi: "
            f"{round(df['Tahminlenen Desi'].min(), 2)}"
        )

    print("\nEXPORT COMPLETED SUCCESSFULLY.")