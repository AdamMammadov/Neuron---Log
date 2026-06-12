from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_PATH = BASE_DIR / "outputs"

OUTPUT_PATH.mkdir(exist_ok=True)


def export_plan(df):

    output_file = OUTPUT_PATH / "Arac_Planlama.xlsx"

    # Tarih sütununu datetime formatına çevir — Excel serial nömrəsi görünməsin
    df = df.copy()
    if "Tarih" in df.columns:
        df["Tarih"] = pd.to_datetime(df["Tarih"]).dt.strftime("%Y-%m-%d")

    df.to_excel(output_file, index=False)

    print(f"\nPLAN EXPORTED: {output_file}")