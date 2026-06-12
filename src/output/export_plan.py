from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_PATH = BASE_DIR / "outputs"

OUTPUT_PATH.mkdir(exist_ok=True)


def export_plan(df):

    output_file = OUTPUT_PATH / "Arac_Planlama.xlsx"

    df.to_excel(output_file, index=False)

    print(f"\nPLAN EXPORTED: {output_file}")