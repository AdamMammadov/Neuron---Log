from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw"


def load_excel_file(file_name):
    """
    Generic Excel loader
    """

    file_path = RAW_DATA_PATH / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_excel(file_path)

    print(f"Loaded: {file_name}")
    print(f"Shape: {df.shape}")

    return df


def load_all_data():
    """
    Load all project datasets
    """

    demand_df = load_excel_file("Desi_talep.xlsx")

    coordinates_df = load_excel_file("Koordinatlar v2.xlsx")

    rental_df = load_excel_file("Kiralik_Araclar.xlsx")

    vehicle_df = load_excel_file("Arac_Kapasite_Maliyet.xlsx")

    return {
        "desi_talep": demand_df,
        "koordinatlar": coordinates_df,
        "kiralik_araclar": rental_df,
        "arac_kapasite": vehicle_df,
    }


if __name__ == "__main__":

    data = load_all_data()

    print("\nDATA SUCCESSFULLY LOADED")