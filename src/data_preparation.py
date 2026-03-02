import pandas as pd
import numpy as np

INPUT_CSV = "data/raw/bank-additional-full.csv"
OUTPUT_CSV = "data/processed/bank-processed.csv"


def preprocess_data(input_path=INPUT_CSV, output_path=OUTPUT_CSV):
    # Load the dataset
    df = pd.read_csv(input_path, sep=";")

    # Adaptar nombres de columnas (evitar puntos en nombres)
    df.columns = df.columns.str.replace(".", "_", regex=False)

    # Transformar los valores 'unknown' en NaN (valores faltantes)
    df.replace("unknown", np.nan, inplace=True)

    # Se agrega el campo contacted_before (binario: yes/no) en función de pdays
    # En este dataset, pdays=999 suele indicar que no hubo contacto previo
    df["contacted_before"] = np.where(df["pdays"] == 999, "no", "yes")

    # (1) Mantener información de pdays creando pdays_clean
    # Convertimos 999 -> NaN para indicar "no aplica / no contactado antes"
    df["pdays_clean"] = df["pdays"].replace(999, np.nan)

    # (2) Transformación cíclica de month -> month_sin, month_cos
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    df["month_num"] = df["month"].map(month_map)

    # Si hubiera meses inválidos quedarán como NaN y se eliminarán en dropna()
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    # Se elimina la columna 'default' ya que tiene muchos valores desconocidos
    df.drop(columns=["default"], inplace=True)

    # Ya tenemos la señal de pdays en contacted_before y pdays_clean
    # Eliminamos pdays original y month_num auxiliar
    df.drop(columns=["pdays", "month_num"], inplace=True)

    # Eliminación de filas con valores nulos (por ejemplo, unknowns convertidos a NaN)
    df.dropna(inplace=True)

    # Eliminación de filas duplicadas
    df.drop_duplicates(inplace=True)

    # Mapear la columna objetivo 'y' a valores binarios
    y_map = {"yes": 1, "no": 0}
    df["y"] = df["y"].map(y_map)

    # Guardar dataset procesado
    df.to_csv(output_path, index=False)

    return df.shape


if __name__ == "__main__":
    dimensiones = preprocess_data()

    with open("docs/transformations.txt", "w", encoding="utf-8") as f:
        f.write("Transformaciones realizadas:\n")
        f.write("- Se reemplazaron los valores 'unknown' por NaN\n")
        f.write("- Se creó 'contacted_before' a partir de 'pdays'\n")
        f.write("- Se creó 'pdays_clean' reemplazando 999 por NaN para conservar información numérica\n")
        f.write("- Se transformó 'month' a representación cíclica (month_sin, month_cos)\n")
        f.write("- Se eliminaron filas con valores nulos\n")
        f.write("- Se eliminaron filas duplicadas\n")
        f.write("- Se eliminó la columna 'default' por alta proporción de valores desconocidos\n")
        f.write(f"- Cantidad de filas finales: {dimensiones[0]}\n")
        f.write(f"- Cantidad de columnas finales: {dimensiones[1]}\n")