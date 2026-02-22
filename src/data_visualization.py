# Importación de librerías y supresión de advertencias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ydata_profiling import ProfileReport

def visualizar_datos(
    fuente: str = "data/raw/bank-additional-full.csv",
    salida: str = "docs/figures/"
):

    # Crear el directorio de salida si no existe
    ruta_salida = Path(salida)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    # Leer los datos
    df = pd.read_csv(fuente, sep=';')

    # Gráfico 1: Distribución de la variable objetivo
    plt.figure(figsize=(8,6))
    sns.countplot(x="y", data=df)
    plt.title("Distribución de la variable objetivo (suscripción al depósito)")
    plt.xlabel("¿Suscribió un depósito a plazo?")
    plt.ylabel("Cantidad de clientes")
    plt.savefig(ruta_salida / "Distribución_target.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("PWD:", Path.cwd())
    print("Guardando en:", (Path(salida) / "grafico_1.png").resolve())

    # Gráfico 2: Distribución del nivel educativo

    plt.figure(figsize=(8,6))
    col = 'education'
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(ruta_salida / "Distribución_educación.png", dpi=150, bbox_inches="tight")
    plt.close()

    ### agregar dos gráficos más

 # =========================================================
    # Gráfico 3: Distribución de edad (histograma)
    # =========================================================
    plt.figure(figsize=(8, 6))
    sns.histplot(df["age"], bins=30, kde=True)
    plt.title("Distribucion de la edad")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    out3 = ruta_salida / "distribucion_edad.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================
    # Gráfico 4: Matriz de correlaciones (numéricas)
    # =========================================================
    num_df = df.select_dtypes(include=["int64", "float64"])
    if not num_df.empty:
        corr = num_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matriz de correlaciones (variables numericas)")
        out4 = ruta_salida / "correlaciones_numericas.png"
        plt.savefig(out4, dpi=150, bbox_inches="tight")
        plt.close()

    # =========================================================
    # Gráfico 5: Tasa de suscripción por nivel educativo
    # =========================================================
    edu_target = (
        df.groupby("education")["y"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

    edu_yes = edu_target[edu_target["y"] == "yes"]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=edu_yes,
        x="proportion",
        y="education",
        order=edu_yes.sort_values("proportion", ascending=False)["education"]
    )
    plt.title("Tasa de suscripción por nivel educativo")
    plt.xlabel("Proporción de clientes que suscriben")
    plt.ylabel("Nivel educativo")
    plt.xlim(0, 1)
    plt.savefig(ruta_salida / "tasa_suscripcion_education.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================
    # Gráfico 6: Suscripción por mes
    # =========================================================
    month_target = (
        df.groupby("month")["y"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

    month_yes = month_target[month_target["y"] == "yes"]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=month_yes,
        x="month",
        y="proportion",
        order=month_yes.sort_values("proportion", ascending=False)["month"]
    )
    plt.title("Tasa de suscripción por mes de campaña")
    plt.xlabel("Mes")
    plt.ylabel("Proporción de suscripción")
    plt.ylim(0, month_yes["proportion"].max() * 1.1)
    plt.savefig(ruta_salida / "tasa_suscripcion_mes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================
    # Reporte automático del dataset
    # =========================================================
    profile = ProfileReport(df, title="Reporte del dataset", minimal=True)
    profile.to_file(ruta_salida / "reporte_dataset.html")



if __name__== "__main__":
    visualizar_datos()




