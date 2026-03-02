import pandas as pd
from pandera.pandas import DataFrameSchema, Column
import pytest

@pytest.fixture
def datos_banco():
    """Fixture para cargar los datos del banco desde un archivp csv.
    Returns:
        pd.DataFrame: DataFrame que contiene los datos del banco.
    """
    return pd.read_csv("data/raw/bank-additional-full.csv", sep=";")

def test_esquema(datos_banco):
    """Test de esquema para el DataFrame de datos_banco.

    Args:
        datos_banco (pd.DataFrame): DataFrame que contiene los datos del banco.
    """
    df = datos_banco

    esquema = DataFrameSchema({
        "age": Column(int, nullable=False),
        "job": Column(str, nullable=False),
        "marital": Column(str, nullable=False),
        "default": Column(str, nullable=False),
        "housing": Column(str, nullable=False),
        "loan": Column(str, nullable=False),
        "contact": Column(str, nullable=False),
        "month": Column(str, nullable=False),
        "day_of_week": Column(str, nullable=False),
        "duration": Column(int, nullable=False),
        "campaign": Column(int, nullable=False),
        "pdays": Column(int, nullable=False),
        "previous": Column(int, nullable=False),
        "poutcome": Column(str, nullable=False),
        "emp.var.rate": Column(float, nullable=False),
        "cons.price.idx": Column(float, nullable=False),
        "cons.conf.idx": Column(float, nullable=False),
        "euribor3m": Column(float, nullable=False),
        "nr.employed": Column(float, nullable=False),
        "y": Column(str, nullable=False),
    })

    esquema.validate(df)

def test_basico(datos_banco):
    """Test basico para verificar que el DataFrame de datos_banco no está vacío 
    y contiene las columnas esperadas.

    Args:
        datos_banco (pd.DataFrame): DataFrame que contiene los datos del banco.
    """
    df = datos_banco

    # Verificar que el DataFra,e mp está vacio
    assert not df.empty, "El DataFrame está vacío."
    # Verificar nulos
    assert df.isnull().sum().sum() == 0, "El DataFrame contiene valores nulos."
    # Verificar duplicados
    #assert df.duplicated().sum() == 0, "El DataFrame contiene filas duplicados."
    # Verificar cantidad de columnas
    assert df.shape[1] == 21, "El DataFrame debería tener 21 columnas, pero tiene {df.shape[1]}."

def test_dominios_y_rangos(datos_banco):
    """Valida dominios de variables categóricas y rangos básicos de variables numéricas."""
    df = datos_banco

    assert df["y"].isin(["yes", "no"]).all(), "La columna 'y' contiene valores no válidos."
    assert df["age"].between(17, 100).all(), "La columna 'age' está fuera del rango esperado (18-100)."
    assert (df["duration"] >= 0).all(), "La columna 'duration' contiene valores negativos."
    assert df["month"].isin(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]).all(), "La columna 'month' contiene valores no válidos."