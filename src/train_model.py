""" 
Script para entrenar un modelo de clasificación utilizando la técnica con mejor rendimiento 
que fuera seleccionada durante la experimentación.
"""
# Importaciones generales
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
import json
# Importaciones para el preprocesamiento y modelado
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler 
from sklearn.utils import resample
# Importaciones para la evaluación - experimentación
from sklearn.model_selection import train_test_split

# from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    f1_score, recall_score, precision_score, 
    accuracy_score, confusion_matrix
)
from mlflow.models import infer_signature
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold


def load_data(path):
    """Función para cargar los datos desde un archivo CSV."""
    df = pd.read_csv(path)
    X = df.drop('y', axis=1)
    y = df['y']
    return train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


def create_preprocessor(X_train):
    numerical_columns = X_train.select_dtypes(exclude='object').columns
    categorical_columns = X_train.select_dtypes(include='object').columns

    X_train = X_train.copy()
    int_columns = X_train.select_dtypes(include='int').columns
    for col in int_columns:
        X_train[col] = X_train[col].astype('float')
    
    # Actualizar numerical_cols
    numerical_columns = X_train.select_dtypes(exclude='object').columns

    # Pipeline para valores numéricos
    num_pipeline = Pipeline(steps=[
        ('RobustScaler', RobustScaler())
    ])

    # Pipeline para valores categóricos
    cat_pipeline = Pipeline(steps=[
        ('OneHotEncoder', OneHotEncoder(drop='first',sparse_output=False))
    ])

    # Se configuran los preprocesadores
    preprocessor_full = ColumnTransformer([
        ('num_pipeline', num_pipeline, numerical_columns),
        ('cat_pipeline', cat_pipeline, categorical_columns)
    ]).set_output(transform='pandas')

    return preprocessor_full, X_train

def balance_data(X, y, random_state=42):
    # Combinar los datos preprocesados con las etiquetas
    train_data = X.copy()
    train_data['target'] = y.reset_index(drop=True)

    # Separar por clase
    class_0 = train_data[train_data['target'] == 0]
    class_1 = train_data[train_data['target'] == 1]

    # Encontrar la clase minoritaria
    min_count = min(len(class_0), len(class_1))

    # Submuestreo balanceado - tomar una muestra igual al tamaño de la clase minoritaria
    class_0_balanced = resample(class_0, n_samples=min_count, random_state=random_state)
    class_1_balanced = resample(class_1, n_samples=min_count, random_state=random_state)

    # Combinar las clases balanceadas
    balanced_data = pd.concat([class_0_balanced, class_1_balanced])

    # Separar características y objetivo
    x_train_resampled = balanced_data.drop('target', axis=1)
    y_train_resampled = balanced_data['target']

    return x_train_resampled, y_train_resampled

def compute_cv_metrics(X_train_raw, y_train, random_state=42, cv_folds=5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    f1s, precs, recs, accs = [], [], [], []

    for tr_idx, val_idx in skf.split(X_train_raw, y_train):
        X_tr = X_train_raw.iloc[tr_idx].copy()
        X_val = X_train_raw.iloc[val_idx].copy()
        y_tr = y_train.iloc[tr_idx].copy()
        y_val = y_train.iloc[val_idx].copy()

        # Preprocesador fit SOLO con el fold train
        preprocessor, X_tr_conv = create_preprocessor(X_tr)

        # Convertir ints en X_val igual que haces en test
        X_val = X_val.copy()
        int_cols = X_val.select_dtypes(include=['int64', 'int32', 'int']).columns
        for col in int_cols:
            X_val[col] = X_val[col].astype('float64')

        X_tr_p = preprocessor.fit_transform(X_tr_conv)
        X_val_p = preprocessor.transform(X_val)

        # Balanceo SOLO en el fold train
        X_tr_b, y_tr_b = balance_data(X_tr_p, y_tr, random_state=random_state)

        model = LinearSVC(random_state=random_state)
        model.fit(X_tr_b, y_tr_b)

        y_hat = model.predict(X_val_p)

        f1s.append(f1_score(y_val, y_hat))
        precs.append(precision_score(y_val, y_hat))
        recs.append(recall_score(y_val, y_hat))
        accs.append(accuracy_score(y_val, y_hat))

    return {
        "cv_folds": cv_folds,
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs, ddof=1)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s, ddof=1)),
        "cv_precision_mean": float(np.mean(precs)),
        "cv_precision_std": float(np.std(precs, ddof=1)),
        "cv_recall_mean": float(np.mean(recs)),
        "cv_recall_std": float(np.std(recs, ddof=1)),
    }


def train_model(
    data_path: str = 'data/processed/data.csv',
    model_output_path: str = 'models/LinearSVC',
    preprocessor_output_path: str = 'models/preprocessor.pkl',
    metrics_output_path: str = 'models/metrics.json'
):
    """ Método principal para entrenar el modelo de clasificación. """
    # Configuración de MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("13MBID - Uxue - Proyecto - Producción")

    with mlflow.start_run(run_name="LinearSVC_Production"):
        print("Cargando datos...")
        X_train, X_test, y_train, y_test = load_data(data_path)
        
        print("Creando preprocesador...")
        preprocessor, X_train_converted = create_preprocessor(X_train)
        X_test = X_test.copy()
        
        # Convertir columnas enteras en X_test también
        int_columns = X_test.select_dtypes(include=['int64', 'int32']).columns
        for col in int_columns:
            X_test[col] = X_test[col].astype('float64')
        
        print("Preprocesando datos...")
        X_train_prep = preprocessor.fit_transform(X_train_converted)
        X_test_prep = preprocessor.transform(X_test)
            
        print("Balanceando datos...")
        X_train_balanced, y_train_balanced = balance_data(X_train_prep, y_train)
        
        print(f"  Tamaño original: {len(X_train_prep)}")
        print(f"  Tamaño balanceado: {len(X_train_balanced)}")
        print(f"  Distribución: {y_train_balanced.value_counts().to_dict()}")

        print("\nEntrenando modelo LinearSVC...")
        model = LinearSVC(random_state=42)
        model.fit(X_train_balanced, y_train_balanced)
        
        print("Evaluando modelo...")
        y_pred = model.predict(X_test_prep)
        
        # Crear pipeline completo
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Crear signatures y ejemplos de entrada
        # raw_input_example = X_train.iloc[:5]
        # preprocessed_input_example = X_train_prep.iloc[:5]
        
        # Signature para el pipeline completo
        pipeline_signature = infer_signature(
            X_train,  # Datos de entrada sin procesar
            y_pred    # Predicciones del modelo
        )
        
        # Signature para el preprocesador
        preprocessor_signature = infer_signature(
            X_train,      # Datos de entrada sin procesar
            X_train_prep  # Datos procesados
        )
        
        # Signature para el modelo
        model_signature = infer_signature(
            X_train_prep,  # Datos procesados
            y_pred         # Predicciones
        )

        # Calcular métricas
        metrics = {
            "f1_score": float(f1_score(y_test, y_pred)),
            "recall_score": float(recall_score(y_test, y_pred)),
            "precision_score": float(precision_score(y_test, y_pred)),
            "accuracy_score": float(accuracy_score(y_test, y_pred))
        }
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Registrar parámetros
        mlflow.log_params({
            "model_type": "LinearSVC",
            "balancing_method": "undersampling",
            "train_samples": int(len(X_train_balanced)),
            "test_samples": int(len(X_test)),
            "random_state": 42,
            # Parámetros propios de LinearSVC
            "C": float(model.C),
            "penalty": str(model.penalty),
            "loss": str(model.loss),
            "dual": str(model.dual),
            "tol": float(model.tol),
            "max_iter": int(model.max_iter),
            "fit_intercept": bool(model.fit_intercept),
            "class_weight": str(model.class_weight),
        })
        
        # Registrar métricas
        mlflow.log_metrics(metrics)

        # Registrar matriz de confusión      
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes']).plot(ax=ax)
        plt.title('Confusion Matrix - Production Model')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

        # Registrar pipeline completo
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            signature=pipeline_signature,
            # input_example=raw_input_example
        )
        
        # Registrar preprocesador
        mlflow.sklearn.log_model(
            sk_model=preprocessor,
            artifact_path="preprocessor",
            signature=preprocessor_signature,
            # input_example=raw_input_example
        )
        
        # Registrar modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="classifier",
            signature=model_signature,
            # input_example=preprocessed_input_example
        )

        # Guardar modelos localmente
        print("\nGuardando modelos...")
        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_output_path)
        joblib.dump(preprocessor, preprocessor_output_path)
        
        # Guardar métricas
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return model, preprocessor, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo de producción")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/bank-processed.csv",
        help="Ruta al archivo de datos procesados"
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="models/LinearSVC_model.pkl",
        help="Ruta donde guardar el modelo"
    )
    parser.add_argument(
        "--preprocessor-output",
        type=str,
        default="models/preprocessor.pkl",
        help="Ruta donde guardar el preprocesador"
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="metrics/metrics.json",
        help="Ruta donde guardar las métricas"
    )
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        model_output_path=args.model_output,
        preprocessor_output_path=args.preprocessor_output,
        metrics_output_path=args.metrics_output
    )