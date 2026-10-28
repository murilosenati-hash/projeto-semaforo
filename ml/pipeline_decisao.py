import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def executar_pipeline_ml():
    """
    Pipeline de Pré-processamento e Treinamento Supervisionado sobre o dataset_farol.csv.
    Integra o módulo de Ciência de Dados / Aprendizado de Máquina ao projeto do Semáforo.
    """
    caminho_csv = Path(__file__).resolve().parent / "dataset_farol.csv"
    df = pd.read_csv(caminho_csv)
    print(f"[ML] Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas.")

    X = df.drop(columns=["decisao_alvo"])
    y = df["decisao_alvo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"[ML] Acurácia do Classificador de Decisão Semafórica: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    return acc


if __name__ == "__main__":
    executar_pipeline_ml()
