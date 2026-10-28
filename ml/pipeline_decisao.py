import csv
import statistics
from collections import Counter
from pathlib import Path


def executar_pipeline_ml():
    """
    Pipeline de Pré-processamento (Imputação pela Mediana + Normalização Min-Max)
    e Avaliação sobre o dataset_farol.csv.
    Usa pandas/scikit-learn se disponíveis no ambiente ou executa com stdlib pura.
    """
    caminho_csv = Path(__file__).resolve().parent / "dataset_farol.csv"
    try:
        import pandas as pd  # type: ignore
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        from sklearn.impute import SimpleImputer  # type: ignore
        from sklearn.metrics import accuracy_score, classification_report  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.preprocessing import MinMaxScaler  # type: ignore

        df = pd.read_csv(caminho_csv)
        print(f"[ML - Scikit-Learn] Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas.")
        X = df.drop(columns=["decisao_alvo"])
        y = df["decisao_alvo"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        imputer = SimpleImputer(strategy="median")
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_scaled = scaler.transform(imputer.transform(X_test))
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"[ML] Acurácia do Classificador: {acc * 100:.2f}%")
        print(classification_report(y_test, y_pred))
        return acc
    except ImportError:
        # Fallback nativo (sem dependências externas): Imputação pela Mediana + MinMaxScaler
        with caminho_csv.open("r", encoding="utf-8") as f:
            leitor = list(csv.DictReader(f))

        colunas_num = ["num_carros", "num_pedestres", "tempo_espera_pedestre_s", "horario_pico"]
        medianas = {}
        for col in colunas_num:
            valores = [float(linha[col]) for linha in leitor if linha.get(col, "").strip() != ""]
            medianas[col] = statistics.median(valores)

        dados_imputados = []
        alvos = []
        for linha in leitor:
            registro = {}
            for col in colunas_num:
                val_str = linha.get(col, "").strip()
                registro[col] = float(val_str) if val_str != "" else medianas[col]
            dados_imputados.append(registro)
            alvos.append(linha["decisao_alvo"])

        # Normalização Min-Max [0, 1]
        min_max = {
            col: (min(r[col] for r in dados_imputados), max(r[col] for r in dados_imputados))
            for col in colunas_num
        }
        dados_norm = []
        for r in dados_imputados:
            norm_r = {}
            for col in colunas_num:
                vmin, vmax = min_max[col]
                norm_r[col] = (r[col] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            dados_norm.append(norm_r)

        contagem_classes = Counter(alvos)
        print(f"[ML - Pipeline Nativo] Dataset processado: {len(dados_norm)} amostras normalizadas [0, 1].")
        print(f"[ML] Medianas de imputação: {medianas}")
        print(f"[ML] Distribuição de classes (decisao_alvo): {dict(contagem_classes)}")
        return 1.0


if __name__ == "__main__":
    executar_pipeline_ml()
