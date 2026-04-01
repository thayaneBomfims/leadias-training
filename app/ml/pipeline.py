import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LeadModel:

    def __init__(self):
        self.embedding_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
    def load_data(self, path):
        df = pd.read_csv(path)
        return df

    def prepare_features(self, df):
        textos = df["texto"].tolist()

        embeddings = self.embedding_model.encode(textos)

        extra_features = df[[
            "valor_veiculo",
            "tempo_primeiro_contato_min",
            "qtd_interacoes"
        ]].values

        X = np.hstack([embeddings, extra_features])
        return X

    def prepare_labels(self, df):
        y = df[[
            "preco",
            "financiamento",
            "concorrencia",
            "atendimento",
            "demora",
            "desinteresse"
        ]]
        return y

    def train(self, df):
        X = self.prepare_features(df)
        y = self.prepare_labels(df)

        # 🔥 ESCALA OS DADOS
        X = self.scaler.fit_transform(X)

        self.models = {}

        for col in y.columns:
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X, y[col])
            self.models[col] = clf

    def predict(self, texto, valor, tempo, interacoes):
        embedding = self.embedding_model.encode([texto])
        extra = np.array([[valor, tempo, interacoes]])

        X = np.hstack([embedding, extra])

        # 🔥 aplicar mesma escala
        X = self.scaler.transform(X)

        results = {}

        for name, model in self.models.items():
            prob = model.predict_proba(X)[0][1]
            results[name] = float(prob)

        return results