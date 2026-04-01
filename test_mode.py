from app.ml.pipeline import LeadModel

model = LeadModel()

df = model.load_data("data/leads_dataset_completo.csv")

model.train(df)

resultado = model.predict(
    texto="cliente achou caro e foi ver outra loja",
    valor=75000,
    tempo=120,
    interacoes=2
)

print(resultado)