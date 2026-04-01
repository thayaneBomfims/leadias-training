from fastapi import APIRouter
from app.ml.pipeline import LeadModel

router = APIRouter()

model = LeadModel()
df = model.load_data("data/leads_dataset_completo.csv")
model.train(df)

@router.post("/predict")
def predict(data: dict):

    resultado = model.predict(
        texto=data["texto"],
        valor=data["valor"],
        tempo=data["tempo"],
        interacoes=data["interacoes"]
    )

    return resultado