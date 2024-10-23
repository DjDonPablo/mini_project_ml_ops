import joblib
import uvicorn

from fastapi import FastAPI

model = joblib.load("regression.joblib")

app = FastAPI()

@app.get("/predict")
async def predict(size: int, nbrooms: int, garden: int):
    return {"prediction": model.predict([[size, nbrooms, garden]])[0]}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=6942, log_level="info")
