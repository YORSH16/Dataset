import pandas as pd
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import joblib



app = FastAPI(
    title="Deploy dataset",
    version="0.0.1"
    )


#--------------------------------------------------------------------------
#LOAD MODEL
#--------------------------------------------------------------------------


model = joblib.load("models/logistic_regression_model_v01.pkl")

@app.post("/api/v1/predict-dataset", tags=["predict-data"])
async def predict(
        Size: float,
        Weight: float,
        Sweetness: float,
        Crunchiness: float,
        Juiciness: float,
        Ripeness: float,
        Acidity: float
):

    dictionary = {
        'Size': Size,
        'Weight': Weight,
        'Sweetness': Sweetness,
        'Crunchiness': Crunchiness,
        'Juiciness': Juiciness,
        'Ripeness': Ripeness,
        'Acidity': Acidity
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)
        return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=prediction[0]
    )

    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
    )


if __name__ == "__main__":
    app.run()



#%%
