from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import uvicorn

application = FastAPI(title="Cafeteria Occupancy Predictor")

application.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- LOAD MODEL ONCE -----------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

try:
    with open("finalModel.pkl", "rb") as file:
        trained_model = joblib.load(file)
except Exception as e:
    trained_model = None
    print(f"MODEL LOAD ERROR: {repr(e)}")


# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- ML PREDICTION -------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

def preprocessDataAndPredict(curr_day, curr_time, curr_occ, occ_24, occ_week, occ_month):

    # Ensure clean numeric input
    curr_day = float(curr_day)
    curr_time = float(curr_time)
    curr_occ = float(curr_occ)
    occ_24 = float(occ_24)
    occ_week = float(occ_week)
    occ_month = float(occ_month)

    data = pd.DataFrame({
        'day': [curr_day],
        'curr_time': [curr_time],
        'curr_occ': [curr_occ],
        'occ_24': [occ_24],
        'occ_week': [occ_week],
        'occ_month': [occ_month]
    })

    if trained_model is None:
        raise ValueError("Model is not loaded properly.")

    prediction = trained_model.predict(data)

    return round(prediction[0], 0)


# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- ROUTES --------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.get("/", response_class=HTMLResponse)
@application.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@application.get("/cafeOccupancyPredictor", response_class=HTMLResponse)
async def cafeOccupancyPredictor(request: Request):
    return templates.TemplateResponse("cafeOccupancyPredictor.html", {"request": request})


@application.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request):
    return templates.TemplateResponse("cafeOccupancyPredictor.html", {"request": request})


@application.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    curr_day: float = Form(...),
    curr_time: float = Form(...),
    curr_occ: float = Form(...),
    occ_24: float = Form(...),
    occ_week: float = Form(...),
    occ_month: float = Form(...)
):

    try:
        prediction = preprocessDataAndPredict(
            curr_day, curr_time, curr_occ,
            occ_24, occ_week, occ_month
        )

        return templates.TemplateResponse(
            "predict.html",
            {"request": request, "prediction": prediction}
        )

    except Exception as e:
        # SHOW REAL ERROR (this is the important fix)
        return HTMLResponse(f"ERROR: {repr(e)}", status_code=500)


# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- RUN APP -------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    uvicorn.run("application:application", host="localhost", port=5000, reload=True)