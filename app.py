# Import python libraries
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from flask import Flask
from constants import TOP_N_RECOMMENDATIONS
import model as model
from data_model import *

# Init fastAPI
app = FastAPI()
# Init flask APP
flask_app = Flask(__name__)

def apply_model(username, limit):
	return model.product_recommendations(username, limit)

# API Predict recommendations	
@app.post("/api/predict", response_model=OutputDataModel)
async def api_post_predictions(inputDataModel: InputDataModel):
    ''' Default limit is 5 if provided Zero or Negative'''
    if (inputDataModel.limit <= 0):
        recommendations = apply_model(inputDataModel.username, TOP_N_RECOMMENDATIONS)
    else:
        recommendations = apply_model(inputDataModel.username, inputDataModel.limit)  
    if (recommendations is not None):
        recommendations = list(recommendations.values.tolist())
        response = {"message": f"Top {len(recommendations)} recommendations for User [{inputDataModel.username}].",
                    "recommendations": recommendations}
    else:
        response = {"message": f"User [{inputDataModel.username}] not found. Please try again!"}
    return response

# API List Users
@app.get("/api/users", response_model=UserDataModel)
async def api_get_users():
    return {"users": model.get_users()}

# FastAPI Templating

# Static files mount
app.mount("/static", StaticFiles(directory='templates/css'), name="static")
# App templates for web pages
deafult_templates = Jinja2Templates(directory='templates')
app_templates = Jinja2Templates(directory='templates/app/')

# Loading Index Html
@app.get('/')
async def index_html(request:Request):
    return deafult_templates.TemplateResponse("index.html", {'request': request})

# Loading Recommendation page
@app.get('/app', response_class=HTMLResponse)
async def index_html(request:Request):
	return app_templates.TemplateResponse('recommendations.html', {'request': request}) 

# Recommendation page with predictions
@app.get("/app/ml_predict", response_class=HTMLResponse)
async def app_post_predictions(request: Request, username: str):
    username=username.strip().lower()
    if (username != ''):
        recommendations = apply_model(username, TOP_N_RECOMMENDATIONS)
        if (recommendations is not None):
            column_names=recommendations.columns.values
            recommendations = list(recommendations.values.tolist())
            return app_templates.TemplateResponse('recommendations.html', {'request': request, 'recommendations': recommendations, 'column_names': column_names, 'user_input': username, 'zip': zip})
        else:
            message=f"User [{username}] not found. Please try again!"
    else:
        message="Username shouldn't be Empty, Whitespaces!"
    return app_templates.TemplateResponse('recommendations.html', {'request': request, 'err_msg': message})
