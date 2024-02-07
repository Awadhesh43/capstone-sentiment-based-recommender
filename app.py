# Import python libraries
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from flask import Flask, render_template
from constants import TOP_N_RECOMMENDATIONS
import model as model
from data_model import *

# Init fastAPI
app = FastAPI()

# Init flask APP
flask_app = Flask(__name__)

# Mount Flask App on FastAPI
#app.mount("/app", WSGIMiddleware(flask_app))

def apply_model(userId, limit):
	return model.product_recommendations(userId, limit)

# API Landing page
@app.get('/api', include_in_schema=False)
async def api_default():
	return {'Message': "Sentiment Based Product Recommendation Model API",
		 	'FastAPI_route': '/docs'}

# API Predict recommendations	
@app.post("/api/predict", response_model=OutputDataModel)
async def api_post_predictions(inputDataModel: InputDataModel):
    ''' Default limit is 5 if provided Zero or Negative'''
    if (inputDataModel.limit <= 0):
        recommendations = apply_model(inputDataModel.user_id, TOP_N_RECOMMENDATIONS)
    else:
        recommendations = apply_model(inputDataModel.user_id, inputDataModel.limit)  
    
    recommendations = list(recommendations.values.tolist())
    
    response = {
    	"recommendations": recommendations
    }
    return response

# API List Users
@app.get("/api/users", response_model=UserDataModel)
async def api_get_users():
    users = model.get_users()
    response = {
    	"users": users
    }
    return response

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
async def app_post_predictions(request: Request, userId: str):
    userId=userId.strip().lower()
    if (userId != ''):
        recommendations = apply_model(userId, TOP_N_RECOMMENDATIONS)
        if (recommendations is not None):
            column_names=recommendations.columns.values
            recommendations = list(recommendations.values.tolist())
            return app_templates.TemplateResponse('recommendations.html', {'request': request, 'recommendations': recommendations, 'column_names': column_names, 'user_input': userId, 'zip': zip})
        else:
            message=f"User [{userId}] not found. Please try again!"
    else:
        message="UserId shouldn't be Empty, Whitespaces!"
    return app_templates.TemplateResponse('recommendations.html', {'request': request, 'err_msg': message})

if __name__ == "__main__":
    uvicorn.run(app)
