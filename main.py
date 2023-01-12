from fastapi import FastAPI, HTTPException, Request
from starlette.responses import RedirectResponse
from starlette.datastructures import URL
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi_utils.tasks import repeat_every
from static.py import functions as fns
import datetime as dt


app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

@app.route('/')
def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/endpoint')
def get_request(symbol, date):
    return fns.get_request_results(symbol=symbol.upper(), date=date)


@app.route('/demo')
def demo_get_request(request: Request):
    redirect_url = URL(request.url_for('get_request')).include_query_params(symbol='BTC', date='2023-01-01')
    return RedirectResponse(redirect_url)


@app.route('/donate')
def donate(request: Request):
    return templates.TemplateResponse('donation.html', {'request': request})


@app.route('/coming-soon')
def coming_soon(request: Request):
    return templates.TemplateResponse('coming_soon.html', {'request': request})



startup = str(dt.date.today()).split('-')
NEXT_DB_UPDATE = dt.datetime(year=int(startup[0]), month=int(startup[1]), day=int(startup[2])) + dt.timedelta(days=1, hours=1)
@app.on_event('startup')
@repeat_every(seconds=60*60)
def daily_db_update():
    global NEXT_DB_UPDATE
    now = str(dt.datetime.now()).split('.')[0][:-3]
    now_date = now.split()[0]
    now_time = now.split()[1]
    next_date = str(NEXT_DB_UPDATE)[:-3].split()[0]
    next_time = str(NEXT_DB_UPDATE)[:-3].split()[1]
    if now_date == next_date and now_time[:2] == next_time[:2]:
        date = str(dt.date.today() - dt.timedelta(days=1)).split('-')
        fns.daily_db_fill(date=(int(date[0]), int(date[1]), int(date[2])))
        NEXT_DB_UPDATE += dt.timedelta(days=1)