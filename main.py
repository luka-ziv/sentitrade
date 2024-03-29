from fastapi import FastAPI, Request
from starlette.responses import RedirectResponse
from starlette.datastructures import URL
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi_utils.tasks import repeat_every
from static.py import functions as fns
import datetime
import pytz


app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

@app.route('/')
def home(request: Request):
    fns.add_site_stats(view=True)
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/endpoint')
def get_request(symbol, date):
    fns.add_site_stats(request=True)
    return fns.get_request_results(symbol=symbol.lower(), date=date)


@app.route('/demo')
def demo_get_request(request: Request):
    redirect_url = URL(request.url_for('get_request')).include_query_params(symbol='BTC', date='2023-01-01')
    return RedirectResponse(redirect_url)


@app.route('/coming-soon')
def coming_soon(request: Request):
    return templates.TemplateResponse('coming_soon.html', {'request': request})


startup = str(datetime.datetime.now(pytz.timezone('EST'))).split()[0].split('-')
NEXT_DB_UPDATE = datetime.datetime(year=int(startup[0]), month=int(startup[1]), day=int(startup[2])) + datetime.timedelta(days=1, hours=1)
@app.on_event('startup')
@repeat_every(seconds=60*60)
def daily_db_update():
    global NEXT_DB_UPDATE
    now = str(datetime.datetime.now(pytz.timezone('EST'))).split('.')[0][:-3]
    print('Current datetime:', now)
    print('Next DB update datetime:', NEXT_DB_UPDATE)
    now_date = now.split()[0]
    now_time = now.split()[1]
    next_date = str(NEXT_DB_UPDATE)[:-3].split()[0]
    next_time = str(NEXT_DB_UPDATE)[:-3].split()[1]
    print('DB Update Date:', now_date == next_date)
    print('DB Update Time:', now_time[:2] == next_time[:2])
    if now_date == next_date and now_time[:2] == next_time[:2]:
        date = str(datetime.datetime.now(pytz.timezone('EST')) - datetime.timedelta(days=1)).split()[0].split('-')
        print('Filling...')
        fns.daily_db_fill(date=(int(date[0]), int(date[1]), int(date[2])))
        print('DB Filled.')
        NEXT_DB_UPDATE += datetime.timedelta(days=1)
        print('Next fill:', NEXT_DB_UPDATE)


@app.on_event('startup')
def run_locally():
    try:
        import envvars
        envvars.set_env_vars()
    except:
        pass