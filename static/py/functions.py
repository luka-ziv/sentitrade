from nltk.sentiment.vader import SentimentIntensityAnalyzer, SentiText
import numpy as np
import pandas as pd
import contractions
import ftfy
import string
import datetime
import newsapi.newsapi_client as newsapi
import json
import mariadb
import os
from fastapi import HTTPException
import pytz


def get_headlines(symbol, from_date, to_date):
    try:
        crypto_file = open('static/metadata/crypto_metadata.json')
        crypto_metadata = json.load(crypto_file)
        crypto_file.close()
        name = crypto_metadata[symbol]
    except KeyError:
        stock_metadata = pd.read_csv('static/metadata/stock_metadata.csv')
        stock_index = np.where(stock_metadata['Symbol'].values == symbol)[0]
        name = stock_metadata['Name'][stock_index]
        # make error if even this one does not show up. Won't happen because user selects from pre-written list.
    if from_date is not None:
        from_date = datetime.datetime(
            year=int(from_date[0]),
            month=int(from_date[1]),
            day=int(from_date[2]),
            hour=0,
            minute=0
        ).isoformat()
    if to_date is not None:
        to_date = datetime.datetime(
            year=int(to_date[0]),
            month=int(to_date[1]),
            day=int(to_date[2]),
            hour=23,
            minute=59
        ).isoformat()
    titles = []
    for page in range(1, 6):
        try:
            news = newsapi.NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
            articles = news.get_everything(
                q=f'{symbol} OR {name}',    # Use the stock symbol as well. Get from other API.
                language='en',
                exclude_domains='reuters.com',
                from_param=from_date,
                to=to_date,
                page=page
            )
            titles += [articles['articles'][i]['title'] for i in range(len(articles['articles']))]
        except:
            # If page number does not exist and error produced.
            break
    return titles


def unbake(sentence):
    return ftfy.fix_text(sentence)


def decontract(sentence):
    return contractions.fix(sentence)


def depunctuate(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))


# def remove_stopwords(sentence):
#     token = nltk.word_tokenize(sentence)
#     stopwords = sw.words('english')
#     filtered_list = [word for word in token if word not in stopwords]
#     filtered = ' '.join(filtered_list)
#     return filtered


def vader_cleaning(sentences):
    sentences = check_convert_type(sentences, 'string')
    clean_titles = []
    for sentence in sentences:
        if sentence is None:
            # Error case where item in list from NewsAPI is None type. Happened once. Very weird.
            continue
        unbaked = unbake(sentence)
        expanded = decontract(unbaked)
        clean_titles.append(expanded)
    if len(clean_titles) == 1:
        return clean_titles[0]
    return clean_titles


def adj_vader_scores(sentences):
    sentences = check_convert_type(sentences, 'string')
    num_texts = len(sentences)
    adj_compounds = np.empty(0)
    for sentence in sentences:
        valence_sum = get_valence_sum(sentence)
        adj_compounds = np.append(adj_compounds, adjust_vader_score(valence_sum, num_texts))
    if len(adj_compounds) == 1:
        return adj_compounds[0]
    return np.mean(adj_compounds)


def vader_compound_scores(sentences):
    sentences = check_convert_type(sentences, 'string')
    vader = SentimentIntensityAnalyzer()
    compounds = np.empty(0)
    for sentence in sentences:
        sentiment = vader.polarity_scores(sentence)
        compounds = np.append(compounds, sentiment['compound'])
    if len(compounds) == 1:
        return compounds[0]
    return np.mean(compounds)


def check_convert_type(variable, array_contents):
    if array_contents == 'number':
        if type(variable) == int or type(variable) == float:
            return np.array([variable])
        elif type(variable) == list:
            return np.array(variable)
        elif type(variable) == np.ndarray:
            return variable
        else:
            raise TypeError('This function takes either a list, numpy array, int, or float type.')
    elif array_contents == 'string':
        if type(variable) == str:
            return [variable]
        elif type(variable) == list or type(variable) == np.ndarray:
            return variable
        else:
            raise TypeError('This function takes a single string, a list or numpy array of string elements.')


def adjust_vader_score(valence_sum, num_articles):
    '''
    Adjusts VADER sentiment compound score to account for number of articles
    used in mean compound sentiment score. Includes a "buzz" factor.
    :param valence_sum: Mean VADER compounds for article collection.
    :param num_articles: Number of articles used in mean compounds.
    :return: Adjusted sentiment score.
    '''
    return valence_sum / np.sqrt(valence_sum**2 + 15 + 30/num_articles)


def compute_sentiment_score(symbol, from_date=None, to_date=None):
    titles = get_headlines(symbol, from_date, to_date)
    if not titles:
        return 0.00, 0
    cleaned = vader_cleaning(titles)
    return adj_vader_scores(cleaned), len(cleaned)


def get_valence_sum(text):
    '''
    This function is copied from the NLTK VADER sentiment library's
    polarity_scores() method, then cut at the variable to be extracted.
    Valence sums are returned and used to get adjusted sentiment scores.
    '''
    vader = SentimentIntensityAnalyzer()
    sentitext = SentiText(
        text, vader.constants.PUNC_LIST, vader.constants.REGEX_REMOVE_PUNCTUATION
    )
    sentiments = []
    words_and_emoticons = sentitext.words_and_emoticons
    for item in words_and_emoticons:
        valence = 0
        i = words_and_emoticons.index(item)
        if (
            i < len(words_and_emoticons) - 1
            and item.lower() == "kind"
            and words_and_emoticons[i + 1].lower() == "of"
        ) or item.lower() in vader.constants.BOOSTER_DICT:
            sentiments.append(valence)
            continue

        sentiments = vader.sentiment_valence(valence, sentitext, item, i, sentiments)

    sentiments = vader._but_check(words_and_emoticons, sentiments)
    if sentiments:
        sum_s = float(sum(sentiments))
        # compute and add emphasis from punctuation in text
        punct_emph_amplifier = vader._punctuation_emphasis(sum_s, text)
        if sum_s > 0:
            sum_s += punct_emph_amplifier
        elif sum_s < 0:
            sum_s -= punct_emph_amplifier
    return sum_s


def daily_db_fill(date):
    try:
        print('Connecting to DB...')
        connection = mariadb.connect(
            user=os.getenv('DB_USERNAME'),
            passwd=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT')),
            database=os.getenv('DB_NAME')
        )
        print('Connected.')
    except mariadb.Error as error:
        print(f'Error connecting to DB: {error}')
        return

    cursor = connection.cursor()
    for coin in ['BTC', 'ETH']:
        print(f'Computing {coin} score...')
        score, num_articles = compute_sentiment_score(
            symbol=coin,
            from_date=date,
            to_date=date
        )
        cursor.execute(f'INSERT INTO {coin.lower()} (Date, Score, NumArticles) VALUES("{date[0]}-{date[1]}-{date[2]}", {score}, {num_articles});')
        print(f'Operation complete ({coin}).')
    connection.commit()
    connection.close()


def get_request_results(symbol, date):
    try:
        connection = mariadb.connect(
            user=os.getenv('DB_USERNAME'),
            passwd=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT')),
            database=os.getenv('DB_NAME')
        )
    except mariadb.Error as error:
        print(f'Error connecting to DB: {error}')
        return

    cursor = connection.cursor()
    cursor.execute(f'SELECT Score, NumArticles FROM {symbol} WHERE Date="{date}";')
    results = None
    for (score, num_articles) in cursor:
        results = (score, num_articles)
    connection.close()
    if not results:
        raise HTTPException(
            status_code=404,
            headers={'Error': 'Invalid search parameters.'},
            detail='No items found that match the search parameters you entered.')
    return {"symbol": symbol.upper(), "score": results[0], "num_articles": results[1], "date": date}


def add_site_stats(view=False, request=False):
    if view == request:
        return

    try:
        connection = mariadb.connect(
            user=os.getenv('DB_USERNAME'),
            passwd=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT')),
            database=os.getenv('DB_NAME')
        )
    except mariadb.Error as error:
        print(f'Error connecting to DB: {error}')
        return

    cursor = connection.cursor()
    date = datetime.datetime.now(pytz.timezone('EST'))
    year = date.year
    month = date.month
    day = date.day

    # Need to check if item for this date already exist in DB to avoid error.
    empty_row = True
    cursor.execute(f'SELECT Date FROM stats WHERE Date="{year}-{month}-{day}";')
    for date in cursor:
        empty_row = False

    # Initialize row if empty.
    if empty_row:
        cursor.execute('SELECT MAX(id) FROM stats;')
        result = None
        for id in cursor:
            result = id[0]
        if not result:
            # Case where table is empty.
            cursor.execute(
                f'INSERT INTO stats (Date) VALUES("{year}-{month}-{day}");')
        else:
            cursor.execute(f'SELECT MainViewsToDate, RequestsToDate FROM stats WHERE id={result};')
            results = None
            for (total_views, total_requests) in cursor:
                results = (total_views, total_requests)
            cursor.execute(f'INSERT INTO stats (Date, MainViewsToDate, RequestsToDate) VALUES("{year}-{month}-{day}", {results[0]}, {results[1]});')
    connection.commit()

    # All other columns receive default value of 0 if nothing committed already.
    if view:
        cursor.execute(f'SELECT MainPageViews, MainViewsToDate FROM stats WHERE Date="{year}-{month}-{day}";')
        results = None
        for (views, total) in cursor:
            results = (views, total)
        updated_views = int(results[0]) + 1
        updated_total = int(results[1]) + 1
        cursor.execute(f'UPDATE stats SET MainPageViews={updated_views}, MainViewsToDate={updated_total} WHERE Date="{year}-{month}-{day}";')

    if request:
        cursor.execute(f'SELECT NumRequests, RequestsToDate FROM stats WHERE Date="{year}-{month}-{day}";')
        results = None
        for (reqs, total) in cursor:
            results = (reqs, total)
        updated_reqs = int(results[0]) + 1
        updated_total = int(results[1]) + 1
        cursor.execute(f'UPDATE stats SET NumRequests={updated_reqs}, RequestsToDate={updated_total} WHERE Date="{year}-{month}-{day}";')

    connection.commit()
    connection.close()