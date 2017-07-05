from io import StringIO
import requests
import datetime
import MySQLdb
import pandas as pd

def getHistoryPrice_5y(ticker):
    url = "http://www.google.com/finance/historical?q=%s&startdate=%s&enddate=%s&num=30&output=csv"
    end_date = datetime.date.today().isoformat()
    start_date = datetime.date(int(end_date[0:4])-5, int(end_date[5:7]), int(end_date[8:10]))
    end_date = datetime.date(int(end_date[0:4]),int(end_date[5:7]),int(end_date[8:10]))
    url = url % (ticker, start_date.strftime('%b %d, %Y'), end_date.strftime('%b %d, %Y'))

    csvString = requests.get(url).text
    csvIO = StringIO(csvString)
    df = pd.read_csv(csvIO, sep=",")
    print(csvString)
    return df

def putHistoryPrice(df, symbol_ticker):
    now = datetime.datetime.utcnow()    # Date,Open,High,Low,Close,Volume
    fieldsString = """
                   symbol_ticker, price_date,
                   open_price, high_price, low_price, close_price, volume,
                   create_date, update_date
                   """

    valuesString = "'%s', %s, %s, %s, %s, %s"
    sqlString = "INSERT INTO daily_price (%s) VALUES('%s', %s, '%s', '%s') " % (fieldsString, symbol_ticker, valuesString, now, now)

    prices = []
    for i in range(0, len(df.values)):
        prices.append( tuple( df.values[i] ) )

    sql = sqlString % prices[0]


df = getHistoryPrice_5y('BABA')
putHistoryPrice(df, 'BABA')

