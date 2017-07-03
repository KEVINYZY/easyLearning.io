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

def putHistoryPrice(df):
    pass

df = getHistoryPrice_5y('BABA')


