import bs4
import requests
import datetime
import MySQLdb

def getAllSymbols():
    rep = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs4.BeautifulSoup(rep.text)

    ## fist <TABLE> in html body, rows from 1 to end
    allSymbolRows = soup.select('table')[0].select('tr')[1:]

    ## return value
    now = datetime.datetime.utcnow()
    allSymbols = []
    for i , row in enumerate(allSymbolRows):
        tds = row.select('td')

        allSymbols.append(
                (
                    tds[0].select('a')[0].text,  # ticker
                    'stock',                     # instrument
                    tds[1].select('a')[0].text,  # name
                    tds[3].text,                 # sector
                    'USD',                       # currency
                    now,                         # create
                    now                          # update
                )
        )


    return allSymbols

def putAllSymbols(allSymbols):
    db = MySQLdb.connect(host="localhost", user="root", passwd="myquant", db="myquant")

    fieldsString = """
                   ticker, instrument, name, sector, currency, create_date, update_date
                   """

    valuesString = ("%s, " * 7)[:-2]
    sqlString = "INSERT INTO symbol (%s) VALUES(%s) " % (fieldsString, valuesString)

    cur = db.cursor()
    cur.executemany(sqlString, allSymbols)
    db.commit()

    cur.close()
    db.close()

symbols = getAllSymbols()
putAllSymbols(symbols)

