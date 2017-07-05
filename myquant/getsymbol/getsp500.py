import bs4
import requests
import json

def getAllSymbols():
    rep = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs4.BeautifulSoup(rep.text)

    ## fist <TABLE> in html body, rows from 1 to end
    allSymbolRows = soup.select('table')[0].select('tr')[1:]

    ## return value
    allSymbols = []
    for i , row in enumerate(allSymbolRows):
        tds = row.select('td')

        allSymbols.append(
                {
                    "ID":       tds[7].text,                 # CIK ID
                    "TICKER":   tds[0].select('a')[0].text,  # ticker
                    "NAME":     tds[1].select('a')[0].text,  # name
                    "SECTOR":   tds[3].text,                 # GICS sector
                    "SUB":      tds[4].text                  # GICS Sub Industry
                }
        )

    return allSymbols

def putAllSymbols(jsonFile, allSymbols):
    with open(jsonFile, "w") as jfile:
        jfile.write("[\n")
        for symbol  in allSymbols:
            jline = json.dumps(symbol)
            jfile.write(jline)
            jfile.write(",\n")
        jfile.write("]")

symbols = getAllSymbols()
putAllSymbols("temp.json", symbols)

