import bs4
import requests
import json

def buildSymbols(allSymbols, allSymbolRows, exchange):
    ## return value
    for i , row in enumerate(allSymbolRows):
        tds = row.select('td')

        allSymbols.append(
                {
                    "TICKER":   tds[1].text,                    # ticker
                    "NAME":     tds[0].text,                    # name
                    "EXCHANGE": exchange,                       # exchange
                    ##"SECTOR":   "Information Technology",       # sector
                    ##"SUB":      "Internet Software & Services"  # sub
                }
        )

def getAllSymbols():
    allSymbols = []

    rep = requests.get('https://en.wikipedia.org/wiki/China_Concepts_Stock')
    soup = bs4.BeautifulSoup(rep.text)

    allSymbolRows = soup.select('table')[0].select('tr')[1:]
    buildSymbols(allSymbols, allSymbolRows, "NYSE")

    allSymbolRows = soup.select('table')[1].select('tr')[1:]
    buildSymbols(allSymbols, allSymbolRows, "NDAQ")

    return allSymbols

def putAllSymbols(jsonFile, allSymbols):
    with open(jsonFile, "w") as jfile:
        jfile.write("[\n")
        for symbol  in allSymbols:
            jline = json.dumps(symbol)
            jfile.write(jline)
            jfile.write(",\n")

        jfile.write("{}\n")
        jfile.write("]")

symbols = getAllSymbols()
putAllSymbols("ccs_symbol.json", symbols)

