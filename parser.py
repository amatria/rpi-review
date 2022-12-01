#!/usr/bin/env python3

import sys

from bs4 import BeautifulSoup

soup = BeautifulSoup(open(f"cache/{sys.argv[1]}"), "html.parser")

tables = soup.select("table.table-1")

# get assembly name
tbody = tables[0].select("tbody")
tr = tbody[0].select("tr")[9]
td = tr.select("td")[1]

if "hg" in td.text:
    print("Homo sapiens")
elif "mm" in td.text:
    print("Mus musculus")
elif "dm" in td.text:
    print("Drosophila melanogaster")
else:
    print("unknown")

# get sequence
tbody = tables[1].select("tbody")
tr = tbody[0].select("tr")[1]
td = tr.select("td")[0].text
print(td)
