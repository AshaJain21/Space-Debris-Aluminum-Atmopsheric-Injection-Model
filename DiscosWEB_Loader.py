from pprint import pprint
import requests
import pandas as pd
from openpyxl import load_workbook
import time

def create_excel(data, filename):
    df = pd.json_normalize(data)
    df.to_excel(filename)
    return df

def pull_data_from_discos():
    URL = 'https://discosweb.esoc.esa.int'
    token = 'ImZiZGQ0ZDMxLWQ2MTktNDQ5YS04N2RjLWNmZWJjNTFiYTUyMyI.aVJHVPsWkUyU0-8Fjg-dIEssFrg'

    page_number = 1
    page_max = 606
    data_list= []

    while page_number <= page_max:
        response = requests.get(
            f"{URL}/api/objects?page[number]={page_number}&page[size]=100&include=reentry&sort=id&fields[reentry]=epoch",
            headers={
                'Authorization': f'Bearer {token}',
                'DiscosWeb-Api-Version': '2',
            },
            params={},
        )
        doc = response.json()
        if response.ok:
            if page_number == 1:
                page_max = doc['meta']['pagination']['totalPages']

            data_list.extend(doc['data'])
            filename = './disco_data/sheet' + str(page_number) + '.xlsx'
            create_excel(doc['data'], filename)
            page_number = page_number + 1
        else:
            print("Error")
            print(response.headers)
            print("Sleeping for a minute. Will try to pull data again.")
            print(str(page_number)+ " out of " + str(page_max))
            time.sleep(60)

    df = create_excel(data_list, "./discoweb_reentries.xlsx")
    return df

def pull_reentry_data_for_object(object_id):
    URL = 'https://discosweb.esoc.esa.int'
    token = 'ImZiZGQ0ZDMxLWQ2MTktNDQ5YS04N2RjLWNmZWJjNTFiYTUyMyI.aVJHVPsWkUyU0-8Fjg-dIEssFrg'

    response = requests.get(
            f"{URL}/api/objects/{object_id}/reentry",
            headers={
                'Authorization': f'Bearer {token}',
                'DiscosWeb-Api-Version': '2',
            },
            params={},
        )
    doc = response.json()
    if response.ok:
        return doc['data']
    else:
        print("Error")
        print(response.headers)
        return None


def main():
    pull_data_from_discos()

if __name__ == "__main__":
  main()

