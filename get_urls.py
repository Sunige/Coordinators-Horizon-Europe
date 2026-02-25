import requests
import json

data = requests.get('https://data.europa.eu/api/hub/search/datasets/cordis-eu-research-projects-under-horizon-europe-2021-2027').json()
urls = [d.get('access_url') for d in data['result']['distributions'] if 'csv' in str(d).lower()]
with open('urls.txt', 'w') as f:
    for url in urls:
        f.write(str(url) + '\n')
