import os
import json
import requests
from urllib.parse import urljoin

import pprint


URL = os.environ["RETRIEVER_API_URL"].split("/api")[0]

# test task_list
task_list_q = '/api/task_list'
response = requests.get(urljoin(URL, task_list_q))
print(response.status_code)
print(response.text)

# test task
task_q = '/api/task'

# default test case
data = json.dumps(
    {
        "query": ["바닷가에서 달리는 사람들",
            "화려한 원피스를 입은 젊은 여성",],
        "k": 4,
    }
)
headers = {'Content-Type': 'application/json; charset=utf-8'}  # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.status_code)
print(response.request)
pprint.pprint(response.json())


