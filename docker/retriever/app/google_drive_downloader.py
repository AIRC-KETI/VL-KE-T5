# Copyright 2022 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import io
import requests
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?id={}&export=download&confirm=t".format(id)

    session = requests.Session()
    response = session.get(URL, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    print(response.cookies.items())
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    total_size_in_bytes= int(response.headers.get('content-length', 0))
    pbar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="download")

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()
    if total_size_in_bytes != 0 and pbar.n != total_size_in_bytes:
        logger.error("ERROR: total_size_in_bytes != downloaded_bytes")



if __name__ == "__main__":
	url = sys.argv[1]
	file_id = url.split('/')[5]
	destination = sys.argv[2]
	download_file_from_google_drive(file_id, destination)
