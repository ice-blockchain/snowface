import os, requests, shutil, urllib3
import asyncio
import concurrent.futures
import random
import time

import pandas as pd
import traceback
path = "../img_align_celeba/"
endpoint = "http://localhost:5000"
similarity_endpoint = "http://localhost:5000"
workers = 4
from prometheus_client import Histogram, Counter, generate_latest, REGISTRY
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
primary_photo_response_time = Histogram("primary_photo_response_time","primary_photo_response_time")
similarity_response_time = Histogram("similarity_response_time","primary_photo_response_time")
primary_photo_failures = Counter("primary_photo_failures", "primary_photo_failures")
similarity_failures = Counter("similarity_failures", "similarity_failures")
def upload_secondary(userData):
    if userData[0]%300 == 0:
        print("secondary",userData[0])
    user = str(userData[0])
    if len(userData[1]['file']) <= 1:
        return
    imgs = random.choices(list(userData[1]['file'].iloc), k = 7)
    to_send = []
    i=1
    for img in imgs:
        #shutil.copy(os.path.join(path, img),os.path.join(path, user ,str(i)+".jpg"))
        to_send.append(('image', (img, open(os.path.join(path, img), 'rb'))))
        i+=1
    similarityURL = f"{similarity_endpoint}/v1w/face-auth/similarity/{user}"
    t = time.time()
    try:
        with similarity_response_time.time():
            response = requests.post(
                url=similarityURL,
                files=to_send,
                #headers={'Authorization': 'Bearer ' + token},
                verify=False,
            )
    except Exception as e:
        print(str(e))
        raise e
    #print("secondary upload took", time.time()-t)
    if response.status_code != 200:
        if not "NO_FACE" in response.text:
            print(f"secondary for {user} failed with {response.status_code} {response.text}")
            similarity_failures.inc()
        else:
            upload_secondary(userData)

def upload_primary(userData):
    if userData[0]%300 == 0:
        print("primary",userData[0])
    user = str(userData[0])
    img = random.choice(list(userData[1]['file'].iloc))
    #shutil.copy(os.path.join(path, img),os.path.join(path, user ,"0.jpg"))
    to_send = [('image', (img, open(os.path.join(path, img), 'rb')))]
    primaryURL = f"{endpoint}/v1w/face-auth/primary_photo/{user}"
    t=time.time()
    try:
        with primary_photo_response_time.time():
            response = requests.post(
                url=primaryURL,
                files=to_send,
                #headers={'Authorization': 'Bearer ' + token},
                verify=False,
            )
    except Exception as e:
        print(str(e))
        raise e
    if response.status_code != 200:
        if not "NO_FACE" in response.text:
            print(f"primary for {user} failed with {response.status_code} {response.text}")
            primary_photo_failures.inc()
        elif len(list(userData[1]['file'].iloc)) > 1:
            try: upload_primary(userData)
            except RecursionError as e:
                print(user, str(e))
                return 0
    return response.status_code

def load_data():
    identities = pd.read_csv(os.path.join(path,"identity_CelebA.txt"), names=["file", "identity"], header = None, sep=" ")
    gr = identities.groupby(by='identity')
    return gr
def upload_both(userData):
    upload_primary(userData)
    upload_secondary(userData)

@asyncio.coroutine
def call(fn):
    print(len(usrs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor,
                fn,
                u
            )
            for u in usrs
        ]
        yield from asyncio.gather(*futures)

usrs = load_data()
loop = asyncio.get_event_loop()
loop.run_until_complete(call(upload_both))
print(str(generate_latest(REGISTRY)))
# loop = asyncio.get_event_loop()
# loop.run_until_complete(call(upload_secondary))
# print(str(generate_latest(REGISTRY)))
