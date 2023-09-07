import os, requests, shutil
import asyncio
import concurrent.futures
import time

import pandas as pd
import traceback
path = "../img_align_celeba/"
endpoint = "http://localhost:5000"
workers = 1

def upload_secondary(userData):
    if userData[0]%300 == 0:
        print(userData[0])
    user = str(userData[0])
    if len(userData[1]['file']) <= 1:
        return
    imgs = userData[1]['file'].iloc[1:]
    if len(imgs) > 7:
        imgs = imgs[:7]
    to_send = []
    i=1
    for img in imgs:
        shutil.copy(os.path.join(path, img),os.path.join(path, user ,str(i)+".jpg"))
        to_send.append(('image', (img, open(os.path.join(path, img), 'rb'))))
        i+=1
    similarityURL = f"{endpoint}/v1w/face-auth/similarity/{user}"
    t = time.time()
    response = requests.post(
        url=similarityURL,
        files=to_send,
        #headers={'Authorization': 'Bearer ' + token},
    )
    #print("secondary upload took", time.time()-t)
    if response.status_code != 200:
        if not "NO_FACE" in response.text:
            print(f"secondary for {user} failed with {response.status_code} {response.text}")

def upload_primary(userData):
    user = str(userData[0])
    img = userData[1]['file'].iloc[0]
    shutil.copy(os.path.join(path, img),os.path.join(path, user ,"0.jpg"))
    to_send = [('image', (img, open(os.path.join(path, img), 'rb')))]
    primaryURL = f"{endpoint}/v1w/face-auth/primary_photo/{user}"
    t=time.time()
    response = requests.post(
        url=primaryURL,
        files=to_send,
        #headers={'Authorization': 'Bearer ' + token},
    )
    #print("primary upload took ", time.time()-t)
    if response.status_code != 200:
        if not "NO_FACE" in response.text:
            print(f"primary for {user} failed with {response.status_code} {response.text}")

def process_usr(u):
    try:
        try:
            os.mkdir(os.path.join(path, str(u[0])))
        except FileExistsError: pass
        upload_primary(u)
        #upload_secondary(u)
        if u[0]%300 == 0:
            print(u[0])
    except Exception as e:
        traceback.print_exc()

def load_data():
    identities = pd.read_csv(os.path.join(path,"identity_CelebA.txt"), names=["file", "identity"], header = None, sep=" ")
    gr = identities.groupby(by='identity')
    return gr

@asyncio.coroutine
def primary():
    print(len(usrs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor,
                process_usr,
                u
            )
            for u in usrs
        ]
        yield from asyncio.gather(*futures)
@asyncio.coroutine
def secondary():
    print(len(usrs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor,
                upload_secondary,
                u
            )
            for u in usrs
        ]
        yield from asyncio.gather(*futures)

usrs = load_data()
# loop = asyncio.get_event_loop()
# loop.run_until_complete(primary())
loop = asyncio.get_event_loop()
loop.run_until_complete(secondary())