import os
from typing import List
from pydantic import BaseModel
import logging
import json
import requests
from datetime import datetime
from threading import Timer
import time
from PIL import Image
import cv2
import numpy as np

# API
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import HTMLResponse

import torch

# CONFIG
from config import Config
from mc_tracker.mct import MultiCameraTracker
from mc_tracker.sct import SingleCameraTracker

# REID_MODEL
from utils.network_wrappers import VectorCNN

logger = logging.getLogger("api")

config = Config()
os.environ['CUDA_VISIBLE_DEVICES'] = config.DEVICE_ID
app = FastAPI()


number_of_cameras = 1

reid = VectorCNN(config)
global tracker
tracker = {}


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def update_db():
    try:
        global parsed_date, parsed_time, tracker
        d, t = datetime.now().strftime("%Y-%m-%d/%H:00:00").split('/')
        if(t != parsed_time):
            for branch_id in tracker:
                url = 'http://52.74.221.188/api/insert.php'
                myobj = {'secure_code': (None, 'P4ssw0rd!'),
                         'sql': (None, json.dumps({'table': 'counting', 'values': {'counting_date': parsed_date, 'counting_time': parsed_time, 'counting_in': tracker[branch_id].scts[0].COUNT_IN, 'counting_out': tracker[branch_id].scts[0].COUNT_OUT, 'branch_id': branch_id}}))}
                requests.post(url, files=myobj)
                parsed_date = d
                parsed_time = t
                tracker[branch_id].scts[0].COUNT_IN = 0
                tracker[branch_id].scts[0].OUNT_OUT = 0

    except:
        print('no internet')


global parsed_date, parsed_time
parsed_date, parsed_time = datetime.now().strftime("%Y-%m-%d/%H:00:00").split('/')
timer = RepeatTimer(60, update_db)
timer.start()

TIME_FM = '-%Y%m%d-%H%M%S'
if not os.path.exists('./raw_data'):
    os.makedirs('raw_data')


@app.get("/status/{branch_id}")
def read_status(branch_id: int):
    global parsed_date, parsed_time, tracker
    return {"counting_in": tracker[branch_id].scts[0].COUNT_IN, "counting_out": tracker[branch_id].scts[0].COUNT_OUT, "date": parsed_date, "time": parsed_time}


@app.get("/clear/")
def reset():
    global tracker
    tracker = {}
    torch.cuda.empty_cache()
    return {"status": "success"}


@app.post("/track/{branch_id}")
async def update_track(branch_id: int, bboxes: str = Body(..., embed=True), files: List[UploadFile] = File(...)):
    global tracker
    if(branch_id not in tracker):
        tracker[branch_id] = MultiCameraTracker(
            number_of_cameras, reid, config)
    d_bboxes = json.loads(bboxes)
    tracker[branch_id].process([[cv2.imdecode(np.fromstring(
        im.file.read(), np.uint8), cv2.IMREAD_COLOR) for im in files]], [d_bboxes])
    return {"status": 'success'}


@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
