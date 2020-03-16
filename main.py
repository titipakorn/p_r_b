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

# API
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import HTMLResponse

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
tracker = MultiCameraTracker(number_of_cameras, reid, config)


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def update_db():
    try:
        global parsed_date, parsed_time
        d, t = datetime.now().strftime("%Y-%m-%d/%H:00:00").split('/')
        if(t != parsed_time):
            url = 'http://52.74.221.188/api/insert.php'
            myobj = {'secure_code': (None, 'P4ssw0rd!'),
                     'sql': (None, json.dumps({'table': 'counting', 'values': {'counting_date': parsed_date, 'counting_time': parsed_time, 'counting_in': SingleCameraTracker.COUNT_IN, 'counting_out': SingleCameraTracker.COUNT_OUT, 'branch_id': 2}}))}
            requests.post(url, files=myobj)
            parsed_date = d
            parsed_time = t
            SingleCameraTracker.COUNT_IN = 0
            SingleCameraTracker.COUNT_OUT = 0

    except:
        print('no internet')


global parsed_date, parsed_time
parsed_date, parsed_time = datetime.now().strftime("%Y-%m-%d/%H:00:00").split('/')
timer = RepeatTimer(60, update_db)
timer.start()

TIME_FM = '-%Y%m%d-%H%M%S'
if not os.path.exists('./raw_data'):
    os.makedirs('raw_data')


@app.get("/status/")
def read_status():
    global parsed_date, parsed_time
    return {"counting_in": SingleCameraTracker.COUNT_IN, "counting_out": SingleCameraTracker.COUNT_OUT, "date": parsed_date, "time": parsed_time}


@app.post("/track/")
async def update_track(bboxes: str = Body(..., embed=True), files: List[UploadFile] = File(...)):
    #time_str = time.strftime(TIME_FM)
    d_bboxes = json.loads(bboxes)
    # for i, f in enumerate(files):
    #     b_name = "_".join(map(str, d_bboxes[i]))
    #     Image.open(f.file).save(f"raw_data/{time_str}_{b_name}.jpg")
    tracker.process([files], [d_bboxes])
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
