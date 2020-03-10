import os
from typing import List
from pydantic import BaseModel
import logging
import json
import requests
from datetime import datetime
from threading import Timer


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
timer = RepeatTimer(60, update_db)
timer.start()
parsed_date, parsed_time = datetime.now().strftime("%Y-%m-%d/%H:00:00").split('/')


@app.get("/status/")
def read_status():
    return {"counting_in": SingleCameraTracker.COUNT_IN, "counting_out": SingleCameraTracker.COUNT_OUT}


@app.post("/track/")
async def update_track(bboxes: str = Body(..., embed=True), files: List[UploadFile] = File(...)):
    tracker.process([files], [json.loads(bboxes)])
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
