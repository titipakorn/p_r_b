import os
from typing import List
from pydantic import BaseModel

# API
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

# CONFIG
from config import Config
from mc_tracker.mct import MultiCameraTracker
from mc_tracker.sct import SingleCameraTracker

# REID_MODEL
from utils.network_wrappers import VectorCNN


config = Config()
os.environ['CUDA_VISIBLE_DEVICES'] = config.DEVICE_ID
app = FastAPI()


number_of_cameras = 1

reid = VectorCNN(config)
tracker = MultiCameraTracker(number_of_cameras, reid, config)


# class Data(BaseModel):
#     bboxes: List[List[float]]


@app.get("/status/")
def read_status():
    return {"counting_in": SingleCameraTracker.COUNT_IN, "counting_out": SingleCameraTracker.COUNT_OUT}


@app.post("/track/")
async def update_track(bboxes: List[List[int]], files: List[UploadFile] = File(...)):
    tracker.process([files], [data.bboxes])
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
