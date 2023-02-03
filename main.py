from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ISORT_inference import ISORT

app = FastAPI()
mutex_lock = asyncio.Lock()
modelname0 = r"{}\\U-Net_256.pt"
modelname1 = r"{}\\U-Net_256.pt"
isort0 = ISORT(scale_img_devide=2, HALF=False , weights=modelname0)
isort1 = ISORT(scale_img_devide=2, HALF=False , weights=modelname1)
save_image = True

class data_input(BaseModel):
    image: bytes = None
    criteria_defect_mole_all: float = 0
    criteria_defect_mole_instant: float = 0
    criteria_defect_pad: float = 0
    selectmodel: str = 0

@app.get("/")
def hello():
    return "Hi server is working"

@app.post("/predict_test")
async def pred(request: data_input):

    selectmodel: str = request.selectmodel
    try:
        if selectmodel == "0" :
         isort = isort0
         isort0.decode_base64(request.image)
         crop: list[int] = [675,675+653,609,609+785]
        if selectmodel == "1" :
         isort = isort1
         isort1.decode_base64(request.image)
         crop: list[int] = []

    except:
        error_convert_image_base64: dict = {"status": "Error can not byte to image please recheck your image in json package"}
        return JSONResponse(status_code=400 , content=error_convert_image_base64)
    image_base64: bytes = request.image
    criteria_defect_mole_all: float = request.criteria_defect_mole_all
    criteria_defect_mole_instant: float = request.criteria_defect_mole_instant
    criteria_defect_pad: float = request.criteria_defect_pad

    try:
        start_time: float = time.time()
        #crop: list[int] = [675,675+653,609,609+785]
        # if use model U-Net_256 set [scale_img_devide] to 2
        # if use model U-Net_512 set [scale_img_devide] to 1
      #  if selectmodel == "1" :
      #   image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pads, defect_mole_instant_per_bgs = await isort.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=mutex_lock)
        
      #  if selectmodel == "2" :
      #   image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pads, defect_mole_instant_per_bgs = await isort1.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=mutex_lock)
        image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pads, defect_mole_instant_per_bgs = await isort.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=mutex_lock)
        defect_mole_per_bg_flag: bool = defect_mole_per_bg > criteria_defect_mole_all
        defect_pad_per_pad_flag: list[bool] = [defect_pad_per_pad > criteria_defect_pad for defect_pad_per_pad in defect_pad_per_pads]
        defect_mole_instant_per_bg_flag: list[bool] = [defect_mole_instant_per_bg > criteria_defect_mole_instant for defect_mole_instant_per_bg in defect_mole_instant_per_bgs]

        pred: dict = {
            "defect_mole_per_bg": defect_mole_per_bg,
            "defect_pad_per_pad": defect_pad_per_pads,
            "defect_mole_instant_per_bg": defect_mole_instant_per_bgs,
            "defect_mole_per_bg_flag": defect_mole_per_bg_flag,
            "defect_pad_per_pad_flag": defect_pad_per_pad_flag,
            "defect_mole_instant_bg_flag": defect_mole_instant_per_bg_flag,
        }
        if save_image:
            path = r"D:\ISORT U-Net\save_image"
            name = str(defect_mole_per_bg)
            #if selectmodel == "1" :
            with ThreadPoolExecutor() as exe:
              _ = exe.submit(isort.save_image, path, name+".png", image_ori_with_pred)
           # if selectmodel == "2" :
          #   with ThreadPoolExecutor() as exe:
           #      _ = exe.submit(isort1.save_image, path, name+".png", image_ori_with_pred)
        end_time: float = time.time()
        
        print(f"Time taken per image: {end_time-start_time}")

        return JSONResponse(status_code=200, content=pred)
    except:
        error_predict: dict = {
            "status": "Error at predict function please report to AUTOMATION Team"
        }
        return JSONResponse(status_code=500 , content=error_predict)


if __name__ == "__main__":
    uvicorn.run("main:app", host="210.4.134.226", port=8000)
