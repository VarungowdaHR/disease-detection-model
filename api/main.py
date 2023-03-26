from ctypes import resize
from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn

app=FastAPI()

MODEL = tf.keras.models.load_model("./saved_models/test10")
CLASS_NAMES=['rust','Tea leaf blight','Tea red leaf spot','Tea red scab','algal-leaf-spot', 'blister-blight', 'healty-leaf']

# @app.get("/ping")
# async def ping():
#     return "hello i am here"

def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    image = read_file_as_image(await file.read()) 
    image=tf.image.resize_with_pad(image, 256, 256)
    imgs_batch=np.expand_dims(image, 0)
    predictions=MODEL.predict(imgs_batch)
    print(predictions[0])
    predicted_class=CLASS_NAMES[np.argmax(predictions[0])]
    print(predictions[0])
    confidence=np.max(predictions[0])
    return {
        'class':predicted_class,
        'confidence': float(confidence)
    }
    
    
    



if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)