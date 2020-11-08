from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

model = load_model("tiger_detector_model.h5")

TESTING_DIR = "validation/test/"
CURR_DIR = os.getcwd()
OUTPUT_DIR = "validation/testing_output"

for image in os.listdir(TESTING_DIR):
    print(image)
    if image.endswith(".DS_Store"):
        continue
    image_path = os.path.join(CURR_DIR, TESTING_DIR, image)
    print(image_path)
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (224, 224))
    im = Image.fromarray(frame, "RGB")
    # Resizing into 224 X 224 because we trained the model with this image size.
    img_array = np.array(im)
    # Our keras model used a 4D tensor, (images x height x width x channel)
    # So changing dimension 128x128x3 into 1x128x128x3
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    print(pred)
    
    # If the prediction scores are above 0.5 we are assuming it is a Tiger
    if pred[0][1] > 0.5:
        cv2.putText(
            frame,
            "Tiger",
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0), 2)
    
    else:
        cv2.putText(
            frame,
            "Not Tiger",
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    # cv2.imshow("Video", image)
    # cv2.waitKey(2000)
    output_path = f"{OUTPUT_DIR}/{image}"
    print(output_path)
    cv2.imwrite(output_path, frame)