# Importing the libraries
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("model_output/tiger_detector_model.h5")

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    tiger = cv2.resize(frame, (224, 224))
    im = Image.fromarray(tiger, "RGB")
    # Resizing into 224 X 224 because we trained the model with this image size.
    img_array = np.array(im)
    # Our keras model used a 4D tensor, (images x height x width x channel)
    # So changing dimension 128x128x3 into 1x128x128x3
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    print(pred)

    name = "None matching"
    print(pred)
    if pred[0][1] > 0.5:
        name = "Tiger"
        cv2.putText(
            frame,
            name,
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
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
