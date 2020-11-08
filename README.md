# Tiger classifier
A simple classifier to detect if there is an Tiger in a Video Stream and using a webcam.

# Prerequisities
Download the Resnet-50 pretrained model [here](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5) to the root folder.

Then run,
```
pip install -r requirements.txt
```
I would suggest to use a virtualenv if you do not want to mess with your existing version of libraries. [Here](https://realpython.com/lessons/creating-virtual-environment/) is a nice video on how to set up one.

# Training the model
I have used transfer learning to train my model and used really low number of images. But I got a decent output. You can play around with the `train.py` file and modify it to suit your needs. You can also add more images to the training and validation dataset in the `images` folder. Once you have made your changes, run,
```
python train.py
```
This will create a `tiger_detector_model.h5` file in the `model_output` directory.


# Testing
To test the model that I already created, run,
```
python testing.py
```
This will read images from the `validation/test` directory. So be sure to add your test images in that folder. This will write the detected images to `validation/testing_output`.

If you want to test the video that I used, simply run,
```
python detect_video.py
```
but if you want to try a video of your own, edit the `detect_video.py` script and assign the path of your video to the  `video_capture` variable. Then run,
```
python detect_video.py
```

You can also play around with your webcam by running,
```
python detect_webcam.py
```

# References
Pyimagesearch.com