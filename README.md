# CancerDaignosisApp
We used CNN to train a breast cancer subtype classifier and did a simple UI interface for it. It's a windows Qt application, you can click to upload a histology image of breast, and the app will tell you which type it is, specifically Benign, Normal, Invasive, In Situ.
You can download all the files and run main.py and it will generate a UI interface in which you can upload images and get a prediction.
Or you can also use the console to train a model or make predictions for a bunch of images in a file.
[![Watch the video](https://raw.github.com/GabLeRoux/WebMole/master/ressources/WebMole_Youtube_Video.png)](https://github.com/WideSu/CancerDaignosisApp/blob/main/Cancer%20diagnosis%20Qt%20application.mp4)
# Functions
## Train a model
## Predict an image
## Generate sound for the diagnosis result
## Train
My network has 21 layers, the train data comes from the npy files on Google cloud disk, which is pre-processed and can grab up to use. My train data set has 4776 cases, and my evaluation data set has 240 cases. My batch size is 16, after training 20 epochs, my model can already have a 77% accuracy and 60% percent specificity. 
If you don't want to train a new model, you can still make predictions. Since I uploaded my model(my_model.h5)
# Predict
You can choose a whole-slide histology image and make predictions using the model. The result will be the probability of each section.
# UI
I use PyQt5 to do the simple UI interface which has a button to upload an image, and another button to predict the result.
# Sound
I use the Aliyun API to generate 4 sound files from text and recorded them. It will play one of the pre-recorded files after diagnosing one type of breast cancer.
