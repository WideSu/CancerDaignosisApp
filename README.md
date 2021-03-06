# CancerDaignosisApp
We used CNN to train a breast cancer subtype classifier and did a simple UI interface for it. It's a windows Qt application, you can click to upload a histology image of breast, and the app will tell you which type it is, specifically Benign, Normal, Invasive, In Situ.
You can download all the files and run main.py and it will generate a UI interface in which you can upload images and get a prediction.
Or you can also use the console to train a model or make predictions for a bunch of images in a file.
Dataset:in the preprocessed npy files.
https://drive.google.com/drive/folders/17LR9ssbENit-3vsEAM63FptNasB5AHrr
![alt text](https://github.com/WideSu/CancerDaignosisApp/blob/main/cancer%20diagnosis%20app.gif)
# Functions
## 1. Train a model
## 2. Predict an image
## 3. Generate sound for the diagnosis result
## 1. Train
My network has 21 layers, the train data comes from the npy files on Google cloud disk, which is pre-processed and can grab up to use. My train data set has 4776 cases, and my evaluation data set has 240 cases. My batch size is 16, after training 20 epochs, my model can already have a 77% accuracy and 60% percent specificity. 
If you don't want to train a new model, you can still make predictions. Since I uploaded my model(my_model.h5)
## 2. Predict
You can choose a whole-slide histology image and make predictions using the model. The result will be the probability of each section.
## 3. UI
I use PyQt5 to do the simple UI interface which has a button to upload an image, and another button to predict the result.
# 4. Text to Sound
I use the Aliyun API to generate 4 sound files from text and recorded them. It will play one of the pre-recorded files after diagnosing one type of breast cancer.
