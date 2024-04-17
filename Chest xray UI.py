#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
from PIL import Image, ImageEnhance
warnings.filterwarnings('ignore')
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QMessageBox

from win32com.client import Dispatch


def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(695, 609)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 701, 611))
        self.frame.setStyleSheet("background-color: #035874;")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(80, -60, 541, 561))
        self.label.setText("")
        self.gif=QMovie("picture.gif")
        self.label.setMovie(self.gif)
        self.gif.start()
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(80, 430, 591, 41))
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(30, 530, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("patient.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton{\n"
"border-radius: 10px;\n"
" background-color:#DF582C;\n"
"\n"
"}\n"
"QPushButton:hover {\n"
" background-color: #7D93E0;\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(450, 530, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("QPushButton{\n"
"border-radius: 10px;\n"
" background-color:#DF582C;\n"
"\n"
"}\n"
"QPushButton:hover {\n"
" background-color: #7D93E0;\n"
"}")
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.upload_image)
        self.pushButton_2.clicked.connect(self.predict_result)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PNEUMONIA Detection Apps"))
        self.label.setToolTip(_translate("MainWindow", "<html><head/><body><p><img src=\":/newPrefix/picture.gif\"/></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Chest X_ray PNEUMONIA Detection"))
        self.pushButton.setText(_translate("MainWindow", "Upload Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Prediction"))
    def upload_image(self):
        filename=QFileDialog.getOpenFileName()
        path=filename[0]
        path=str(path)
        print(path)
        model=load_model('chest_xray.h5') 
        img_file=image.load_img(path,target_size=(224,224))
        x=image.img_to_array(img_file)
        x=np.expand_dims(x, axis=0)
        img_data=preprocess_input(x)
        classes=model.predict(img_data)
        global result
        result=classes

    def predict_result(self):
        print(result)
        if result[0][0]>0.5:
            print("Result is Normal")
            speak("Result is Normal")
        else:
            print("Affected By PNEUMONIA")
            speak("Affected By PNEUMONIA")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


# In[15]:


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Function to preprocess chest X-ray images
def preprocess_image(image_path, target_size=(224, 224)):
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_path)
    
    # Resize the image to the target size
    image = image.resize(target_size)
    
    # Convert the image to a NumPy array and normalize pixel values
    image_array = np.array(image) / 255.0
    
    return image_array

# Function to load and preprocess chest X-ray images
def load_images_and_labels(dataset_path):
    image_paths = []
    labels = []

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith(".png"):  # Assuming images are in PNG format
                    image_paths.append(file_path)
                    labels.append(folder)

    return image_paths, labels

# Define the dataset path
dataset_path = 'C:/Users/chaithanya/Downloads/datasets'

# Load chest X-ray images and labels
'C:/Users/chaithanya/Downloads/datasets/chest_xray/train', labels = load_images_and_labels(dataset_path)

# Perform train-test split
X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Preprocess training and testing images
X_train_processed = [preprocess_image(image_path) for image_path in X_train_paths]
X_test_processed = [preprocess_image(image_path) for image_path in X_test_paths]

# Convert labels to binary (1 for pneumonia, 0 for no pneumonia)
label_encoder = LabelEncoder()
y_train_binary = label_encoder.fit_transform(y_train)
y_test_binary = label_encoder.transform(y_test)

# Train a simple logistic regression model
model = LogisticRegression()
scaler = StandardScaler()

# Flatten the processed images for logistic regression
X_train_flat = np.array(X_train_processed).reshape(len(X_train_processed), -1)
X_test_flat = np.array(X_test_processed).reshape(len(X_test_processed), -1)

# Scale the features
X_train_scaled = scaler.fit_transform(X_train_flat)
model.fit(X_train_scaled, y_train_binary)

# Make predictions on the test set
X_test_scaled = scaler.transform(X_test_flat)
y_pred = model.predict(X_test_scaled)

# Plotting the confusion matrix using seaborn heatmap
cm = confusion_matrix(y_test_binary, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Pneumonia', 'Pneumonia'], yticklabels=['No Pneumonia', 'Pneumonia'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()


# In[ ]:




