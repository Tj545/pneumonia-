#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[3]:


from keras.layers import Input,Lambda,Dense, Flatten 


# In[4]:


# using tensorflow backend 


# In[5]:


from keras.models import Model 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input 
from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
import numpy as np
from glob import glob 
import matplotlib.pyplot as plt 


# In[6]:


IMAGE_SIZE =[224,224]
train_path ='C:/Users/chaithanya/Downloads/datasets/chest_xray/train'
test_path = 'C:/Users/chaithanya/Downloads/datasets/chest_xray/test'


# In[7]:


vgg = VGG16(input_shape =IMAGE_SIZE + [3], weights ='imagenet',include_top=False)


# In[8]:


for layer in vgg.layers:
    layer.trainable = False 


# In[9]:


folders = glob('C:/Users/chaithanya/Downloads/datasets/chest_xray/train/*')
x = Flatten() (vgg.output)


# In[16]:


prediction = Dense(len(folders), activation ='softmax')(x)
model = Model(inputs = vgg.input,outputs=prediction)
model.summary()


# In[10]:


model.compile(
    loss = 'categorical_crossentropy', 
    optimizer='adam',
    metrics= ['accuracy']
)


# In[11]:


from keras.preprocessing.image import ImageDataGenerator 


# In[12]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True
                                  )
test_datagen = ImageDataGenerator(rescale =1./255)
training_set = train_datagen.flow_from_directory('C:/Users/chaithanya/Downloads/datasets/chest_xray/train',
                                                target_size=(224,224),
                                                batch_size=10,
                                                class_mode='categorical')
test_set = test_datagen.flow_from_directory('C:/Users/chaithanya/Downloads/datasets/chest_xray/test',
                                            target_size=(224,224),
                                            batch_size=10,
                                            class_mode='categorical')


# In[13]:


r = model.fit_generator(
 training_set,
 validation_data=test_set,
 epochs=1,
 steps_per_epoch=len(training_set),
 validation_steps=len(test_set)
) 





# In[95]:


import tensorflow as tf 
from keras.models import load_model 
model.save('chest_xray.h5')


# In[96]:


from keras.models import load_model


# In[97]:


from keras.preprocessing import image 


# In[98]:


from keras.applications.vgg16 import preprocess_input


# In[100]:


import numpy as np


# In[102]:


model = load_model('chest_xray.h5')


# In[112]:


img = image.load_img('C:/Users/chaithanya/Downloads/datasets/chest_xray/chest_xray/train/PNEUMONIA/person9_bacteria_38.jpeg', target_size = (224,224))


# In[113]:


x= image.img_to_array(img)


# In[114]:


x= np.expand_dims(x, axis=0)


# In[115]:


img_data = preprocess_input(x)


# In[116]:


classes = model.predict(img_data)


# In[117]:


classes


# In[118]:


result = classes[0][0]


# In[119]:


if result > 0.5:
    print("Result is Normal")
else: 
    print("Affected by pnemonina")


# In[ ]:




