#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np


# In[14]:


# Process Model
model = VGG16()
image = load_img("D:/Projects/VGG16/cars.jpg", target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

# Generate predictions
pred = model.predict(image)
print('Predicted:', decode_predictions(pred, top=3)[0])
np.argmax(pred[0])


# In[15]:


# Grad-CAM algorithm
specoutput=model.output[:, 668]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(specoutput, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([image])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap=np.mean(conv_layer_output_value, axis=-1)
# Heatmap post processing
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()


# In[17]:


# Superimposing heatmap
import cv2
img = cv2.imread("D:/Projects/VGG16/cars.jpg")
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('heatmap.jpg', superimposed_img)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




