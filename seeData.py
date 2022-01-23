#import libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

#define array n path
path = "images"
images = []
myList = os.listdir(path)
list_dir = []

for list in myList:
    list_dir.append(list)

#get images and label
i = 1
for dir in list_dir:
    image_folder_path = os.path.join(path, dir)
    for image_path in os.listdir(image_folder_path):
        img = cv2.imread(os.path.join(image_folder_path, image_path))[:,:,::-1]
        plt.subplot(4, 4, i)
        plt.axis("off")
        image = np.array(img)
        plt.imshow(image)
        i += 1

# plt.xticks([])
# plt.yticks([])
plt.show()
#encode images
