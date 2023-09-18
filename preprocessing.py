import cv2
from PIL import Image
import numpy as np
import os, glob

def data_gen(infected, uninfected):
    data = []
    labels = []

    for i in glob.glob(os.path.join(infected,"*")):
        try:
        
            image = cv2.imread(i)
            image_array = Image.fromarray(image , 'RGB')
            resize_img = image_array.resize((50 , 50))
            rotated45 = resize_img.rotate(45)
            rotated75 = resize_img.rotate(75)
            blur = cv2.blur(np.array(resize_img) ,(10,10))
            data.append(np.array(resize_img))
            data.append(np.array(rotated45))
            data.append(np.array(rotated75))
            data.append(np.array(blur))
            labels.append(1)
            labels.append(1)
            labels.append(1)
            labels.append(1)
            
        except AttributeError:
            print('')
        
    for u in glob.glob(os.path.join(uninfected,"*")):
        try:
            
            image = cv2.imread(u)
            image_array = Image.fromarray(image , 'RGB')
            resize_img = image_array.resize((50 , 50))
            rotated45 = resize_img.rotate(45)
            rotated75 = resize_img.rotate(75)
            data.append(np.array(resize_img))
            data.append(np.array(rotated45))
            data.append(np.array(rotated75))
            labels.append(0)
            labels.append(0)
            labels.append(0)
            
        except AttributeError:
            print('')

    cells = np.array(data)
    labels = np.array(labels)

    np.save('Cells' , cells)
    np.save('Labels' , labels)

    return cells, labels