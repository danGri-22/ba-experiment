#this file is used to save some images 

import numpy as np
import torch
import torchvision
import cv2
from uuid import uuid4 # D.Griesser: import added
from google.colab.patches import cv2_imshow # necessary to use cv2.imshow() in colab

def save_image(name,input1):
    
    input1 = torchvision.utils.make_grid(input1)
    inp = input1.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([129.1863, 104.7624,93.5940  ])
    inp = inp + mean
    inp = np.clip(inp, 0, 255)  
    inp = inp.astype('uint8')
    name += str(uuid4()) # D.Griesser: added to fix cv2.imwrite()

    # cv2.imshow("test", inp) --> D.Griesser: added to display image
    # cv2.waitKey(0) --> D.Griesser: necessary to display the image (image is displayed until key stroke)
    cv2.imwrite("data/output/"+name+".jpg", inp) # --> doesn't work because of special characters in name 
    # cv2.imwrite("./image_test/test" + ".jpg",inp)
        
def save_image2(name,input1):
    
    #input1 = torchvision.utils.make_grid(input1)
    inp = input1.transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)*255
    inp = inp.astype('uint8')
    
    cv2.imwrite("./image_test/"+name+".jpg",inp)


# D.Griesser: Function added
def show_image(input, name="image"):
    
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)

    input = torchvision.utils.make_grid(input)
    inp = input.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([129.1863, 104.7624,93.5940  ])
    inp = inp + mean
    inp = np.clip(inp, 0, 255)  
    inp = inp.astype('uint8')

    # cv2.imshow(name, inp) # --> D.Griesser: added to display image
    cv2_imshow(inp) # used to display images in google colab
    cv2.waitKey(0) # --> D.Griesser: necessary to display the image (image is displayed until key stroke)

# D.Griesser: Function added
def compare_images(org_file, adv_file, path="data/images/"):

    old = cv2.imread(path + org_file)
    new = cv2.imread(path + adv_file)
    
    difference = new - old
    
    cv2.imshow("difference", difference)
    cv2.waitKey(0)