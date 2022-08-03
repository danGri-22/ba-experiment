# D.Griesser: file added

import torch
import torch.nn as nn
import numpy as np
import cv2
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from models.utils import extract_from_array, preprocess_data
from models.vgg_face import get_pretrained_model
from image_utils import save_image, show_image, compare_images
from art.attacks.evasion import ProjectedGradientDescent # will choose the correct version (e.g. pytorch) based on the classifier (e.g. PyTorchClassifier) used
from art.estimators.classification import PyTorchClassifier
import torch.optim as optim


if __name__ == "__main__":

    names = [line.rstrip('\n') for line in open('data/names.txt')]

    model = get_pretrained_model()
    model.eval()

    ### Adversarial Examples ###

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        nb_classes=2622,
        clip_values=(0, 255),
        input_shape=(1, 3, 224, 224)
    )

    pgd_attack = ProjectedGradientDescent(estimator=classifier, eps=200, eps_step=10, norm=np.inf, max_iter=100)

   ######################################################################################### 

    batch_size = 16
    
    dataloaders, dataset_sizes = preprocess_data(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    glass = cv2.imread('data/glasses/silhouette.png') # shape: (224, 224, 3)
    # print(glass.shape)
    glasses = transforms.ToTensor()(glass) # shape: (224, 224, 3) --> transforms.ToTensor() changes the shape of the np.array
    torch.manual_seed(12345)

    dataiter = iter(dataloaders)
    data = dataiter.next()

    images, labels = data
    images = images[:,[2,1,0],:,:]

    x = images.detach().numpy()
    y = labels.detach().numpy()
    mask = glasses.detach().numpy()

    x_adv = pgd_attack.generate(x=x, y=y, batch_size=batch_size, mask=mask)

    show_image(x_adv)

    sys.exit()

    ### use for experiment? ###

    torch.manual_seed(12345)

    cnt = 0
    for data in dataloaders['test']:
        print("Dataloaders: %d / %d" % (cnt + 1, len(dataloaders["test"])))
        cnt += 1

        images, labels = data
        
        # print("data[0] == images: ", data[0].shape, "data[1] == labels: ", data[1].shape)

        images = images[:,[2,1,0],:,:] #rgb to bgr

        images = images.to(device)
        labels = labels.to(device)
        glasses = glasses.to(device)    
   
        

