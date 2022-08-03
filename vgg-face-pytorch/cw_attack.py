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
from art.attacks.evasion import CarliniLInfMethod, CarliniL2Method, CarliniL0Method
from art.estimators.classification import PyTorchClassifier
import torch.optim as optim


########### currently not working ###########

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

    batch_size = 5
    torch.manual_seed(12345)                                              
    
    dataloaders, dataset_sizes = preprocess_data(batch_size)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)

    dataiter = iter(dataloaders)
    data = dataiter.next()

    images, labels = data
    images = images[:,[2,1,0],:,:]

    x = images.detach().numpy()
    y = labels.detach().numpy()
    '''
    images = cv2.imread("data/test/abbie_cornish/abbie_cornish_00000147.jpg")
    images = cv2.resize(images, (224, 224))
    images = np.transpose(images,(2, 0, 1)).astype(np.float32)
    x = images[np.newaxis, :, :, :]
    '''
    glass = cv2.imread('data/glasses/silhouette.png')
    glasses = transforms.ToTensor()(glass)
    
    glasses = torch.zeros_like(torch.from_numpy(x)) + glasses # should only work with L0-method

    mask = glasses.detach().numpy()  

    cw_l0 = CarliniL0Method(classifier=classifier,
                            confidence=1,
                            learning_rate=1.5,
                            # binary_search_steps=10,
                            max_iter=1,
                            initial_const=1e5,
                            batch_size=batch_size,
                            mask=mask) 


    x_adv = cw_l0.generate(x)

    show_image(x_adv)

    # difference = x_adv - x

    # show_image(difference)

