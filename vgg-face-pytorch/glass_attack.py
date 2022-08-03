import torch
import torch.nn as nn
import torch.optim as optim
from models.utils import preprocess_data
import numpy as np
import argparse
import torchvision
import cv2
from torchvision import datasets, models, transforms
import sys
import numpy
from models.vgg_face import get_pretrained_model, get_prediction
import matplotlib.pyplot as plt # D.Griesser: import added
from image_utils import save_image, show_image  #
import torch.nn.functional as F
#uncomment to see some images 
numpy.set_printoptions(threshold=sys.maxsize)



def choose_color(model,X,y,glass,mean):
    model.eval()
    potential_starting_color0 = [128,220,160,200,220]
    potential_starting_color1 = [128,130,105,175,210]
    potential_starting_color2 = [128,  0, 55, 30, 50]

    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(X)
     
    
    for i in range(len(potential_starting_color0)):
        delta1 = torch.zeros(X.size()).to(y.device)

        delta1[:,0,:,:] = glass[0,:,:]*potential_starting_color2[i]
        delta1[:,1,:,:] = glass[1,:,:]*potential_starting_color1[i]
        delta1[:,2,:,:] = glass[2,:,:]*potential_starting_color0[i]

        # X + delta1 --> applies new color to eyeglass frame (frame is already added to the image)
        all_loss = nn.CrossEntropyLoss(reduction='none')(model(X+delta1-mean),y)
        
        max_delta[all_loss >= max_loss] = delta1.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

        print("choose_color() Iteration %d / %d" % (i + 1, len(potential_starting_color0))) # D.Griesser: print-statement added
        
        

    print("choose_color() finished") # D.Griesser: print-statement added
 
    # returns colored glass silhouttes for the whole batch
    return max_delta


def glass_attack(model, X, y, glass, alpha=1, num_iter=20 ,momentum=0.4, y_target=None):
    """ Construct glass frame adversarial examples on the examples X"""

    targeted = y_target is not None

    model.eval()
    mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
    de = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = mean.to(de)
    X1 = torch.zeros_like(X,requires_grad = True)
    X1.data = (X+mean)*(1-glass) # add glasses to images

    with torch.set_grad_enabled(False):
        print("glass_attack() calls choose_color()") # D.Griesser: print-statement added
        color_glass = choose_color(model,X1,y,glass,mean) # sole purpose is to pick a starting color
        
    with torch.set_grad_enabled(True):
        X1.data = X1.data+color_glass-mean
        delta =torch.zeros_like(X)
        
        #D.Griesser: change this area (incorporate the different algorithms)? 
        for t in range(num_iter):
            print("glass_attack() Iteration %d / %d" % (t + 1, num_iter)) # D.Griesser: print-statement added
            loss = nn.CrossEntropyLoss()(model(X1), y_target if targeted else y)
            loss.backward()

            delta_change =  X1.grad.detach()*glass
            max_val,indice = torch.max(torch.abs(delta_change.view(delta_change.shape[0], -1)),1)
            r = alpha * delta_change /max_val[:,None,None,None]

            if t == 0:
                delta.data = r

            else:
                if targeted:
                    delta.data = momentum * delta.detach() - r
                else:
                    delta.data = momentum * delta.detach() + r

            delta.data[(delta.detach() + X1.detach() + mean) > 255] = 0 
            delta.data[(delta.detach() + X1.detach() + mean) < 0 ] = 0 

            X1.data = (X1.detach() + delta.detach())
            X1.data = torch.round(X1.detach()+mean) - mean
            X1.grad.zero_()
           
        print("glass_attack() finished") # D.Griesser: print-statement added
        return (X1).detach()


if __name__ == "__main__":

    names = [line.rstrip('\n') for line in open('data/names.txt')]
 
    alpha   = 20
    iterations = 50
    batch_size = 10

    glass = cv2.imread('data/glasses/silhouette.png')

    glasses = transforms.ToTensor()(glass)

    model = get_pretrained_model()

    model.eval()

    # torch.manual_seed(1234)
    
    dataloaders, dataset_sizes = preprocess_data(batch_size)

    dataiter = iter(dataloaders)
    data = dataiter.next()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    images, labels = data

    images = images[:,[2,1,0],:,:] #rgb to bgr

    model.to(device)
    images.to(device)
    labels.to(device)
    glasses.to(device)
    print(images.device)

    values, indices = get_prediction(images, model)

    predictions = [(names[index], value) for index, value in zip(indices.tolist(), values.tolist())]
    ground_truth = [names[label] for label in labels.tolist()]   

    print(f"Predictions: {predictions}")
    print(f"Labels: {ground_truth}")

    # show_image(images)

    target = torch.full((batch_size,), 3) # target class = 3
    
    target.to(device)

    x_adv = glass_attack(
        model=model,
        X=images,
        y=labels,
        glass=glasses,
        alpha=alpha,
        num_iter=iterations,
        y_target=target
    )
    
    print("Expectation: %s" %names[target[0].item()])

    values, indices = get_prediction(x_adv, model)
    
    predictions = [(names[index], value) for index, value in zip(indices.tolist(), values.tolist())]
    print(f"Predictions: {predictions}")

    show_image(x_adv)
    
    