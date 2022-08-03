#D.Griesser: extract_face.py added

from matplotlib import pyplot
from PIL import Image
from numpy import asarray
# from mtcnn.mtcnn import MTCNN
import torch
from torchvision import datasets, models, transforms
import cv2
import os
from models.dataset import FaceDataset
from torch.utils.data import DataLoader
 

def preprocess_data(batch_size=64):
	mean = [0.367035294117647,0.41083294117647057,0.5066129411764705]

	transform = transforms.Compose([
		transforms.Resize(size = (224,224)),
		# transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean, [1/255, 1/255, 1/255])
		])

	data_dir = "./data/experiment/images/"
	label_file = "./data/experiment/labels/labels.csv"

	dataset = FaceDataset(data_dir, label_file, transform=transform)
	
	dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

	dataset_size = len(dataset)
	class_names = dataset.class_names

	print(class_names)
	print(f"Dataset size: {dataset_size} entries")

	return dataloader, dataset_size


def data_process(batch_size=64):
    mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] # the mean value of vggface dataset 
    data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size = (224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    'experiment/images': transforms.Compose([
        transforms.Resize(size = (224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ])
    }
                                
    data_dir = './data/'   # change this if the data is in different loaction 
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test', 'experiment/images']} 


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True)
              for x in ['test', 'experiment/images']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['test', 'experiment/images']}
    class_names = image_datasets['experiment/images'].classes

    print(class_names)
    print(dataset_sizes)
    return dataloaders,dataset_sizes	


if __name__ == "__main__":
	# load the photo and extract the face
	pixels = extract_face('images/adam-driver.jpg')
	# plot the extracted face
	pyplot.imshow(pixels)
	# show the plot
	pyplot.show()
