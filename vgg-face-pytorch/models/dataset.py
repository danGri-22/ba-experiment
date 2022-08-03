# D.Griesser: file added

from torch.utils.data import Dataset
import pandas as pd
import glob
from PIL import Image
from torchvision.io import read_image


class FaceDataset(Dataset):

    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(annotations_file, header=0)
        self.transform = transform
        file_list = sorted(glob.glob(self.img_dir + "*"))
        # print(file_list)

        self.data = []
        self.class_names = []
        for path in file_list:
            class_name = path.split("/")[-1]
            # print(class_name)
            self.class_names.append(class_name)
            for img_path in glob.glob(path + "/*.jpg"):
                # print(img_path)
                self.data.append(img_path)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path)
       
        label = self.labels.iloc[idx, 1]
        
        if self.transform:
            img = self.transform(img)
        return img, label 

if __name__ == "__main__":

    dataset = FaceDataset("./data/experiment/images/", "./data/experiment/labels/labels.csv")
    
    img, label = dataset[800]
    
    img.show()