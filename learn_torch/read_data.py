from torch.utils.data import Dataset
from cv2 import imread
import os



class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        self.img_name = self.img_path[idx]
        self.img_item_path = os.path.join(self.root_dir,self.label_dir,self.img_name)
        img = imread(self.img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

if '__name__' == '__main__':
    root_dir = "learn_torch/练手数据集/hymenoptera_data/hymenoptera_data/train"
    label_dir = "ants"
    ants_dataset = MyDataset(root_dir,label_dir)