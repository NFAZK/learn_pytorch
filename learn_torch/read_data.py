from torch.utils.data import Dataset
import cv2
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
        img = cv2.imread(self.img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

if __name__ == '__main__':
    root_dir = r"D:\jupyter_notebook_file\learn_torch\hymenoptera_data\hymenoptera_data\train"
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    ants_dataset = MyDataset(root_dir,ants_label_dir)
    bees_dataset = MyDataset(root_dir, bees_label_dir)
    # img,label = ants_dataset[0]
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    train_dataset = ants_dataset + bees_dataset