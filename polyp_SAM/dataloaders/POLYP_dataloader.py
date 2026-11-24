import torch
from torch.utils import data
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
import torchvision.transforms as transforms
import numpy as np

class POLYP_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=512, is_ori=False):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.is_ori = is_ori

        self.img_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()])
        if self.is_ori:
            self.ori_transform = transforms.Compose([
                transforms.Resize(self.target_size)])

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = os.path.join(self.root, self.img_list[item])
        label_file = os.path.join(self.root, self.label_list[item])
        img = Image.open(img_file).convert('RGB')
        label = Image.open(label_file).convert('L')
        if self.is_ori:
            img_ori = self.ori_transform(img)
            img_ori = torch.tensor(np.array(img_ori), dtype=torch.float32).permute(2, 0, 1)
        img = self.img_transform(img)
        label = self.gt_transform(label).to(torch.uint8).to(torch.float32)
        if self.is_ori:
            return img, label, img_file, img_ori
        else:
            return img, label, img_file

