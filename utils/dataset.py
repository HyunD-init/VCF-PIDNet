import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2
import os
from glob import glob
import random

#from utils import noise_injection

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "mask", "*.npy")))

    test_x = sorted(glob(os.path.join(path, "valid", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "valid", "mask", "*.npy")))
    return (train_x, train_y), (test_x, test_y)

class Custom_Dataset(data.Dataset):

    def __init__(self, data_path, label_path, size=(384, 384), mode='train', mean=0.0, std=0.05):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.resize = torchvision.transforms.Resize(size=(1024, 1024)) # aug
        self.size = size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        top_left_bottom_right = [random.randint(0, 1)*50  for a in range(4)]
        # if self.crop == "moveCrop":
        #     top_left_bottom_right[0] *= random.choice([-1, 1])
        #     top_left_bottom_right[1] *= random.choice([-1, 1])
        angle = random.randint(-1, 1)*10
        data = self.read_image(idx, angle, *top_left_bottom_right)
        label = self.read_mask(idx, angle, *top_left_bottom_right)

        edge = self.edgeGen(label)

        data = self.resize(data)
        label = self.resize(label)
        edge = cv2.resize(edge, (1024, 1024), interpolation=cv2.INTER_NEAREST)


        return data, label, edge
    
    def edgeGen(self, label):
        output = list()
        edge_size = 4
        kernel = np.ones((edge_size, edge_size), np.uint8)
        y_k_size = 6
        x_k_size = 6
        for i in range(label.shape[0]):
            edge = cv2.Canny(np.array(label[i, :,:]*255, dtype=np.uint8),  0.2, 10)
            
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
            edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0
            output.append(edge)

        return np.stack(output, axis=-1)
    

    def read_image(self, idx, angle, top, left, bottom, right):
        path = self.data_path[idx]
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, self.size)
        x = x / 255.0  # Normalize to [0, 1]
        x = x.astype(np.float32)  # Ensure data type
        x = x.transpose(2, 0, 1)  # Convert from HWC to CHW format expected by PyTorch
        x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor
        
        if self.mode=="train":
            #x = F.crop(x, top, left, self.size[0], self.size[1]) # move crop
            x = F.rotate(x, angle)
            # if self.crop == 'expandedCrop':
            x = F.resized_crop(x, top, left, self.size[0]-top-bottom, self.size[1]-left-right, self.size) # expanded crop
            # elif self.crop == "moveCrop":
            #     x = F.crop(x, top, left, self.size[0], self.size[1]) # move crop
            #x = noise_injection(x, mode="input",mean=self.mean, std=self.std)
        return x
        
    def read_mask(self, idx, angle, top, left, bottom, right):
        path = self.label_path[idx]
        class_mask = np.load(path)
        class_mask = cv2.resize(class_mask, self.size, interpolation=cv2.INTER_NEAREST)  # Resize mask
        class_mask = class_mask.astype(np.float32)  # Ensure data type
        class_mask = torch.tensor(class_mask, dtype=torch.int32)  # Convert to tensor
        class_mask = class_mask.permute(2, 0, 1)  # Add channel dimension if necessary
        
        if self.mode=="train":
            # if self.crop == 'expandedCrop':
            #class_mask = F.resized_crop(class_mask, top, left, self.size[0]-top-bottom, self.size[1]-left-right, self.size) # expanded crop
            # elif self.crop == 'moveCrop':
            #     class_mask = F.crop(class_mask, top, left, self.size[0], self.size[1]) # move crop
            #     if top<0:
            #         class_mask[0,:-top,:]=1
            #     elif top>0:
            #         class_mask[0, -top:,:]=1
            #     if left<0:
            #         class_mask[0,:,:-left]=1
            #     elif left>0:
            #         class_mask[0,:,-left:]=1
            fill = [1 if a == 0 else 0 for a in range(class_mask.size(0))]
            class_mask = F.rotate(class_mask, angle, fill=fill)
            class_mask = F.resized_crop(class_mask, top, left, self.size[0]-top-bottom, self.size[1]-left-right, self.size) # expanded crop
            
        return class_mask
