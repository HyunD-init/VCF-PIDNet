import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as Ft
import numpy as np
import cv2
import os
import json
from glob import glob
import random
import albumentations as albu
import mclahe

#from utils import noise_injection

def load_data(path):
    train_x = glob(os.path.join(path, "train", "images", "*.npy"))
    valid_x = glob(os.path.join(path, "valid", "images", "*.npy"))
    return train_x, valid_x

class Custom_Dataset(data.Dataset):

    def __init__(self, data_path, size=(384, 384), mode='train', class_num=11, edge_pad=False):
        self.data_path = data_path
        self.mode = mode
        self.resize = torchvision.transforms.Resize(size=(1024, 1024)) # aug
        albu_transform = [
            # albu.Normalize(normalization='image'),#mean= 3028.37, std = 1720.31, max_pixel_value=1.0),
            albu.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_CUBIC, p=1),
            albu.PadIfNeeded(border_mode=cv2.BORDER_CONSTANT, min_height=1024, min_width=1024, p=1)
        ]
        self.vcf_transforms = albu.Compose(albu_transform)
        self.size = size
        self.class_num = class_num
        self.edge_pad = edge_pad

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx): #랜덤으로 잘라서 augment
        top_left_bottom_right = [random.randint(-2, 2)*50  for a in range(4)]
        # if self.crop == "moveCrop":
        #     top_left_bottom_right[0] *= random.choice([-1, 1])
        #     top_left_bottom_right[1] *= random.choice([-1, 1])
        angle = random.randint(-1, 1)*10
        
        data = self.read_image(idx, angle, *top_left_bottom_right)
        # vcf_label = self.read_mask(idx, angle, *top_left_bottom_right, label_type='vcf').squeeze(0)
        level_label = self.read_mask(idx, angle, *top_left_bottom_right, label_type='level').squeeze(0)
        vcf_cls = self.read_cls(idx)


        edge = self.edgeGen(level_label)


        return (data, (level_label, vcf_cls), edge)
    def read_cls(self, idx):
        Label_Class = {
            'L5': 1, 'L4': 2, 'L3': 3, 'L2': 4, 'L1': 5, 'T12': 6,
            'T11': 7, 'T10': 8, 'T9': 9
        }
        path = self.data_path[idx].replace('images', f'masks_vcf').replace('.npy', '.json')
        with open(path, 'r') as f:
            json_file = json.load(f)
        if len(json_file['vcf']) == 0:
            return torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            detected_vcf = torch.tensor([Label_Class[k] for k in json_file['vcf']])
            detected_vcf = Ft.one_hot(detected_vcf, num_classes=len(Label_Class) + 1).sum(dim=0)

            return detected_vcf
        
    def edgeGen(self, label):

        edge_size = 4
        y_k_size = 6
        x_k_size = 6

        #return np.stack(output, axis=-1)
    
        edge = cv2.Canny(np.array(label, dtype=np.uint8), 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if self.edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0
        return edge
    

    def read_image(self, idx, angle, top, left, bottom, right):
        path = self.data_path[idx]
        # ---------------------
        image_clahe = np.load(path)
        # image = cv2.imread(path, cv2.IMREAD_COLOR)
        # image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        # # ----------------------------------------------------------------
        # if len(image.shape) == 2:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # # outlier clip
        # x_cutoff_max = int(np.percentile(image, 99))
        # image_clip = image.clip(0, x_cutoff_max)
        # image_clip = image_clip.astype(np.float32) / 255.0
        # # normalization, resize
        # sample = self.vcf_transforms(image=image_clip)
        # image_transform = sample['image']
        # # clahe
        # image_clahe = self.clahe(image_transform.astype(np.uint8),True)
        image_clahe = np.moveaxis(image_clahe,-1,0)
        x = torch.tensor(image_clahe)
        # ----------------------------------------------------------------
        # x = cv2.resize(x, self.size)
        # x = x / 255.0  # Normalize to [0, 1]
        # x = x.astype(np.float32)  # Ensure data type
        # x = x.transpose(2, 0, 1)  # Convert from HWC to CHW format expected by PyTorch
        # x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor
        
        if self.mode=="train":
            #x = F.crop(x, top, left, self.size[0], self.size[1]) # move crop
            x = F.rotate(x, angle)
            # if self.crop == 'expandedCrop':
            x = F.resized_crop(x, top, left, self.size[0]-top-bottom, self.size[1]-left-right, self.size) # expanded crop
            # elif self.crop == "moveCrop":
            #     x = F.crop(x, top, left, self.size[0], self.size[1]) # move crop
            #x = noise_injection(x, mode="input",mean=self.mean, std=self.std)
        return x
    @staticmethod
    def clahe(img, adaptive_hist_range=False):
        """
        input 1 numpy shape image (H x W x (D) x C)
        """
        temp = np.zeros_like(img)
        for idx in range(temp.shape[-1]):
            temp[...,idx] = mclahe.mclahe(img[...,idx], n_bins=128, clip_limit=0.04, adaptive_hist_range=adaptive_hist_range)
        return temp
    def read_mask(self, idx, angle, top, left, bottom, right, label_type='vcf'):
        path = os.path.join(self.data_path[idx].replace('images', f'masks_{label_type}'))
        class_mask = np.load(path)
        class_mask = cv2.resize(class_mask, self.size, interpolation=cv2.INTER_NEAREST)  # Resize mask
        class_mask = class_mask.astype(np.float32)  # Ensure data type
        class_mask = torch.tensor(class_mask)  # Convert to tensor
        if len(class_mask.shape) == 3:
            class_mask = class_mask.permute(2, 0, 1)  # Add channel dimension if necessary
        elif len(class_mask.shape) == 2:
            class_mask = class_mask.unsqueeze(0)

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
            class_mask = F.rotate(class_mask, angle)
            class_mask = F.resized_crop(class_mask, top, left, self.size[0]-top-bottom, self.size[1]-left-right, self.size, interpolation=cv2.INTER_NEAREST) # expanded crop
            
        return class_mask
