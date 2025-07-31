import json, os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

def vcf_overlap_check(detected_vcf, shape, vcf_info=None):
    for vcf in detected_vcf:
        
        if shape['label'] == 'VCF':
            break
        vcf_array = np.array(vcf['points'])
        obj_array = np.array(shape['points'])
            
        if vcf_array.shape[0] == obj_array.shape[0]:
            dif_list = vcf_array - obj_array
            if (dif_list[0] == dif_list).all():
                if vcf_info is not None:
                    vcf_info['vcf'].append(shape['label'])
                return True
    return False

def convert_colored_mask(json_file, root, label_class, vcf_normal_class=None):
    name = os.path.basename(json_file).split('.')[0]
    try:
        with open(json_file, 'r') as f:
            all_labels = json.load(f)
    except Exception as e:
        print(str(e))
        raise RuntimeError(json_file)
    height = all_labels['imageHeight']
    width = all_labels['imageWidth']

    if 'shapes' not in all_labels:
        print(f"'shapes' key not found in {json_file}")
        return

    maskImage = np.zeros((height, width, len(label_class)), dtype=np.uint8)
    detected_vcf = list()
    vcf_info = {'vcf':list()}
    if vcf_normal_class is not None:
        for shape in all_labels['shapes']:
            if shape['label'] == 'VCF':
                detected_vcf.append(shape)
    for shape in all_labels['shapes']:
        label = shape['label']
        if (vcf_normal_class is None) and (label not in label_class.keys()):
            continue
        elif (vcf_normal_class is not None):
            if not (label in label_class.keys() or label in vcf_normal_class):
                continue
            elif vcf_overlap_check(detected_vcf, shape, vcf_info):
                continue
        if vcf_normal_class is not None:
            label = shape['label']
            if label in vcf_normal_class:
                label = 'Normal'
        points = [(int(point[0]), int(point[1])) for point in shape['points']]
        points_ = np.array(points, dtype=np.int32)
        
        # 클래스별로 정의된 색상을 사용하여 다각형을 채웁니다.
        
        # Fill the polygon on the temporary mask
        channel = maskImage[:, :, label_class[label]].copy()
        cv2.fillPoly(channel, [points_], color=(1))
        maskImage[:, :, label_class[label]] = channel

    background_template = np.sum(maskImage, axis=-1)
    maskImage[:,:,0][background_template==0] = 1
    try:
        with open(os.path.join(root, f"{name}.json"), 'w') as f:
            json.dump(vcf_info, f)
        np.save("{}.npy".format(os.path.join(root, name)), maskImage.argmax(-1).astype(np.uint8))
    except Exception as e:
        print(e)
        print(vcf_info)
        print(f"Failed to save colored mask image to {os.path.join(root, name)}")
import albumentations as albu
import mclahe
import torchvision

class Preprocessing(object):

    def __init__(self, size=(1024, 1024)):
        self.resize = torchvision.transforms.Resize(size=(1024, 1024))
        albu_transform = [
            # albu.Normalize(normalization='min_max'), #mean= 3028.37, std = 1720.31, max_pixel_value=1.0),
            albu.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_CUBIC, p=1),
            albu.PadIfNeeded(border_mode=cv2.BORDER_CONSTANT, min_height=1024, min_width=1024, p=1)
        ]
        self.size = size
        self.vcf_transforms = albu.Compose(albu_transform)

    def preprocessing_image(self, path, save_direc):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        # ----------------------------------------------------------------
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # outlier clip
        x_cutoff_max = int(np.percentile(image, 99))
        image_clip = image.clip(0, x_cutoff_max)
        # normalization, resize
        sample = self.vcf_transforms(image=image_clip)
        image_transform = sample['image']
        # clahe
        #image_clahe = self.clahe(image_transform.astype(np.float32),True)
        np.save(
            os.path.join(save_direc, os.path.basename(path).replace('.jpg', '.npy')),
            image_transform.astype(np.float32)
        )
    @staticmethod
    def clahe(img, adaptive_hist_range=False):
        """
        input 1 numpy shape image (H x W x (D) x C)
        """
        temp = np.zeros_like(img)
        for idx in range(temp.shape[-1]):
            temp[...,idx] = mclahe.mclahe(img[...,idx], n_bins=128, clip_limit=0.04, adaptive_hist_range=adaptive_hist_range)
        return temp

if __name__ == '__main__':
    data_root_path = '/Users/spinai_dev/Dropbox/006_researchdata/0005_Lat_Lxray_label'
    save_root_path = '/Users/spinai_dev/Desktop/Spinai/Projects/VCF-PIDNet/data'

    # Update the Label_Class based on JSON annotations if needed
    Label_Class = {
        'background':0,
        'L5': 1, 'L4': 2, 'L3': 3, 'L2': 4, 'L1': 5, 'T12': 6,
        'T11': 7, 'T10': 8, 'T9': 9, 'latSacrum': 10
    }

    vcf_Label_Class = {
        'background':0,
        'Normal': 1, 'VCF': 2, 'latSacrum':3
    }
    vcf_normal_class = ['L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9']
    
    # vcf label - vcf object not contained
    save_path = os.path.join(save_root_path, 'vcf_mask')
    os.makedirs(save_path, exist_ok=True)
    normal_json_paths = glob(os.path.join(data_root_path, 'Lat_Lxray_label', '*.json'))
    for path in tqdm(normal_json_paths):
        convert_colored_mask(path, save_path, vcf_Label_Class, vcf_normal_class)
    
    # normal label - vcf object not contained
    save_path = os.path.join(save_root_path,'level_mask')
    os.makedirs(save_path, exist_ok=True)
    normal_json_paths = glob(os.path.join(data_root_path, 'Lat_Lxray_label', '*.json'))
    for path in tqdm(normal_json_paths):
        convert_colored_mask(path, save_path, Label_Class)

    # vcf label - vcf object contained
    save_path = os.path.join(save_root_path, 'vcf_mask')
    os.makedirs(save_path, exist_ok=True)
    vcf_json_paths = glob(os.path.join(data_root_path, 'vcf', '*.json'))
    for path in tqdm(vcf_json_paths):
        convert_colored_mask(path, save_path, vcf_Label_Class, vcf_normal_class)

    # normal label- vcf object contained
    save_path = os.path.join(save_root_path, 'level_mask')
    os.makedirs(save_path, exist_ok=True)
    vcf_json_paths = glob(os.path.join(data_root_path, 'vcf', '*.json'))
    for path in tqdm(vcf_json_paths):
        convert_colored_mask(path, save_path, Label_Class)

