import json, os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

def vcf_overlap_check(detected_vcf, shape):
    for vcf in detected_vcf:
        
        if shape['label'] == 'VCF':
            break
        vcf_array = np.array(vcf['points'])
        obj_array = np.array(shape['points'])
            
        if vcf_array.shape[0] == obj_array.shape[0]:
            dif_list = vcf_array - obj_array
            if (dif_list[0] == dif_list).all():
                return True
    return False

def convert_colored_mask(json_file, root, label_class, vcf_normal_class=None):
    name = os.path.basename(json_file).split('.')[0]
    with open(json_file, 'r') as f:
        all_labels = json.load(f)
    height = all_labels['imageHeight']
    width = all_labels['imageWidth']

    if 'shapes' not in all_labels:
        print(f"'shapes' key not found in {json_file}")
        return

    maskImage = np.zeros((height, width, len(label_class)), dtype=np.uint8)
    detected_vcf = list()
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
            elif vcf_overlap_check(detected_vcf, shape):
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
        np.save("{}.npy".format(os.path.join(root, name)), maskImage.argmax(-1).astype(np.uint8))
    except Exception as e:
        print(e)
        print(f"Failed to save colored mask image to {os.path.join(root, name)}")

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

