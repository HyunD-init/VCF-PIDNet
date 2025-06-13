import os
from glob import glob
import shutil
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from generate_labels import convert_colored_mask

if __name__ == "__main__":
    data_root_path = '/Users/spinai_dev/Dropbox/006_researchdata/0005_Lat_Lxray_label'
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
    vcf_data_paths = glob(os.path.join(data_root_path, 'vcf', '*.jpg'))
    normal_data_paths = glob(os.path.join(data_root_path, 'Lat_Lxray_label', '*.jpg'))

    valid_test_ratio = 0.2
    test_ratio = 0.5
    random_state = 67

    vcf_train_path, vcf_valid_test_path = train_test_split(vcf_data_paths, test_size=valid_test_ratio, random_state=random_state, shuffle=True)
    vcf_valid_path, vcf_test_path = train_test_split(vcf_valid_test_path, test_size=test_ratio, random_state=random_state, shuffle=True)
    normal_train_path, normal_valid_test_path = train_test_split(normal_data_paths, test_size=valid_test_ratio, random_state=random_state, shuffle=True)
    normal_valid_path, normal_test_path = train_test_split(normal_valid_test_path, test_size=test_ratio, random_state=random_state, shuffle=True)

    orig_data = {
        'vcf':{
            'train':vcf_train_path,
            'valid':vcf_valid_path,
            'test':vcf_test_path,
        },
        'Lat_Lxray_label':{
            'train':normal_train_path,
            'valid':normal_valid_path,
            'test':normal_test_path,
        }
    }
    for orig_type, img_dataset in orig_data.items():
        print(f"[{orig_type}]")
        for training_mode, img_paths in img_dataset.items():
            print(f"-------[{training_mode}]------")
            dst_mask_vcf_direc = os.path.join(data_root_path, 'won_dataset', training_mode, 'masks_vcf')
            dst_mask_normal_direc = os.path.join(data_root_path, 'won_dataset', training_mode, 'masks_level')
            dst_img_direc = os.path.join(data_root_path, 'won_dataset', training_mode, 'images')
            os.makedirs(dst_mask_vcf_direc, exist_ok=True)
            os.makedirs(dst_mask_normal_direc, exist_ok=True)
            os.makedirs(dst_img_direc, exist_ok=True)
            for img_src_path in tqdm(img_paths):
                json_src_path = img_src_path.replace('.jpg', '.json')
                # shutil.copy(img_src_path, os.path.join(dst_img_direc, os.path.basename(img_src_path)))
                convert_colored_mask(json_src_path, dst_mask_vcf_direc, vcf_Label_Class, vcf_normal_class)
                convert_colored_mask(json_src_path, dst_mask_normal_direc, Label_Class)