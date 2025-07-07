from glob import glob
from tqdm import tqdm
import os
import numpy as np

if __name__ == '__main__':
    dataset_direc = '/Users/spinai_dev/Dropbox/006_researchdata/0005_Lat_Lxray_label/won_dataset'

    for data_type in ['train', 'valid', 'test']:
        cur_dataset = os.path.join(dataset_direc, data_type)
        print(f"[{data_type}]")
        class_info = {f'{i}':0 for i in range(4)}
        for image_path in tqdm(glob(os.path.join(cur_dataset, 'images', '*.npy'))):
            vcf_mask_path = image_path.replace('/images/', '/masks_vcf/')
            vcf = np.load(vcf_mask_path)
            for idx in np.unique(vcf):
                class_info[str(idx)] += 1
        print(class_info)
        print('--'*30)