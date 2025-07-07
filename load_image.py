from utils.models import get_model
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import yaml

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Selcted device is {device}")
    return device


if __name__ == '__main__':
    info_path = '/Users/spinai_dev/Dropbox/012_업무상황기록/CWC/vcf-pidnet logs/ver2-pidnet_s-vcf_2/dropout_0_0_0/20250628_175144'

    with open(os.path.join(info_path, 'setting.yaml'), 'r', encoding='UTF8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = get_device()
    model = get_model(
        name=config['model_name'],
        model_config={
            **config,
            'model_parameters':{
                'class_num':config['class_num'],
                'vcf_class_num':config['vcf_class_num'],
                'vcf_mode':config['vcf_mode'],
                'p3': config['p3'],
                'p4': config['p4'],
                'p5': config['p5'],
            }
        },
        device=device
    )
    model_config = torch.load(os.path.join(info_path, 'model.pt'), device)
    model.load_state_dict(model_config['model'], strict=False)
    model.eval()
    print(f"Model is successfully loaded.: {next(model.parameters()).device}")

    test_direc = '/Users/spinai_dev/Dropbox/006_researchdata/0005_Lat_Lxray_label/won_dataset/test'
    level_palette = torch.tensor([
        [0, 0, 0, 0],            # label 0: 투명
        [255, 0, 0, 255],        # label 1: 빨강
        [0, 255, 0, 255],        # label 2: 초록
        [0, 0, 255, 255],        # label 3: 파랑
        [255, 255, 0, 255],      # label 4: 노랑
        [255, 0, 255, 255],      # label 5: 자홍
        [0, 255, 255, 255],      # label 6: 청록
        [128, 128, 0, 255],      # label 7: 올리브
        [128, 0, 128, 255],      # label 8: 보라
        [0, 128, 128, 255],      # label 9: 어두운 청록
        [128, 128, 128, 255],    # label 10: 회색
        [255, 165, 0, 255],      # label 11: 주황
    ], dtype=torch.uint8)

    vcf_palette = torch.tensor([
        [0, 0, 0, 0],            # label 0: 투명
        [139, 0, 0, 255],        # label 1: 진한 빨강 (dark red)
        [0, 100, 0, 255],        # label 2: 진한 초록 (dark green)
        [0, 0, 139, 255],        # label 3: 진한 파랑 (dark blue)
    ], dtype=torch.uint8)
    data_paths = glob(os.path.join(test_direc, 'images', '*.npy'))
    print(f"Total Test Image: {len(data_paths)}")
    level_score = {f'level-{i}':list() for i in range(11)}
    level_cnt = {f'level-{i}':0 for i in range(11)}
    vcf_score = {f'vcf-{i}':list() for i in range(4)}
    vcf_cnt = {f'vcf-{i}':0 for i in range(4)}
    eps = 1e-6
    
    with PdfPages('output.pdf') as pdf:
        for i, image_path in tqdm(enumerate(data_paths)):
            if i % 10 == 0:
                fig, ax = plt.subplots(10, 5, figsize=(15, 30))
                ax[0, 0].set_title('Orig')
                ax[0, 1].set_title('Level Pred')
                ax[0, 2].set_title('Level Label')
                ax[0, 3].set_title('VCF Pred')
                ax[0, 4].set_title('VCF Label')
            ax[i%10, 0].set_ylabel(i)
            level_label_path = image_path.replace('/images/', '/masks_level/')
            vcf_label_path = image_path.replace('/images/', '/masks_vcf/')
            img = np.load(image_path)
            for j in range(5):
                ax[i%10, j].set_xticks([])
                ax[i%10, j].set_yticks([])
                ax[i % 10, j].imshow(img)
            img = torch.tensor(np.moveaxis(img, -1, 0)).to(device=device, dtype=torch.float32)
            level = torch.tensor(cv2.resize(np.load(level_label_path), img.shape[1:], interpolation=cv2.INTER_NEAREST), dtype=torch.long)
            vcf = torch.tensor(cv2.resize(np.load(vcf_label_path), img.shape[1:], interpolation=cv2.INTER_NEAREST), dtype=torch.long)
            with torch.no_grad():
                pred = model(img.unsqueeze(0))
            
            level_pred = pred[1].squeeze(0).cpu().argmax(0)
            vcf_pred = pred[-1].squeeze(0).cpu().argmax(0)

            ax[i%10, 1].imshow(level_palette[level_pred])
            ax[i%10, 2].imshow(level_palette[level])
            ax[i%10, 3].imshow(level_palette[vcf_pred])
            ax[i%10, 4].imshow(level_palette[vcf])
            for idx in range(11):
                cur_level_pred = level_pred == idx
                cur_level = level == idx
                inter = cur_level & cur_level_pred
                union = cur_level | cur_level_pred
                level_score[f'level-{idx}'].append(((inter.sum() + eps) / (union.sum() + eps)).item())
            for idx in range(4):
                cur_vcf_pred = vcf_pred == idx
                cur_vcf = vcf == idx
                inter = cur_vcf & cur_vcf_pred
                union = cur_vcf | cur_vcf_pred
                vcf_score[f'vcf-{idx}'].append(((inter.sum() + eps) / (union.sum() + eps)).item())
            if i % 10 == 9 or i == len(data_paths) - 1:
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
    for idx in range(11):
        level_score[f'level-{idx}'].append(np.mean(level_score[f'level-{idx}']))
    for idx in range(4):
        print(np.mean(vcf_score[f'vcf-{idx}']))
        vcf_score[f'vcf-{idx}'].append(np.mean(vcf_score[f'vcf-{idx}']))
    pd.DataFrame(level_score).to_csv('level score.csv')
    pd.DataFrame(vcf_score).to_csv('vcf score.csv')