import json, os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

# Update the Label_Class based on JSON annotations if needed
Label_Class = {
    'background':0,
    'L5': 1, 'L4': 2, 'L3': 3, 'L2': 4, 'L1': 5, 'T12': 6,
    'T11': 7, 'T10': 8, 'T9': 9, 'S1':10, 'latSacrum': 11
}


def convert_colored_mask(json_file, root, name):
    with open(json_file, 'r') as f:
        all_labels = json.load(f)
    height = all_labels['imageHeight']
    width = all_labels['imageWidth']

    if 'shapes' not in all_labels:
        print(f"'shapes' key not found in {json_file}")
        return

    maskImage = np.zeros((height, width, len(Label_Class)), dtype=np.uint8)

    for shape in all_labels['shapes']:
        label = shape['label']
        if label not in Label_Class.keys():
            print(f"Label {label} not found in Label_Class dictionary")
            return
        points = [(int(point[0]), int(point[1])) for point in shape['points']]
        points_ = np.array(points, dtype=np.int32)
        
        # 클래스별로 정의된 색상을 사용하여 다각형을 채웁니다.
        # Create a temporary single-channel mask for the current class
        temp_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill the polygon on the temporary mask
        cv2.fillPoly(temp_mask, [points_], color=(1))
        
        # Assign the filled temporary mask to the corresponding channel in the maskImage
        maskImage[:, :, Label_Class[label]] = temp_mask
    background_template = np.sum(maskImage, axis=-1)
    maskImage[:,:,0][background_template==0] = 1
    try:
        np.save("{}.npy".format(os.path.join(root, name)), maskImage)
    except Exception as e:
        print(e)
        print(f"Failed to save colored mask image to {os.path.join(root, name)}")

root = r'C:\Users\taesh\cwc\dataset\dataset'

# train mask

train_mask_root = os.path.join(root, 'train', 'mask')
os.mkdir(train_mask_root)
train_json_paths = glob(os.path.join(root, 'train', '*.json'))
for path in tqdm(train_json_paths):
    convert_colored_mask(path, train_mask_root, path.split('\\')[-1].split('.')[0])

# valid mask

valid_mask_root = os.path.join(root, 'valid', 'mask')
os.mkdir(valid_mask_root)
valid_json_paths = glob(os.path.join(root, 'valid', '*.json'))
for path in tqdm(valid_json_paths):
    convert_colored_mask(path, valid_mask_root, path.split('\\')[-1].split('.')[0])

# test mask

test_mask_root = os.path.join(root, 'test', 'mask')
os.mkdir(test_mask_root)
test_json_paths = glob(os.path.join(root, 'test', '*.json'))
for path in tqdm(test_json_paths):
    convert_colored_mask(path, test_mask_root, path.split('\\')[-1].split('.')[0])