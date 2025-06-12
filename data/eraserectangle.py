import json
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

# Path to the train folder, relative to the current script
train_folder_path = r'C:\Users\taesh\cwc\dataset\dataset'

Label_Class = {
    'L5': 1, 'L4': 2, 'L3': 3, 'L2': 4, 'L1': 5, 'T12': 6,
    'T11': 7, 'T10': 8, 'T9': 9, 'S1':10, 'latSacrum': 11
}

# def move_files(root, mode):
#     yolo_path = os.path.join(root, mode, "YOLODataset")
#     x_path = glob(os.path.join(yolo_path, "images","*", "*.png"))
#     y_path = glob(os.path.join(yolo_path, "labels","*", "*.txt"))
#     for x in tqdm(x_path):
#         tmp_x = x.split("\\")
#         tmp_x.pop(-2)
#         shutil.move(x, "\\".join(tmp_x))
#     for y in tqdm(y_path):
#         tmp_y = y.split("\\")
#         tmp_y.pop(-2)
#         shutil.move(y, "\\".join(tmp_y))
#     check_x = sorted(glob(os.path.join(root, mode, "images", "*.png")))
#     check_y = sorted(glob(os.path.join(root, mode, "labels", "*.txt")))
#     for i, j in zip(check_x, check_y):
#         assert i.split("\\")[-1] == j.split("\\")
# for mode in [ "test"]:
#     move_files(train_folder_path, mode)


x = sorted(glob(os.path.join(train_folder_path, "*.jpg")))
y = sorted(glob(os.path.join(train_folder_path, "*.json")))
for _x, _y in zip(x, y):
    assert _x.split('.')[0] == _y.split('.')[0]
X_tv, X_test, Y_tv, Y_test = train_test_split(x, y, test_size=100, shuffle=True, random_state=67)
for _x, _y in zip(X_tv, Y_tv):
    assert _x.split('.')[0] == _y.split('.')[0]
for _x, _y in zip(X_test, Y_test):
    assert _x.split('.')[0] == _y.split('.')[0]

X_train, X_valid, Y_train, Y_valid = train_test_split(X_tv, Y_tv, test_size=0.1, shuffle=True, random_state=56)

for _x, _y in zip(X_train, Y_train):
    assert _x.split('.')[0] == _y.split('.')[0]
for _x, _y in zip(X_valid, Y_valid):
    assert _x.split('.')[0] == _y.split('.')[0]
print(len(X_train))
print(len(X_valid))
print(len(X_test))
def make_folder(root, mode, x, y):
    os.mkdir(os.path.join(root, mode))
    for x_path, y_path in zip(x, y):
        new_x_path = x_path.split('\\')
        new_x_path.insert(-1, mode)
        shutil.copyfile(x_path, "\\".join(new_x_path))

        new_y_path = y_path.split('\\')
        new_y_path.insert(-1, mode)
        shutil.copyfile(y_path, "\\".join(new_y_path))
#make_folder(train_folder_path, 'train', X_train, Y_train)
#make_folder(train_folder_path, 'valid', X_valid, Y_valid)
#make_folder(train_folder_path, 'test', X_test, Y_test)
for mode in ['train', 'valid', 'test']:
    y_names = glob(os.path.join(train_folder_path, mode, "*.json"))
    x_names = glob(os.path.join(train_folder_path, mode, "*.jpg"))
    print("{}: {}, {}".format(mode, len(x_names), len(y_names)))
"""
"""
# Iterate over each file in the train folder
for filename in tqdm(os.listdir(train_folder_path)):
    if filename.endswith('.json'):
        file_path = os.path.join(train_folder_path, filename)
        
        # Read and load JSON data from the file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Filter out shapes with a shape_type of 'rectangle'
        data['shapes'] = [shape for shape in data['shapes'] if (shape['shape_type'] != 'rectangle' and shape['label'] in Label_Class.keys())]

        # Write the modified data back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

print("All JSON files in the train folder have been updated.")

