import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def create_holdout_set(source_dir, train_dir, holdout_dir, holdout_ratio=0.2):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(holdout_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if os.path.isdir(class_dir):
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(holdout_dir, class_name), exist_ok=True)

            files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            train_files, holdout_files = train_test_split(files, test_size=holdout_ratio)

            for file in train_files:
                shutil.move(os.path.join(class_dir, file), os.path.join(train_dir, class_name, file))

            for file in holdout_files:
                shutil.move(os.path.join(class_dir, file), os.path.join(holdout_dir, class_name, file))

# Set your directories here
source_dir = './imageSet'  # Replace with your source directory
train_dir = './train'
holdout_dir = './holdout'

create_holdout_set(source_dir, train_dir, holdout_dir)
