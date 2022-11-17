import glob
import os
import subprocess
import uuid
import PIL.Image
import cv2
import torch.utils.data


class ImageClassificationDataSet(torch.utils.data.Dataset):
    def __init__(self, directory, categories, transform=None):
        self.categories = categories
        self.directory = directory
        self.transform = transform
        self._refresh()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, ann['category_index']

    def get_count(self, category):
        i = 0
        for a in self.annotations:
            if a['category'] == category:
                i += 1
        return i

    def _refresh(self):
        print("get actual data")
        self.annotations = []
        for category in self.categories:
            category_index = self.categories.index(category)
            # 透過split可把路徑分開，而透過join就可把路徑結合起來
            for image_path in glob.glob(os.path.join(self.directory, category, '*.jpg')):
                print(image_path)
                print(category_index)
                print(category)
                self.annotations += [{
                    'image_path': image_path,
                    'category_index': category_index,
                    'category': category
                }]