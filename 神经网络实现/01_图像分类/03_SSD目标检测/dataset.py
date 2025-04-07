# 数据加载

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

class VOCDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_ids = [f.split('.')[0] for f in os.listdir(os.path.join(root, 'JPEGImages'))]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(os.path.join(self.root, 'JPEGImages', f'{image_id}.jpg'))
        annotation = self._parse_xml(os.path.join(self.root, 'Annotations', f'{image_id}.xml'))
        
        boxes = torch.as_tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        
        if self.transform:
            image, boxes = self.transform(image, boxes)
        
        return image, {'boxes': boxes, 'labels': labels}

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
        return {'boxes': boxes}

    def __len__(self):
        return len(self.image_ids)