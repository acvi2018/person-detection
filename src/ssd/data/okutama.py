import torch
from torch.utils.data import Dataset 
import pandas as pd
import cv2
import json

class OkutamaDataset(Dataset):
    
    def __init__(self, meta_data_file, frame_folder, ann_folder, ann_transform=None, img_transform=None, 
                 include_occluded=True, include_generated= True):
        self.meta_data = pd.read_csv(meta_data_file)
        self.frame_folder = frame_folder
        self.ann_folder = ann_folder
        self.include_occluded = include_occluded
        self.include_generated = include_generated
        self.ann_transform = ann_transform
        self.img_transform = img_transform
        
    def __len__(self):
        return self.meta_data.shape[0]
    
    def __getitem__(self, idx):
        img, anns, height, weight = self.pull_item(idx)
        return img, anns
        
        
    def pull_item(self, idx):
        valid_ann_list = []
        image_name = self.meta_data.iloc[idx]['img_file']
        ann_file = self.meta_data.iloc[idx]['ann_file']
        # Read image and json ann file
        with open('{0}/{1}'.format(self.ann_folder, ann_file), 'r') as jfile:
            ann_json = json.load(jfile)
        img = cv2.imread('{0}/{1}'.format(self.frame_folder, image_name))
        height, width, channels = img.shape
        
        # iterate through json and decide to include occluded and generated based on flags
        for i in range(len(ann_json)):
            if not self.include_occluded:
                if int(ann_json[i]['occluded']) == 1 :
                    continue
            if not self.include_generated:
                if int(ann_json[i]['generated']) == 1 :
                    continue
            valid_ann_list.append([ann_json[i]['xmin'], ann_json[i]['ymin'], 
                                  ann_json[i]['xmax'], ann_json[i]['ymax'], ann_json[i]['label']])
        
        # if annotation transform is present
        if self.ann_transform is not None:
            # To DO : converts label to label id and scale
            valid_ann_list = self.ann_transform(valid_ann_list, width, height)
        
        # if image transformation is present
        if self.img_transform is not None:
            valid_ann_list = np.array(valid_ann_list)
            img, boxes, labels = self.img_transform(img, valid_ann_list[:, :4], valid_ann_list[:, 4])
            # to rgb (Need to understand more....)
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            valid_ann_list = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return torch.from_numpy(img).permute(2, 0, 1), valid_ann_list, height, width
    
    def pull_image(self, idx):
        image_name = self.meta_data.iloc[idx]['img_file']
        return cv2.imread('{0}/{1}'.format(self.frame_folder, image_name), cv2.IMREAD_COLOR), image_name
    
    def pull_annotation(self, idx):
        ann_file = self.meta_data.iloc[idx]['ann_file']
        with open('{0}/{1}'.format(self.ann_folder, ann_file), 'r') as jfile:
            ann_json = json.load(jfile)
        valid_ann_list = []
        for i in range(len(ann_json)):
            if not self.include_occluded:
                if int(ann_json[i]['occluded']) == 1 :
                    continue
            if not self.include_generated:
                if int(ann_json[i]['generated']) == 1 :
                    continue
            valid_ann_list.append([ann_json[i]['xmin'], ann_json[i]['ymin'], 
                                  ann_json[i]['xmax'], ann_json[i]['ymax'], ann_json[i]['label']])
        return valid_ann_list
    
    def pull_tensor(self, idx):
        return torch.Tensor(self.pull_image(idx)).unsqueeze_(0)
    
    
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

class AnnotationTransform(object):
    
    def __init__(self, class_map):
        self.class_map = class_map
        
    def __call__(self, anns, width, height):
        '''
        * map each annotation with corresponding index from class_map
        * scale with height or width
        '''
        transformed_ann_list = []
        for index, ann in enumerate(anns):
            ann[0] = ann[0] / width
            ann[1] = ann[1] / height
            ann[2] = ann[2] / width
            ann[3] = ann[3] / height
            ann[4] = self.class_map[ann[4]]
            transformed_ann_list.append(ann)
            
        return transformed_ann_list