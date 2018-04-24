import os
import sys
module_path = os.path.abspath(os.path.join('/home/vijin/iith/project/workpad/ssd.pytorch'))
if module_path not in sys.path:
    sys.path.append(module_path)

import cv2
import pickle
import numpy as np

from data import v2, v1, AnnotationTransform, MiniDroneDataset, detection_collate, OkutamaDataset

def save_annotated_image(img, image_name, ground_truth, predictions, root_folder):
	for box in ground_truth:
		cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
	for box in predictions:
		cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
	cv2.imwrite('{0}/{1}'.format(root_folder, image_name),img)



root_folder = '/home/vijin/iith/project/workpad/results/2class_SSD300_mini_drone'
annotated_img_folder = '{0}/predicted_images'.format(root_folder)

with open('{0}/pred_details.pkl'.format(root_folder), 'rb') as fp:
	pred_dict = pickle.load(fp)

testset = MiniDroneDataset('//home/vijin/iith/project/data/mini-drone-data/DroneProtect-testing-set/metadata.csv', 
    '/home/vijin/iith/project/data/mini-drone-data/DroneProtect-testing-set/frames', 
    '/home/vijin/iith/project/data/mini-drone-data/DroneProtect-testing-set/annotations')

num_images = len(pred_dict.keys())
for idx in pred_dict.keys():
	print('Annotating image {:d}/{:d}....'.format(idx+1, num_images))
	img, image_name = testset.pull_image(idx)
	ground_truth = pred_dict[idx]['ground_truth']
	predictions = pred_dict[idx]['prediction']
	save_annotated_image(img, image_name, ground_truth, predictions, annotated_img_folder)
