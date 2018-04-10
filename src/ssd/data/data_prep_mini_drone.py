import cv2
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree
import sys
import pickle
import warnings
import shutil
import os
import pandas as pd
import re
from moviepy.video.io.VideoFileClip import VideoFileClip
import json

warnings.filterwarnings('ignore')

def convert_video_to_sequences(video_file_path, video_file_name, output_path):
    image_list = None
    if video_file_path is not None and output_path is not None:
        video = VideoFileClip(video_file_path)
        image_list = video.write_images_sequence('{0}/{1}_frame_%05d.jpg'.format(output_path, video_file_name), fps=None, verbose=True, withmask=True, progress_bar=True)
    return image_list

def get_frames_from_span(frame_dict, label):
    frames = {}
    if frame_dict:
        x = int(frame_dict['x'])
        y = int(frame_dict['y'])
        width = int(frame_dict['width'])
        height = int(frame_dict['height'])
        frame_span = frame_dict['framespan']
        
        for i in range(int(frame_span.split(':')[0]) , int(frame_span.split(':')[1])+1):
            if i not in frames:
                frames[i] = list()
            #frames[i].append((x-1, y-1, x + width-1,y + height-1,label))
            frames[i].append({'xmin':x-1, 'ymin':y-1, 'xmax':x + width-1, 'ymax':y + height-1, 'label': label })
    return frames

def parse_xgtf_ann_file(file_name, object_type): 
    master_frame_map = {}
    root = xml.etree.ElementTree.parse(file_name).getroot()
    for child in root:
        if 'data' in child.tag:
            for obj in child.getchildren()[0].getchildren():
                if 'object' in obj.tag and obj.attrib['name'] == object_type:
                    object_id = obj.attrib['id']
                    for attr in obj.getchildren():
                        if 'attribute' in attr.tag and (attr.attrib['name'] == 'box' or attr.attrib['name'] == 'obox'):
                            for bbox in attr.getchildren():
                                if 'bbox' in bbox.tag:
                                    frame_map = get_frames_from_span(bbox.attrib, object_type)
                                    # Check whether multiple persons are present in same frame
                                    if not master_frame_map:
                                        master_frame_map = frame_map
                                    else:
                                        for frame in frame_map.keys():
                                            if frame_map[frame] is not None:
                                                if frame not in master_frame_map:
                                                    master_frame_map[frame] = list()
                                                master_frame_map[frame].extend(frame_map[frame])
    return master_frame_map

def write_ann_files(video_file_name, frame_map, output_folder):
    for frame in frame_map.keys():
        # frame -1 because frame generation index is zero
        with open('{0}/{1}_frame_{2:05d}.json'.format(output_folder, video_file_name, int(frame)-1), 'w') as fp:
            json.dump(frame_map[frame], fp)

root_video_folder = '/home/vparambath/Desktop/iith/project/workpad/data/test/video'
root_annotation_folder = '/home/vparambath/Desktop/iith/project/workpad/data/test/video_annotation'

frame_output_folder = '/home/vparambath/Desktop/iith/project/workpad/data/test/frames'
ann_output_folder = '/home/vparambath/Desktop/iith/project/workpad/data/test/annotations'
root_folder = '/home/vparambath/Desktop/iith/project/workpad/data/test'

ann_ext = '.xgtf'
object_type= 'Person'

meta_data = pd.DataFrame(columns=['id', 'img_file', 'ann_file'])
file_counter = 0

for x in os.listdir(root_video_folder):
    video_file_name = x[0:x.rindex('.')]
    video_file_path = '{0}/{1}'.format(root_video_folder, x)
    ann_file = '{0}/{1}{2}'.format(root_annotation_folder, video_file_name, ann_ext)
    
    print('Processing {0}...'.format(x))
    image_list = convert_video_to_sequences(video_file_path, video_file_name, frame_output_folder)

    print('Creating frame level annotations...')
    frame_map = parse_xgtf_ann_file(ann_file, object_type)
    frames = np.array(frame_map.keys()) - 1
    write_ann_files(video_file_name, frame_map, ann_output_folder)
    
    meta_data_tup = []
    for index, file_path in enumerate(image_list):
        frame_number_search = re.search('.*_frame_(\d+).jpg', file_path[file_path.rindex('/')+1 :], re.IGNORECASE)
        if frame_number_search:
            if int(frame_number_search.group(1)) in frames:
                file_counter = file_counter + 1
                meta_data_tup.append((file_counter, file_path[file_path.rindex('/')+1 :],
                      '{0}.json'.format(file_path[file_path.rindex('/')+1 :file_path.rindex('.')])))
        
#     meta_data_tup = [(file_counter+index, file_path[file_path.rindex('/')+1 :],
#                       '{0}.json'.format(file_path[file_path.rindex('/')+1 :file_path.rindex('.')])) 
#                      for index, file_path in enumerate(image_list)]
    meta_data = meta_data.append(pd.DataFrame(meta_data_tup, columns=['id', 'img_file', 'ann_file']), ignore_index=True)
    
meta_data.to_csv('{0}/metadata.csv'.format(root_folder), index=False)
print('Finished processing....')