import cv2
import matplotlib.pyplot as plt
import numpy as np
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

def plot_boundaries(img_path, x1, y1, x2, y2):
    img = cv2.imread(img_path)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
    plt.figure(figsize=(25,28))
    plt.imshow(gray)
    plt.show()

# ffmpeg -i ../1.1.1.mov -start_number 0 frame_%05d.jpg
def split_video(video_file, image_name_prefix):
    return subprocess.check_output('ffmpeg -i ' + os.path.abspath(video_file) + ' '+ image_name_prefix +'%d.jpg', shell=True, cwd=os.path.join(destination_path, 'JPEGImages'))\

def convert_video_to_sequences(video_file_path, video_file_name, output_path):
    image_list = None
    if video_file_path is not None and output_path is not None:
        video = VideoFileClip(video_file_path)
        image_list = video.write_images_sequence('{0}/{1}_frame_%05d.jpg'.format(output_path, video_file_name), fps=None, verbose=True, withmask=True, progress_bar=True)
    return image_list


# Create json file for a frame
def create_frame_level_annotations(annotation_file, video_file_name, output_path):
    # map {frame : list(annotations)}
    sep = ' '
    columns = ['track_id', 'xmin', 'ymin', 'xmax', 'ymax','frame','lost','occluded','generated','label','action']
    cols_req = ['track_id', 'xmin', 'ymin', 'xmax', 'ymax','lost','occluded','generated','label','action']
    annotation_df = pd.read_csv(annotation_file, sep=sep , names=columns, header=None)
    
    frames = annotation_df.frame.unique()
    for frame in frames:
        frame_df = annotation_df[np.logical_and(annotation_df.frame == frame, annotation_df.lost == 0)][cols_req]
        with open('{0}/{1}_frame_{2:05d}.json'.format(output_path, video_file_name, frame), 'w') as fp:
            json.dump(list(frame_df.T.to_dict().values()), fp)
    return frames



root_video_folder = '/home/vparambath/Desktop/iith/project/workpad/data/okutama_data/video'
root_annotation_folder = '/home/vparambath/Desktop/iith/project/workpad/data/okutama_data/video_annotation'
frame_output_folder = '/home/vparambath/Desktop/iith/project/workpad/data/okutama_data/frames'
ann_output_folder = '/home/vparambath/Desktop/iith/project/workpad/data/okutama_data/annotations'
root_folder = '/home/vparambath/Desktop/iith/project/workpad/data/okutama_data'
ann_ext = '.txt'

meta_data = pd.DataFrame(columns=['id', 'img_file', 'ann_file'])
file_counter = 0

for x in os.listdir(root_video_folder):
    video_file_name = x[0:x.rindex('.')]
    video_file_path = '{0}/{1}'.format(root_video_folder, x)
    ann_file = '{0}/{1}{2}'.format(root_annotation_folder, video_file_name, ann_ext)
    
    print('Processing {0}...'.format(x))
    image_list = convert_video_to_sequences(video_file_path, video_file_name, frame_output_folder)

    print('Creating frame level annotations...')
    frames = create_frame_level_annotations(ann_file, video_file_name, ann_output_folder)
    
    meta_data_tup = []
    for index, file_path in enumerate(image_list):
        frame_number_search = re.search('.*_frame_(\d+).jpg', file_path[file_path.rindex('/')+1 :], re.IGNORECASE)
        if frame_number_search:
            if int(frame_number_search.group(1)) in frames:
                meta_data_tup.append((file_counter, file_path[file_path.rindex('/')+1 :],
                      '{0}.json'.format(file_path[file_path.rindex('/')+1 :file_path.rindex('.')])))
                file_counter = file_counter + 1        
#     meta_data_tup = [(file_counter+index, file_path[file_path.rindex('/')+1 :],
#                       '{0}.json'.format(file_path[file_path.rindex('/')+1 :file_path.rindex('.')])) 
#                      for index, file_path in enumerate(image_list)]
    meta_data = meta_data.append(pd.DataFrame(meta_data_tup, columns=['id', 'img_file', 'ann_file']), ignore_index=True)
    
meta_data.to_csv('{0}/metadata.csv'.format(root_folder), index=False)
print('Finished processing....')
