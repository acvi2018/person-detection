import json
import xmltodict
import pandas as pd
import os

def parse_ann_file(file_path, object_type):
    width = None
    height = None
    depth = None
    person_anns = []

    with open(file_path) as fd:
        doc = xmltodict.parse(fd.read())

        width = int(doc['annotation']['size']['width'])
        height = int(doc['annotation']['size']['height'])
        depth = int(doc['annotation']['size']['depth'])
        
        object_list = doc['annotation']['object']
        if type(object_list).__name__ != 'OrderedDict':
            for obj in object_list:
                if obj['name'] == object_type:
                    person_anns.append({'difficult' : int(obj['difficult']), 'truncated' : int(obj['truncated']), 
                                   'xmin' : int(obj['bndbox']['xmin']) -1, 
                                    'ymin' : int(obj['bndbox']['ymin']) -1,
                                   'xmax' : int(obj['bndbox']['xmax']) -1, 
                                    'ymax' : int(obj['bndbox']['ymax']) -1, 
                                    'label': object_type})
        else:
            if object_list['name'] == object_type:
                person_anns.append({'difficult' : int(object_list['difficult']), 'truncated' : int(object_list['truncated']), 
                                   'xmin' : int(object_list['bndbox']['xmin']) -1, 
                                    'ymin' : int(object_list['bndbox']['ymin']) -1,
                                   'xmax' : int(object_list['bndbox']['xmax']) -1, 
                                    'ymax' : int(object_list['bndbox']['ymax']) -1, 
                                    'label': object_type})
    return(person_anns, width, height, depth)

ann_folder = '/home/vijin/iith/project/data/VOCdevkit/VOC2012/Annotations'
object_file = '/home/vijin/iith/project/data/VOCdevkit/VOC2012/ImageSets/Main/person_test.txt'
ann_out_folder = '/home/vijin/iith/project/data/VOCdevkit/VOC2012/Person_Annotations_test'
root_folder = '/home/vijin/iith/project/data/VOCdevkit/VOC2012'

data = pd.read_csv(object_file, header=None, sep='\s+', names=['image_name', 'type'], encoding='gbk', dtype={'image_name':'str', 'type' : 'float'})
print(data['image_name'])
counter = 1
meta_data_tup = []
#for img in list(data[data.type >= 0]['image_name']):
for img in list(data['image_name']): # changes to create test annotations for VOC 2012
	print('processing {0}.....'.format(img))
	if os.path.exists('{0}/{1}.xml'.format(ann_folder, img)):
		print('processing {0}.....'.format(img))
		anns, width, height, depth = parse_ann_file('{0}/{1}.xml'.format(ann_folder, img), 'person')
		if len(anns) > 0:
			with open('{0}/{1}.json'.format(ann_out_folder, img,), 'w') as fp:
				json.dump(anns, fp)
			meta_data_tup.append((counter, '{0}.jpg'.format(img), '{0}.json'.format(img)))
			counter = counter + 1
meta_data = pd.DataFrame(meta_data_tup, columns=['id', 'img_file', 'ann_file'])
meta_data.to_csv('{0}/test_metadata.csv'.format(root_folder), index=False)


