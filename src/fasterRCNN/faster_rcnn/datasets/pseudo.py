# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:42:58 2018

@author: cananthapatn
"""

class pascal_voc():
    def __init__(self):
        '''
        self._classes = ('__background__',  # always index 0
                 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor')
        '''
        self._classes = ('__background__',  # always index 0
                 'person')
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        print "here ", self._class_to_ind, self._classes, self.num_classes
        #cls = self._class_to_ind['__background__']
        obj = 'person'
        cls = int(obj == 'person')
        print "cls" , cls
      
name = pascal_voc()
