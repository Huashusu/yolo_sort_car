import xml.etree.ElementTree as ET
from os import getcwd

sets = ['trainval']

wd = getcwd()
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
gdut_classes = {'yellow': 0, 'white': 0, 'none': 1, 'blue': 0, 'red': 0}
# gdut_classes={'yellow':0, 'white':0, 'none':1, 'blue':0, 'red':0}
# helmet_classes={'hat': '1', 'head': '2', 'smoking': '3', 'none': '4'}

for image_set in sets:
    image_ids = open('../datasets/Helmet/ImageSets/Main/%s.txt' %
                     (image_set)).read().strip().split()
    # print(image_ids)
    list_file = open('ann.txt', 'w')
    for image_id in image_ids:
        in_file = open(f'../datasets/Helmet/Annotations/{image_id}.xml')
        # print(in_file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        if root.find('object') == None:
            print('none')
            continue

        list_file.write('../datasets/Helmet/JPEGImages/%s.jpg' % (image_id))
        for obj in tree.findall('object'):
            # print(obj)
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in gdut_classes or int(difficult) == 1:
                continue
            cls_id = gdut_classes[cls]
            # print(cls_id)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                 int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a)
                                            for a in b]) + ',' + str(cls_id))
        # print('11111')
        list_file.write('\n')
    list_file.close()
