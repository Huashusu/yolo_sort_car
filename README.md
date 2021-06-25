## YOLOV4+sort实现的人员和车辆跟踪

代码还在整理中，注释什么的都不完善，但代码是可以直接执行的。

## 执行前

在执行前，下载预训练模型到```model_data```文件夹，先要安装pytorch pil opencv numpy等包，其他遗漏的看输出安装上即可。所有的包安装最新版本即可。比如pytorch在1.9上可以直接运行。
```
conda install opencv pytorch matplotlib numpy pillow
pip3 install filterpy
```

## 思路

检测是基于YOLOv4，跟踪基于SORT。

## 运行

1. 看视频效果，直接```python video.py``` 修改```video.py```中视频的文件路径
2. 看图片效果，使用```python image.py```修改```image.py```中图片的文件路径

## 训练自己的模型

1. 要训练自己的数据集，在model_data下新建自己的class文件。格式参考coco_class.txt。
2. 数据标注参考ann.txt中给出的例子。格式：image_name x1,y1,x2,y2,class_num
3. 都准备好了直接```python train.py```即可。
