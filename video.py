import time

import cv2
import numpy as np
from PIL import Image

from yolo2 import YOLO

model = YOLO()
# 调用摄像头
# capture=cv2.VideoCapture(0) # 
i = 2
capture = cv2.VideoCapture('hbmz.mp4')
fps = 0.0
c = 0

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
out = cv2.VideoWriter('out_' + 'hbmz.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
FPS = capture.get(cv2.CAP_PROP_FPS)
while True:
    t1 = time.time()
    # 读取某一帧
    ref, frame = capture.read()
    if not ref:
        print('exit')
        break
    if not c % 5 == 0:
        c += 1
        continue
    # 格式转变，BGRtoRGB    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # # 进行检测
    image, anns = model.detect_image(frame)
    frame = np.array(image)

    # print(track_bbs_ids)
    # # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    fps = (fps + (1. / (time.time() - t1))) / 2
    print("fps= %.2f" % (fps))
    out.write(frame)
