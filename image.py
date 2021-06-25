import cv2
import numpy as np
from PIL import Image

from yolo2 import YOLO

if __name__ == "__main__":
    model = YOLO()
    file_path = '2021-6-22.jpg'
    frame = cv2.imread(file_path, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # # 进行检测
    image, anns = model.detect_image(frame)
    frame = np.array(image)

    # print(track_bbs_ids)
    # # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imwrite('out_' + file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
