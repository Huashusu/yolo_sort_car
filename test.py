# import torch
# from torchsummary import summary
# from nets.CSPdarknet import darknet53
# from nets.yolo4 import YoloBody

# if __name__ == "__main__":
#     # 需要使用device来指定网络在GPU还是CPU运行
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = YoloBody(3,20).to(device)
#     summary(model, input_size=(3, 416, 416))

# import os
# classes_path = os.path.expanduser('model_data/helmet_classes.txt')
# with open(classes_path) as f:
#     class_names = f.readlines()
# class_names = [c.strip() for c in class_names]
# print(class_names) 

# print(gtss,dtss)
def IOU(boxA, boxB):
    # if boxes dont intersect
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # print(xA,yA,xB,yB)
    INVAILD = (xB - xA) < 0 or (yB - yA) < 0 or (xB == xA and yA == yB)
    if INVAILD:
        return 0
    union = (boxA[3] - boxA[1]) * (boxA[2] - boxA[0]) + (boxB[3] - boxB[1]) * (boxB[2] - boxB[0])
    interArea = (xB - xA) * (yB - yA)
    # print(union,interArea)
    iou = interArea / (union - interArea)
    return iou


iou = IOU([0, 0, 2, 2], [.5, .5, 2.5, 2.5])
print(iou)
