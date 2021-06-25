# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from nets.yolo4 import YoloBody
from nets.yolo_training import YOLOLoss, Generator

yolo_cls = 'youshang'
ext = '.pth'


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file
        3'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0
    for iteration in range(epoch_size):
        start_time = time.time()
        images, targets = next(gen)
        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(
                    images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(
                    torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(
                    images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(
                    torch.FloatTensor)) for ann in targets]
        # print(images)
        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        total_loss += loss
        waste_time = time.time() - start_time
        # print('\nEpoch:' + str(epoch+1) + '/' + str(Epoch))
        # print('iter:' + str(iteration) + '/' + str(epoch_size) +
        #       ' || Total Loss: %.4f || %.4fs/step' % (total_loss/(iteration+1), waste_time))

    print('Start Validation')
    for iteration in range(epoch_size_val):
        images_val, targets_val = next(genval)

        with torch.no_grad():
            if cuda:
                images_val = Variable(torch.from_numpy(
                    images_val).type(torch.FloatTensor)).cuda()
                targets_val = [Variable(torch.from_numpy(ann).type(
                    torch.FloatTensor)) for ann in targets_val]
            else:
                images_val = Variable(torch.from_numpy(
                    images_val).type(torch.FloatTensor))
                targets_val = [Variable(torch.from_numpy(ann).type(
                    torch.FloatTensor)) for ann in targets_val]
            optimizer.zero_grad()
            outputs = net(images_val)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets_val)
                losses.append(loss_item[0])
            loss = sum(losses)
            val_loss += loss
    print('Finish Validation')
    print('\nEpoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' %
          (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/%s/Epoch%d-Total_Loss%.4f-Val_Loss%.4f%s' %
               (yolo_cls, (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1), ext))
    files = glob.glob(f'logs/{yolo_cls}/*{ext}')
    if len(files) > 20:
        model_path = max(files, key=os.path.getctime)
        for f in files:
            if not model_path == f:
                os.remove(f)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    # -------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    # -------------------------------#
    input_shape = (608, 608)
    # -------------------------------#
    #   tricks的使用设置
    # -------------------------------#
    Cosine_lr = True
    mosaic = False
    # 用于设定是否使用cuda
    Cuda = True
    smoooth_label = 0
    # 
    annotation_paths = ['ann0126.txt']
    # -------------------------------#
    # 获得先验框和类
    # -------------------------------#
    anchors_path = f'model_data/yolo_anchors.txt'
    classes_path = f'model_data/{yolo_cls}_classes.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)

    # 创建模型
    model = YoloBody(len(anchors[0]), num_classes)
    import glob

    files = glob.glob(f'logs/{yolo_cls}/*{ext}')
    # files=glob.glob(f'logs/{yolo_cls}/old_best.pth')
    model_path = ''
    if len(files) > 0:
        model_path = max(files, key=os.path.getctime)
    # # 加快模型训练的效率
    print('Loading weights into state dict...')
    if os.path.exists(model_path):
        print(model_path)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items(
        ) if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes,
                                    (input_shape[1], input_shape[0]), smoooth_label, Cuda))

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    lines = []
    for annotation_path in annotation_paths:
        with open(annotation_path) as f:
            lines += f.readlines()
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    # print(len(lines),num_train,num_val)
    while True:
        lr = 1e-6
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 25

        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.9)

        gen = Generator(Batch_size, lines[:num_train],
                        (input_shape[0], input_shape[1])).generate(mosaic=mosaic)
        gen_val = Generator(Batch_size, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate(mosaic=False)

        epoch_size = int(max(1, num_train // Batch_size // 2.5)
                         ) if mosaic else max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size,
                          epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()

        # if True:
        lr = 1e-7
        Batch_size = 1
        Freeze_Epoch = 25
        Unfreeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.9)

        gen = Generator(Batch_size, lines[:num_train],
                        (input_shape[0], input_shape[1])).generate(mosaic=mosaic)
        gen_val = Generator(Batch_size, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate(mosaic=False)

        epoch_size = int(max(1, num_train // Batch_size // 2.5)
                         ) if mosaic else max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size,
                          epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()
        del lr_scheduler
        del optimizer
        del gen
        del gen_val
