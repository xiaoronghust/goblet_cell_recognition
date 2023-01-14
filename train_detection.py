import os 
import json
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from models.yolo4_tiny import yolo_body
from models.loss import yolo_loss
from keras.backend.tensorflow_backend import set_session
from utils.utils import get_random_data,get_random_data_with_Mosaic,rand,WarmUpCosineDecayScheduler

####################### 全局变量 #######################
DATA_DIR = 'dataset/CellDetection/goblet_epithelium/'#E:/Detection/LM_Dataset/Detection/Pathology/'
classes_path = 'model_data/voc_classes.txt'    
anchors_path = 'model_data/yolo_anchors.txt'
weights_path = 'model_data/yolov4_tiny_voc.h5'
val_split = 0.2
freeze_layers = 60
########################################################

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()


    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

#---------------------------------------------------#
#   训练数据生成器
#---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(config['batch_size']):
            if i==0:
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i+4) < n:
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i+4], input_shape, max_boxes=100)
                    i = (i+4) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape, max_boxes=100)
                    i = (i+1) % n
                flag = bool(1-flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape, max_boxes=100)
                i = (i+1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(config['batch_size'])

#---------------------------------------------------#
#   读入xml文件，并输出y_true
#---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = 2#len(anchors)//3
    # 先验框
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32') # 416,416
    # 读出xy轴，读出长宽
    # 中心点(m,n,2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 计算比例
    true_boxes[..., 0:2] = boxes_xy/input_shape[:]
    true_boxes[..., 2:4] = boxes_wh/input_shape[:]

    # m张图
    m = true_boxes.shape[0]
    # 得到网格的shape为13,13;26,26;
    grid_shapes = [input_shape//{0:4, 1:2, 2:1}[l] for l in range(num_layers)]
    # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    # [1,9,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # 长宽要大于0才有效
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 计算真实框和哪个先验框最契合
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # 维度是(n) 感谢 消尽不死鸟 的提醒
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # floor用于向下取整
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    # 找到真实框在特征层l中第b副图像对应的位置
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true
# --------------------------------------------------#
# 保存训练参数 #
def save_config(config, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + 'config.json', 'w') as f:
        f.write(json.dumps(config))

def get_generators(config, anchors, num_classes, lines, num_train):
    return data_generator(lines[:num_train], config['batch_size'], config['input_shape'], anchors, num_classes, mosaic=config['mosaic']),\
        data_generator(lines[num_train:], config['batch_size'], config['input_shape'], anchors, num_classes, mosaic=False)

def get_data(annotation_path, val_split):
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    return lines, num_val, num_train

if __name__ == "__main__":
    # TRAIN INFO 
    log_dir = 'logs/goblet_epithelium/chn8_small/'
    annotation_path = DATA_DIR + '2007_train.txt'

    config = dict()
    config['input_shape'] = (256, 256)
    config['mosaic'] = False
    config['Cosine_scheduler'] = False
    config['label_smoothing'] = 0
    config['featrue_strides'] = {0:4, 1:2}
    config['anchors'] = get_anchors(anchors_path).tolist()
    config['class_names'] = get_classes(classes_path)
    config['pretrain'] = False
    config['batch_size'] = 4
    config['learning_rate_base'] = 1e-3
    config['Freeze_epoch'] = 0
    config['Epoch'] = 200
    config['trainsteps'] = 50
    config['input_chn'] = 16
    save_config(config, log_dir)

    anchors = get_anchors(anchors_path)
    num_classes = len(config['class_names'])
    num_anchors = len(anchors)

    print('classes numbers:', num_classes)

    lines, num_val, num_train = get_data(annotation_path, val_split)
    traingen, valgen = get_generators(config, anchors, num_classes, lines, num_train)

    ########################## 构建模型和训练 ##########################
    K.clear_session()
    h, w = config['input_shape']
    image_input = Input(shape=(h, w, 3))
    print('Create YOLOv4-Tiny model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors//2, num_classes, config['input_chn'])
    model_body.summary()

    ########################## 载入预训练权重 ##########################
    if config['pretrain']:
        print('Load weights {}.'.format(weights_path))
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    y_true = [Input(shape=(h//config['featrue_strides'][l], w//config['featrue_strides'][l], num_anchors//2, num_classes+5)) for l in range(2)]

    loss_input = [*model_body.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.20, 'label_smoothing': config['label_smoothing']})(loss_input)
    model = Model([model_body.input, *y_true], model_loss)

    ########################## get callback ###########################
    logging = TensorBoard(log_dir=log_dir)
    # checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    #     monitor='val_loss', save_weights_only= False, save_best_only=False, period= 1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta= 1, patience= 30, verbose=1, restore_best_weights= True)
    if config['Cosine_scheduler']:
        # 预热期
        warmup_epoch = int((Freeze_epoch-0)*0.2)
        # 总共的步长
        total_steps = int((Freeze_epoch-0) * num_train / config['batch_size'])
        # 预热步长
        warmup_steps = int(warmup_epoch * num_train / config['batch_size'])
        # 学习率
        reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                    total_steps=total_steps,
                                                    warmup_learning_rate=1e-4,
                                                    warmup_steps=warmup_steps,
                                                    hold_base_rate_steps=num_train,
                                                    min_learn_rate=1e-6
                                                    )
        model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    else:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience= 5, verbose=1)
    callbacks = [logging, early_stopping, reduce_lr]#, checkpoint
    ###################################################################

    if (config['Freeze_epoch'] > 0):
        for i in range(freeze_layers): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
        model.compile(optimizer=Adam(config['learning_rate_base']), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, config['batch_size']))
        model.fit_generator(traingen,
            steps_per_epoch= config['trainsteps'],
            validation_data= valgen,
            validation_steps=max(1, num_val//config['batch_size']),
            epochs= config['Freeze_epoch'],
            initial_epoch= 0,
            callbacks=callbacks)
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
    else:
        for i in range(freeze_layers): model_body.layers[i].trainable = True
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, config['batch_size']))
        model.compile(optimizer=Adam(config['learning_rate_base']), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        model.fit_generator(traingen,
            steps_per_epoch= config['trainsteps'],
            validation_data= valgen,
            validation_steps=max(1, num_val//config['batch_size']),
            epochs= config['Epoch'],
            initial_epoch= config['Freeze_epoch'],
            callbacks=callbacks,
            verbose = 1
            )
        model.save_weights(log_dir + 'last1.h5')
