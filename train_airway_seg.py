import os 
import numpy as np
import tensorflow as tf 
from glob import *
from utils.seg_generator import segmentation_generator
from models.segmentation.Mobile_DeepLabv3 import *
from models.segmentation.unet import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import save_model

############## GLOBAL PARAMETERS ##############
# [experiment]
log_dir = 'logs/test1/' # 对应log文件夹
# [dataset]
data_dir = 'dataset/Airway/' #数据集路径
# [training] 
Epochs = 25
batch_size = 4 #批量大小
target_size = 512 # 输入图像尺寸
lr = 1e-3 # 学习率
############## GLOBAL PARAMETERS ##############
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
sample_num = len(glob(data_dir + '/*.tif'))//2
img_list = [data_dir + str(k).zfill(3) for k in np.arange(1,sample_num + 1)]
np.random.seed(1337)
np.random.shuffle(img_list)

train_list = img_list[:int(0.5*len(img_list))]
test_list = img_list[int(0.5*len(img_list)):]

# 存储训练和测试样本列表
np.save('seg_train_list.npy', train_list) 
np.save('seg_test_list.npy', test_list)

train_sample_num = int(np.ceil(0.8*len(train_list)))

traingen = segmentation_generator(train_list[0:train_sample_num], True, batch_size, target_size)
valgen = segmentation_generator(train_list[train_sample_num:], False, batch_size, target_size)

segmentor = MobileNet_DeepLab(input_shape = (target_size, target_size, 3))
segmentor.compile(Adam(lr = lr), loss = [bce_dice], metrics = [h_dc])
segmentor.summary()

callbacks = [
    EarlyStopping(monitor= 'val_loss', min_delta= 1e-3, patience= 20, restore_best_weights= True),
    ReduceLROnPlateau(monitor = 'val_loss', factor= 0.5, patience= 1, verbose = 1)
]

segmentor.fit_generator(
    traingen,
    steps_per_epoch = 100,
    epochs = Epochs,
    validation_data = valgen,
    callbacks = callbacks,
    shuffle = True,
    validation_steps = 100,
    initial_epoch = 0,
)

save_model(segmentor, log_dir + '/weights.h5')
