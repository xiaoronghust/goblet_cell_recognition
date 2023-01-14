import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.losses import *
from keras.layers import *
import tensorflow as tf
from keras.regularizers import *
# from keras_contrib.layers import *
# from receptivefield.keras import KerasReceptiveField
# from receptivefield.image import get_default_image
# import efficientnet.keras as efn 

REGULAR = l2(1e-5)
USE_BIAS = True
N = 8

def conv_layer(input_tensor, filter_num, layer_name, kernel_size = 3, strides = 1, dilation_rate = 1, kernel_regularizer = l2(5e-4), use_bias = USE_BIAS, kernel_initializer = 'glorot_normal'):
	conv_tensor = Conv2D(filters = filter_num, kernel_size = kernel_size, strides = strides, padding = 'same', activation = 'linear',
		kernel_regularizer = kernel_regularizer, use_bias = USE_BIAS, dilation_rate = dilation_rate, kernel_initializer = kernel_initializer,
		name = layer_name)(input_tensor)
	return conv_tensor

def deconv_layer(input_tensor, filter_num, layer_name, kernel_size = 3, strides = 1, dilation_rate = 1, kernel_regularizer = l2(5e-4), activation = 'linear', use_bias = USE_BIAS, kernel_initializer = 'glorot_normal'):
	conv_tensor = Deconv2D(filters = filter_num, kernel_size = kernel_size, strides = strides, padding = 'same', activation = activation,
		kernel_regularizer = kernel_regularizer, use_bias = use_bias , dilation_rate = dilation_rate,
		kernel_initializer = kernel_initializer,
		name = layer_name)(input_tensor)
	return conv_tensor

def bn_relu(input_tensor):
	bn = input_tensor
	bn = BatchNormalization()(input_tensor)
	ac = Activation('relu')(bn)
	return ac

def interp(output_shape):
	def tf_interp(x):
		return tf.image.resize(x, size = output_shape)
	return Lambda(tf_interp)

def ppm(input_tensor, block_name):
	b,h,w,c = input_tensor.get_shape().as_list()
	######## pool1 ########
	pool1 = AveragePooling2D(pool_size = h//1, strides = h//1)(input_tensor)
	pool1 = conv_layer(pool1, c//4, block_name + '_pool1_conv', kernel_size =1)
	pool1 = bn_relu(pool1)
	pool1 = interp(output_shape = (h, w))(pool1)
	######## pool2 ########
	pool2 = AveragePooling2D(pool_size = h//2, strides = h//2)(input_tensor)
	pool2 = conv_layer(pool2, c//4, block_name + '_pool2_conv', kernel_size =1)
	pool2 = bn_relu(pool2)
	pool2 = interp(output_shape = (h, w))(pool2)
	######## pool3 ########
	pool3 = AveragePooling2D(pool_size = h//3, strides = h//3)(input_tensor)
	pool3 = conv_layer(pool3, c//4, block_name + '_pool3_conv', kernel_size =1)
	pool3 = bn_relu(pool3)
	pool3 = interp(output_shape = (h, w))(pool3)
	######## pool6 ########
	pool6 = AveragePooling2D(pool_size = h//6, strides = h//6)(input_tensor)
	pool6 = conv_layer(pool6, c//4, block_name + '_pool6_conv', kernel_size =1)
	pool6 = bn_relu(pool6)
	pool6 = interp(output_shape = (h, w))(pool6)
	#######################
	concat_tensor = Concatenate(name = block_name + '_concat')([pool1, pool2, pool3, pool6, input_tensor])
	return concat_tensor

def aspp(input_tensor, block_name):
	b,h,w,c = input_tensor.get_shape().as_list()
	######## pool1 ########
	conv1 = conv_layer(input_tensor, c, block_name + '_dilation1', kernel_size =1)
	conv1 = bn_relu(conv1)
	######## pool2 ########
	conv6 = conv_layer(input_tensor, c, block_name + '_dilation6', kernel_size =3, dilation_rate= 6)
	conv6 = bn_relu(conv6)
	######## pool3 ########
	conv12 = conv_layer(input_tensor, c, block_name + '_dilation12', kernel_size =3, dilation_rate= 12)
	conv12 = bn_relu(conv12)
	######## pool6 ########
	conv18 = conv_layer(input_tensor, c, block_name + '_dilation18', kernel_size =3, dilation_rate = 18)
	conv18 = bn_relu(conv18)
	#######################
	add_tensor = Add(name = block_name + '_add')([conv1, conv6, conv12, conv18, input_tensor])
	acti_tensor = conv_layer(add_tensor, c, block_name + '_conv1_1', kernel_size = 1)
	acti_tensor = bn_relu(acti_tensor)
	return acti_tensor

def ag(g, t):
    i_channel = t.get_shape().as_list()[-1]
    t1 = Conv2D(filters = i_channel, kernel_size = [2,2], strides = [2,2], padding = 'same', kernel_regularizer = l2(5e-4), use_bias = False)(t)
    g1 = Conv2D(filters = i_channel, kernel_size = [1,1], strides = [1,1], padding = 'same', kernel_regularizer = l2(5e-4), use_bias = False)(g)
    a1 = Add()([t1, g1])
    a1 = Activation('relu')(a1)
    ph = Conv2D(filters = 1, kernel_size = [1,1], strides = [1,1], padding = 'same', kernel_regularizer = l2(5e-4))(a1)
    ph = Activation('sigmoid')(ph)
    cf = UpSampling2D([2,2])(ph)
    t  = Multiply()([t, cf])
    return t

def unet_backbone(input_size = (128,128,1)):
    inputs = Input(input_size, name = 'input_image')
    conv1 = Conv2D(N, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(N, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(N*2, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(N*2, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(N*4, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv3 = Conv2D(N*4, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(N*8, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    conv4 = Conv2D(N*8, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    # conv4 = ppm(conv4, 'ppm')

    drop4 = Dropout(0.0)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(N*16, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    conv5 = Conv2D(N*16, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu', name = 'top')(conv5)

    conv5 = ppm(conv5, 'ppm')
    drop5 = Dropout(0.0)(conv5)

    drop4 = ag(conv5, drop4)
    up6 = Conv2D(N*8, 2, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(UpSampling2D(size = (2,2))(drop5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)

    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(N*8, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv6 = Conv2D(N*8, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    # conv3 = ag(conv6, conv3)
    up7 = Conv2D(N*4, 2, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)

    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(N*4, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(N*4, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    # conv2 = ag(conv7, conv2)
    up8 = Conv2D(N*2, 2, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)

    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(N*2, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(N*2, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    # conv1 = ag(conv8, conv1)
    up9 = Conv2D(N, 2, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchNormalization()(up9)
    up9 = Activation('relu')(up9)

    featrue_map = concatenate([conv1,up9], axis = 3)
    # conv9 = Conv2D(N, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(merge9)
    # # conv9 = BatchNormalization()(conv9)
    # conv9 = Activation('relu')(conv9)

    # conv9 = Conv2D(N, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(conv9)
    # conv9 = BatchNormalization()(conv9)
    # featrue_map = Activation('relu')(conv9)

    model = Model(input = inputs, output = featrue_map)
    return model

def unet(input_size):
    backbone = unet_backbone(input_size)
    featrue_map = Conv2D(N, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(backbone.output)
    featrue_map = BatchNormalization()(featrue_map)
    featrue_map = Activation('relu')(featrue_map)
    featrue_map = Conv2D(N, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(featrue_map)
    featrue_map = BatchNormalization()(featrue_map)
    featrue_map = Activation('relu')(featrue_map)
    pred = Conv2D(filters = 1, kernel_size = [1,1], strides = [1,1], padding = 'same', activation = 'sigmoid')(featrue_map)
    return Model(backbone.input, pred)

def unet_multitask(input_size):
    backbone = unet_backbone(input_size)
    featrue_map = Conv2D(N, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(backbone.output)
    # featrue_map = BatchNormalization()(featrue_map)
    featrue_map = Activation('relu')(featrue_map)
    featrue_map = Conv2D(N, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(featrue_map)
    # featrue_map = BatchNormalization()(featrue_map)
    featrue_map = Activation('relu')(featrue_map)
    pred_task1 = Conv2D(filters = 1, kernel_size = [1,1], strides = [1,1], padding = 'same', activation = 'sigmoid', name = 'task1')(featrue_map)
    
    featrue_map = Conv2D(N, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(backbone.output)
    # featrue_map = BatchNormalization()(featrue_map)
    featrue_map = Activation('relu')(featrue_map)
    featrue_map = Conv2D(N, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = l2(5e-4))(featrue_map)
    # featrue_map = BatchNormalization()(featrue_map)
    featrue_map = Activation('relu')(featrue_map)
    pred_task2 = Conv2D(filters = 1, kernel_size = [1,1], strides = [1,1], padding = 'same', activation = 'sigmoid', name = 'task2')(featrue_map)
    return Model(backbone.input, [pred_task1, pred_task2])

def dice_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1- K.epsilon())
    i = K.sum(y_true*y_pred) + K.epsilon()
    u = K.sum(y_true) + K.sum(y_pred) + K.epsilon()
    return 1 - 2*i/u

def wbce(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1- K.epsilon())
    ratio = K.mean(y_true) + K.epsilon()
    loss = -y_true*K.log(y_pred)*(1 - ratio) - (1 - y_true)*K.log(1 - y_pred)*ratio
    return K.sum(loss)/K.sum(y_true*(1 - ratio) + (1 - y_true)*ratio)

def h_dc(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, 0.5), 'float')
    return 1 - dice_loss(y_true, y_pred)

def bce_dice(y_true, y_pred):
    return dice_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

############################### build unet from vgg16 ##################################
from keras.applications import *

output_layers = {
    'VGG16':['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'],
    'ResNet50':['activation_1', 'activation_10', 'activation_22', 'activation_40', 'activation_49'],
    'DenseNet121':['conv1/relu', 'pool2_relu', 'pool3_relu', 'pool4_relu', 'relu'],
    'DenseNet169':['conv1/relu', 'pool2_relu', 'pool3_relu', 'pool4_relu', 'relu'],
    'MobileNetV2':['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu'],
    'EfficientNetB0': ['block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation']
}


def unet_backbonenet(input_shape, backbone, use_multitask = False, stage_num = None, top = None):
    def build_backbone(input_shape, backbone):
        if backbone == 'VGG16':
            return VGG16(input_shape = input_shape, include_top = False, weights = 'imagenet')
        if backbone == 'ResNet50':
            return ResNet50(input_shape = input_shape, include_top = False, weights = 'imagenet')
        if backbone == 'DenseNet121':
            return DenseNet121(input_shape = input_shape, include_top = False, weights = 'imagenet')
        if backbone == 'DenseNet169':
            return DenseNet169(input_shape = input_shape, include_top = False, weights = 'imagenet')
        if backbone == 'MobileNetV2':
            return MobileNetV2(input_shape = input_shape, include_top = False, weights = 'imagenet')
        if backbone == 'EfficientNetB0':
            return efn.EfficientNetB0(input_shape = input_shape, include_top = False, weights = 'noisy-student')#weights = 'imagenet'
        else:
            return None

    def uplayers(input_tensor, skip_tensor):
        i_channel = skip_tensor.get_shape().as_list()[-1]
        up_tensor = Deconv2D(filters = i_channel, kernel_size = [2,2], strides = [2,2], padding = 'same', kernel_regularizer = l2(0))(input_tensor)
        up_tensor = bn_relu(up_tensor)
        concat_tensor = Concatenate()([up_tensor, skip_tensor])
        conv_tensor = Conv2D(filters = i_channel, kernel_size = [3,3], strides = [1,1], padding = 'same', kernel_regularizer = l2(0))(concat_tensor)
        conv_tensor = bn_relu(conv_tensor)
        conv_tensor = Conv2D(filters = i_channel, kernel_size = [3,3], strides = [1,1], padding = 'same', kernel_regularizer = l2(0))(conv_tensor)
        conv_tensor = bn_relu(conv_tensor)
        # conv_tensor = SpatialDropout2D(rate = 0.5)(conv_tensor)
        return conv_tensor

    def add_uplayers(input_tensor, skip_tensor):
        skip_tensor = ag(input_tensor, skip_tensor)
        i_channel = skip_tensor.get_shape().as_list()[-1]
        # skip_tensor = Conv2D(filters = i_channel, kernel_size = [1,1], strides = [1,1], padding = 'same')(skip_tensor)
        up_tensor = Deconv2D(filters = i_channel, kernel_size = [2,2], strides = [2,2], padding = 'same')(input_tensor)
        up_tensor = bn_relu(up_tensor)
        concat_tensor = Concatenate()([up_tensor, skip_tensor])
        conv_tensor = Conv2D(filters = i_channel, kernel_size = [3,3], strides = [1,1], padding = 'same')(concat_tensor)
        conv_tensor = bn_relu(conv_tensor)
        conv_tensor = Conv2D(filters = i_channel, kernel_size = [3,3], strides = [1,1], padding = 'same')(conv_tensor)
        conv_tensor = bn_relu(conv_tensor)
        return conv_tensor

    backbone_net = build_backbone(input_shape, backbone)
    skip_tensors = []
    if stage_num == 5:
        stage_num = len(output_layers[backbone]) - 1

    for l, l_name in enumerate(output_layers[backbone]):
        skip_tensors.append(backbone_net.get_layer(l_name).output)
        if l > stage_num - 1:
            break
    
    skip_tensors = skip_tensors[::-1]

    conv_tensor = skip_tensors[0]
    if top == 'PPM':
        conv_tensor = ppm(skip_tensors[0], 'ppm')
    if top == 'ASPP':
        conv_tensor = aspp(skip_tensors[0], 'aspp')

    for stage in range(stage_num):
        conv_tensor = uplayers(conv_tensor, skip_tensors[stage + 1])
    if use_multitask:
        # pred1 = Deconv2D(filters = 1, kernel_size = [2,2], strides = [2,2], padding = 'same', activation = 'sigmoid', kernel_regularizer = l2(0), name = 'pred/t1')(conv_tensor)
        # pred2 = Deconv2D(filters = 1, kernel_size = [2,2], strides = [2,2], padding = 'same', activation = 'sigmoid', kernel_regularizer = l2(0), name = 'pred/t2')(conv_tensor)
        # return Model(backbone_net.input, [pred1, pred2])
        pred1 = Conv2D(filters = 1, kernel_size = [1,1], strides = [1,1], padding = 'same', activation = 'sigmoid', name = 'pred/task1_conv')(conv_tensor)
        pred1 = UpSampling2D(size = (2,2), interpolation = 'bilinear', name = 'pred/task1')(pred1)
        pred2 = Conv2D(filters = 1, kernel_size = [1,1], strides = [1,1], padding = 'same', activation = 'sigmoid', name = 'pred/task2_conv')(conv_tensor)
        pred2 = UpSampling2D(size = (2,2), interpolation = 'bilinear', name = 'pred/task2')(pred2)
        return Model(backbone_net.input, [pred1, pred2])
    else:
        pred = Deconv2D(filters = 1, kernel_size = [2,2], strides = [2,2], padding = 'same', activation = 'sigmoid')(conv_tensor)
        return Model(backbone_net.input, pred)

def TernausNet(input_shape, backbone, use_multitask = False, stage_num = None):
    def build_backbone(input_shape, backbone):
        if backbone == 'VGG16':
            return VGG16(input_shape = input_shape, include_top = False, weights = 'imagenet')
        if backbone == 'ResNet50':
            return ResNet50(input_shape = input_shape, include_top = False, weights = 'imagenet')
        if backbone == 'DenseNet121':
            return DenseNet121(input_shape = input_shape, include_top = False, weights = 'imagenet')
        if backbone == 'DenseNet169':
            return DenseNet169(input_shape = input_shape, include_top = False, weights = 'imagenet')
        if backbone == 'MobileNetV2':
            return MobileNetV2(input_shape = input_shape, include_top = False, weights = 'imagenet')
        if backbone == 'EfficientNetB0':
            return efn.EfficientNetB0(input_shape = input_shape, include_top = False, weights = 'noisy-student')#weights = 'imagenet'
        else:
            return None

    def uplayers(input_tensor, skip_tensor):
        i_channel = skip_tensor.get_shape().as_list()[-1]
        conv_tensor = Conv2D(filters = i_channel, kernel_size = [3,3], strides = [1,1], padding = 'same')(input_tensor)
        conv_tensor = bn_relu(conv_tensor)
        up_tensor = Deconv2D(filters = i_channel//2, kernel_size = [2,2], strides = [2,2], padding = 'same')(conv_tensor)
        up_tensor = bn_relu(up_tensor)
        concat_tensor = Concatenate()([up_tensor, skip_tensor])
        return concat_tensor, i_channel//2

    backbone_net = build_backbone(input_shape, backbone)
    skip_tensors = []
    if stage_num == None:
        stage_num = len(output_layers[backbone]) - 1

    # for layer in backbone_net.layers:
    #     layer.trainable = True

    for l, l_name in enumerate(output_layers[backbone]):
        skip_tensors.append(backbone_net.get_layer(l_name).output)
        if l > stage_num - 1:
            break
    
    skip_tensors = skip_tensors[::-1]

    conv_tensor = skip_tensors[0]
    # conv_tensor = ppm(skip_tensors[0], 'ppm')

    for stage in range(stage_num):
        conv_tensor, o_channel = uplayers(conv_tensor, skip_tensors[stage + 1])
    conv_tensor = Conv2D(filters = o_channel, kernel_size = [3,3], strides = [1,1], padding = 'same')(conv_tensor)
    conv_tensor = bn_relu(conv_tensor)
    if use_multitask:
        pred1 = Conv2D(filters = 1, kernel_size = [1,1], strides = [1,1], padding = 'same', activation = 'sigmoid', name = 'pred/task1_conv')(conv_tensor)
        pred1 = UpSampling2D(size = (2,2), interpolation = 'bilinear', name = 'pred/task1')(pred1)
        pred2 = Conv2D(filters = 1, kernel_size = [1,1], strides = [1,1], padding = 'same', activation = 'sigmoid', name = 'pred/task2_conv')(conv_tensor)
        pred2 = UpSampling2D(size = (2,2), interpolation = 'bilinear', name = 'pred/task2')(pred2)
        return Model(backbone_net.input, [pred1, pred2])
    else:
        pred = Deconv2D(filters = 1, kernel_size = [2,2], strides = [2,2], padding = 'same', activation = 'sigmoid')(conv_tensor)
        return Model(backbone_net.input, pred)

########################################################################################

# if __name__ == '__main__':
#     shape = [512,128,3]
#     rf = KerasReceptiveField(unet, init_weights=True)
#     rf_params = rf.compute(shape, 'input_image', ['top'])
#     rf.plot_rf_grids(get_default_image(shape, name='doge'))