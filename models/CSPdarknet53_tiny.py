import sys 
sys.path.append('../')

from functools import wraps
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Layer, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose
import tensorflow as tf

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


#---------------------------------------------------#
#   CSPdarknet的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#
def resblock_body(x, num_filters, down = True):
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3))(x)
    route = x
    x = Lambda(route_group,arguments={'groups':2, 'group_id':1})(x) 
    x = DarknetConv2D_BN_Leaky(int(num_filters/2), (3,3))(x)
    route_1 = x
    x = DarknetConv2D_BN_Leaky(int(num_filters/2), (3,3))(x)
    x = Concatenate()([x, route_1])
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    feat = x
    x = Concatenate()([route, x])
    if not down:
        x = MaxPooling2D(pool_size=[3,3], strides = [1,1])(x)
    else:
        x = MaxPooling2D(pool_size=[2,2], strides = [2,2])(x)

    # 最后对通道数进行整合
    return x, feat

#---------------------------------------------------#
#   darknet53 的主体部分
#---------------------------------------------------#

def darknet_body(x, input_chn = 32):
    # 进行长和宽的压缩
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(input_chn, (3,3), strides=(1,1))(x)
    # 进行长和宽的压缩
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(input_chn*2, (3,3), strides=(1,1))(x)
    
    x, _ = resblock_body(x,num_filters = input_chn*2, down = False)
    x, _ = resblock_body(x,num_filters = input_chn*4)
    x, feat1 = resblock_body(x,num_filters = input_chn*8)

    x = DarknetConv2D_BN_Leaky(input_chn*16, (3,3))(x)

    feat2 = x
    return feat1, feat2

