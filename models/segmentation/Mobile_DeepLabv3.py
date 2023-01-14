from keras.layers import *
from keras.models import *
from keras.regularizers import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model

def conv_layer(input_tensor, filter_num, layer_name, kernel_size = 3, strides = 1, dilation_rate = 1, kernel_regularizer = l2(5e-4), use_bias = False, kernel_initializer = 'glorot_normal', activation = 'linear'):
	conv_tensor = Conv2D(filters = filter_num, kernel_size = kernel_size, strides = strides, padding = 'same', activation = activation,
		kernel_regularizer = kernel_regularizer, use_bias = use_bias, dilation_rate = dilation_rate, kernel_initializer = kernel_initializer,
		name = layer_name)(input_tensor)
	return conv_tensor

def deconv_layer(input_tensor, filter_num, layer_name, kernel_size = 3, strides = 1, dilation_rate = 1, kernel_regularizer = l2(5e-4), activation = 'linear', use_bias = False, kernel_initializer = 'glorot_normal'):
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

def aspp(input_tensor, block_name, i_channel):
	b,h,w,c = input_tensor.get_shape().as_list()
	######## pool1 ########
	conv1 = conv_layer(input_tensor, i_channel, block_name + '_dilation1', kernel_size =1)
	conv1 = bn_relu(conv1)
	######## pool2 ########
	conv6 = conv_layer(input_tensor, i_channel, block_name + '_dilation6', kernel_size =3, dilation_rate= 6)
	conv6 = bn_relu(conv6)
	######## pool3 ########
	conv12 = conv_layer(input_tensor, i_channel, block_name + '_dilation12', kernel_size =3, dilation_rate= 12)
	conv12 = bn_relu(conv12)
	######## pool6 ########
	conv18 = conv_layer(input_tensor, i_channel, block_name + '_dilation18', kernel_size =3, dilation_rate = 18)
	conv18 = bn_relu(conv18)
	######## global ########
	gp_tensor = AveragePooling2D(pool_size = [h, w], strides = [h, w], name = block_name + '_global_avgpooling')(input_tensor)
	gp_tensor = conv_layer(gp_tensor, filter_num = i_channel, layer_name = block_name + '_global_conv')
	gp_tensor = UpSampling2D(size = (h, w))(gp_tensor)
	#######################
	add_tensor = Add(name = block_name + '_add')([conv1, conv6, conv12, conv18, gp_tensor])
	acti_tensor = conv_layer(add_tensor, c, block_name + '_conv1_1', kernel_size = 1)
	acti_tensor = bn_relu(acti_tensor)
	return acti_tensor

def MobileNet_DeepLab(input_shape):
    input_tensor = Input(input_shape)
    backbone = model_from_json(open('models/segmentation/MobileNetV2_Dilation.json', 'r').read())
    backbone.load_weights('models/segmentation/MobileNet_ImageNet.h5')
    featruemap = backbone(input_tensor)
    acti_tensor = aspp(featruemap, 'ASPP', 256)
    pred = conv_layer(acti_tensor, filter_num = 1, layer_name = 'pred', kernel_size = 1, strides = 1, use_bias = True, activation = 'sigmoid')
    pred = UpSampling2D(size = (16, 16), interpolation = 'bilinear')(pred)
    return Model(input_tensor, pred)

if __name__ == '__main__':
    DeepLabMobileNet = MobileNet_DeepLab((512,512,3))
    DeepLabMobileNet.summary()
    # plot_model(DeepLabMobileNet, to_file = 'DeepLab_Mobile.png')
