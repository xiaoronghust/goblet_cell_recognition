import numpy as np 
import matplotlib.pyplot as plt 
from skimage import io 
from skimage.filters import gaussian
from skimage.transform import resize, rescale
from skimage.color import rgb2lab, rgb2hsv, lab2rgb
from skimage.exposure import adjust_gamma
from copy import deepcopy

IMAGE2D_AXIS = [0,1]

def load_sample(image_name):
	image = io.imread(image_name + '.tif').astype('float')
	seg_label = io.imread(image_name + '-seg.tif') > 0
	seg_label = np.expand_dims(seg_label.astype('float'), axis = -1)
	return image, seg_label

def random_crop(image, seg_label, target_size):
	def crop(image, centerxy, target_size):
		xmin = centerxy[0] - target_size//2
		xmax = centerxy[0] + target_size//2
		ymin = centerxy[1] - target_size//2
		ymax = centerxy[1] + target_size//2
		crop_region =image[xmin:xmax, ymin:ymax,:]
		return crop_region
	if np.min(image.shape) < target_size:
		axes = [0,1]
		pad_shape = []
		for k in axes:
			pad_range = max(target_size - image.shape[k], 0)  +32
			if pad_range == 0:
				pad_shape.append((0,0))
			else:
				random_pad_size = np.random.choice(np.arange(0, pad_range))
				pad_shape.append((random_pad_size, pad_range - random_pad_size))
		pad_shape.append((0,0))
		# print(pad_shape)
		image = np.pad(image, pad_shape, 'constant')
		seg_label = np.pad(seg_label, pad_shape, 'constant')
	# print(image.shape, seg_label.shape)
	centerxy = []
	for ax in IMAGE2D_AXIS:
		centerxy.append(np.random.choice(np.arange(target_size//2, image.shape[ax] - target_size//2, 1)))
	image = crop(image, centerxy, target_size)
	seg_label = crop(seg_label, centerxy, target_size)
	return image, seg_label

def random_blur(img):
	if np.random.rand() > 0.5:
		random_sigma = np.random.choice([5])# default  5
		img = gaussian(img, sigma= random_sigma, multichannel= True)
	return img

def random_gamma(img):
	img = adjust_gamma(img, (np.random.rand()*0.5 + 0.5))
	return img

def random_scale(image, seg_label, target_size):
	scale = np.random.rand()*0.3 + 0.7
	image, seg_label = random_crop(image, seg_label, int(np.ceil(target_size/scale) + 10)) # 10 fangzhi crop shixiao 
	image = rescale(image, scale)
	seg_label = rescale(seg_label, scale)
	return image, seg_label

def random_flip(image, seg_label):
	for ax in IMAGE2D_AXIS:
		if np.random.random() > 0.5:
			image = np.flip(image, axis= ax)
			seg_label = np.flip(seg_label, axis = ax)
	return image, seg_label

def random_rotate(image, seg_label):
	rot_num = np.random.choice(np.arange(0,4), size = 1)
	image = np.rot90(image, k = rot_num, axes = IMAGE2D_AXIS)
	seg_label = np.rot90(seg_label, k = rot_num, axes = IMAGE2D_AXIS)
	return image, seg_label

def random_channel_shift(image):
	for nc in range(int(image.shape[-1])):
		image[...,nc] += np.random.random()*20 # default 20
	image = np.clip(image, 0, 255)
	return image

def augmentation_samples(image, seg_label, target_size, use_augment):
	image, seg_label = random_crop(image, seg_label, target_size)
	if use_augment:
		image, seg_label = random_flip(image, seg_label)
		image, seg_label = random_rotate(image, seg_label)
		image = random_blur(image)
		image = random_channel_shift(image)
		seg_label = (seg_label > 0.5).astype('float')
	return image, seg_label

def segmentation_generator(img_list, use_augment, batch_size, target_size):
	cur_list = deepcopy(img_list) 
	while True:
		batch_img = []
		batch_seg_label = []
		for k in range(batch_size):
			if len(cur_list) != 0:
				np.random.shuffle(cur_list)
			else:
				cur_list = deepcopy(img_list)
			sample_img, sample_seg_label = load_sample(cur_list.pop())
			sample_img, sample_seg_label = augmentation_samples(sample_img, sample_seg_label, target_size, use_augment)
			sample_img /=255.
			batch_img.append(sample_img)
			batch_seg_label.append(sample_seg_label)
		batch_img = np.array(batch_img)
		batch_seg_label = np.array(batch_seg_label)
		yield [batch_img, batch_seg_label]