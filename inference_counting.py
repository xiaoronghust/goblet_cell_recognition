import matplotlib.pyplot as plt 
from yolo import YOLO
from PIL import Image
from glob import *
import numpy as np 
from time import *
from skimage.io import imread, imsave
from skimage.transform import resize
from models.unet import *
from skimage.morphology import remove_small_objects, closing, disk, remove_small_holes
from skimage.measure import label
from models.segmentation.unet import *

print('loading model')
tic = time()
yolo = YOLO()

TARGET_SIZE = 512
UPSAMPLE_RATIO = 1
INPUT_SHAPE = 512

# segmentor = unet_backbonenet(input_shape = (TARGET_SIZE, TARGET_SIZE, 3), backbone = 'DenseNet121', use_multitask= True)
segmentor = load_model('logs/test1/weights.h5', custom_objects= {'bce_dice':bce_dice,'h_dc':h_dc})
toc = time()
print('using ' + str(toc - tic) + ' sec')

def crop_patch(img, left_top, target_size = TARGET_SIZE):
    w = TARGET_SIZE
    xmin = left_top[0]
    ymin = left_top[1]
    xmax = left_top[0] + w 
    ymax = left_top[1] + w
    patch = img[xmin:xmax, ymin:ymax,:]
    return patch

while True:
    
    img_pth = input('Input image directory:')
    img = imread(img_pth)
    o_img = img.copy()

    op_size = 64
    stride = TARGET_SIZE - 2*op_size

    W,H,C = img.shape
    w_num, h_num = W//stride + 1, H//stride + 1

    pad_shape = ((op_size, w_num*stride - img.shape[0] + op_size),(op_size, h_num*stride - img.shape[1] + op_size),(0,0))
    img = np.pad(img, pad_shape, mode = 'constant')
    res = np.zeros(img.shape[:2] + (3,))

    tic1 = time()
    cell_num = np.zeros(2)
    for ix in range(w_num):
        for iy in range(h_num):
            left_top = [int(ix*stride), int(iy*stride)]
            patch = crop_patch(img, left_top)
            try:
                image = Image.fromarray(patch)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = segmentor.predict(np.expand_dims(patch/255., axis = 0))[0,...,0]
                # print(r_image.shape)
                r_image = remove_small_objects(r_image > 0.5, min_size = 10000).astype('uint8')
                r_image = remove_small_holes(r_image, area_threshold= 2000)
                r_image = closing(r_image, disk(20)).astype('float')
                patch_target = patch.copy()
                patch_target[r_image < 0.5] = 0
                patch_target = patch_target.astype('uint8')
                image_target = Image.fromarray(patch_target)
                r_image, count_num = yolo.detect_image(image_target.resize((INPUT_SHAPE*UPSAMPLE_RATIO, INPUT_SHAPE*UPSAMPLE_RATIO)))
                r_image = np.array(r_image)
                r_image = resize(r_image, [TARGET_SIZE,TARGET_SIZE])*255
                # print(count_num)
                for k in range(2):
                    cell_num[k] += count_num[k]
            res[left_top[0]:left_top[0]+TARGET_SIZE - 2*op_size, left_top[1]:left_top[1]+TARGET_SIZE - 2*op_size] = r_image[op_size:-op_size,op_size:-op_size,:]
    res = res[:W,:H,:]
    print('saving to ' + img_pth.split('\\')[-1])
    imsave(img_pth.split('\\')[0] + '\\result_seg' + img_pth.split('\\')[-1], (np.mean(res, axis = -1) > 0).astype('uint8')*255)
    imsave(img_pth.split('\\')[0] + '\\result_pt' + img_pth.split('\\')[-1], res.astype('uint8'))
    toc2 = time()
    print('#######################', str(round(toc2 - tic1, 2)), 's')
    plt.figure()
    plt.subplot(121)
    plt.imshow(o_img)
    plt.subplot(122)
    plt.imshow(res.astype('uint8'))
    plt.title('goblet cell: ' + str(cell_num[0]) + ' epithelium cell ' + str(cell_num[1]), fontsize = 10, loc = 'left')
    plt.show()

yolo.close_session()
