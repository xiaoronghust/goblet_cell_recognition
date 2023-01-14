import os
import numpy as np
import copy
import colorsys
import PIL
import matplotlib.pyplot as plt 
from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from models.yolo4_tiny import yolo_body,yolo_eval
from utils.utils import letterbox_image

use_bbox = False
INPUT_CHN = 16
class YOLO(object):
    #--------------------------------------------#
    #   使用自己训练好的模型预测需要修�?个参�?
    #   model_path和classes_path都需要修改！
    #--------------------------------------------#
    _defaults = {
        "model_path": 'logs/goblet_epithelium/chn8_small/last1.h5',
        "anchors_path": 'model_data/anchors.txt',#yolo_anchors.txt
        "classes_path": 'model_data/voc_classes.txt',
        "score" : 0.1,
        "iou" : 0.1,
        # 显存比较小可以使�?16x416
        # 显存比较大可以使�?08x608
        "model_image_size" : (512, 512)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的先验�?
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入�?
        # 否则先构建模型再载入
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes, INPUT_CHN)
            self.yolo_model.summary()
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜�?
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2, ))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                num_classes, self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def color_normalization(self, img, mean_value = np.array([185.84944838, 165.13078433, 225.38486652]), std_value = np.array([59.39843597, 61.1276493 , 28.43154981])):
        img = (img - img.mean())/img.std()*std_value + mean_value
        img = np.clip(img, 0, 255)
        return img

    #---------------------------------------------------#
    #   检测图�?
    #---------------------------------------------------#
    def detect_image(self, image):
        # start = timer()
        image = np.array(image)
        if image.ndim == 2:
            image = np.expand_dims(image, axis = -1)
            image = np.repeat(image, 3, axis = -1).astype('uint8')
        image = PIL.Image.fromarray(image)

        # 调整图片使其符合输入要求
        new_image_size = self.model_image_size
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        # image_data = self.color_normalization(image_data)
        image_data /= 255.
        # print(image_data.max())
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        # fig = plt.figure()
        # plt.imshow(np.array(image))
        # ax = fig.add_subplot(1,1,1)

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 设置字体
        font = ImageFont.truetype(font='font/simhei.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        small_pic=[]
        counts = np.zeros(2)
        for i, c in list(enumerate(out_classes)):
            counts[c] += 1
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            # if (top - bottom)*(left - right) < 50:
            #     continue
            top = top + 3# - 5
            left = left + 3# - 5
            bottom = bottom -3# + 5
            right = right - 3# + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # rect = plt.Rectangle([left, top], (right - left), (top - bottom), fill = False, edgecolor = 'green')
            # ax.add_patch(rect)
            
            # 画框�?
            label = '{} {:.2f}'.format(predicted_class, score)
            # image = np.array(image)
            # image = image[32:-32,32:-32,:]
            # image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                # print(top, left, right, bottom)
                if use_bbox:
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                else:
                    if c == 1:
                        # draw.point([(left + right)/2 - i, (top + bottom)/2 + i], fill = (0,255,0))
                        draw.ellipse([(left + right)/2- 6, (top + bottom)/2 - 6, (left + right)/2 + 6, (top + bottom)/2 + 6], fill=(0, 255, 0), outline=(0, 255, 0))
                    else:
                        # draw.point([(left + right)/2 - i, (top + bottom)/2 + i], fill = (255,0,0))
                        draw.ellipse([(left + right)/2- 6, (top + bottom)/2 - 6, (left + right)/2 + 6, (top + bottom)/2 + 6], fill=(255, 0, 0), outline=(255, 0, 0))
            if use_bbox:
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        # plt.show()
        # end = timer()
        # print(end - start)
        return image, counts

    def close_session(self):
        self.sess.close()

############################## DEBUG ###################################
    # def detect_image(self, image):
    #     start = timer()
    #     image = np.array(image)
    #     if image.ndim == 2:
    #         image = np.expand_dims(image, axis = -1)
    #         image = np.repeat(image, 3, axis = -1).astype('uint8')
    #     image = PIL.Image.fromarray(image)

    #     # 调整图片使其符合输入要求
    #     new_image_size = self.model_image_size
    #     boxed_image = letterbox_image(image, new_image_size)
    #     image_data = np.array(boxed_image, dtype='float32')
    #     image_data /= 255.
    #     image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    #     # 预测结果
    #     out_boxes, out_scores, out_classes = self.sess.run(
    #         [self.boxes, self.scores, self.classes],
    #         feed_dict={
    #             self.yolo_model.input: image_data,
    #             self.input_image_shape: [image.size[1], image.size[0]],
    #             K.learning_phase(): 0
    #         })

    #     print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    #     # 设置字体
    #     font = ImageFont.truetype(font='font/simhei.ttf',
    #                 size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #     thickness = (image.size[0] + image.size[1]) // 300

    #     small_pic=[]
    #     for i, c in list(enumerate(out_classes)):
    #         predicted_class = self.class_names[c]
    #         box = out_boxes[i]
    #         score = out_scores[i]

    #         top, left, bottom, right = box
    #         top = top# - 5
    #         left = left# - 5
    #         bottom = bottom# + 5
    #         right = right + 5
    #         top = max(0, np.floor(top + 0.5).astype('int32'))
    #         left = max(0, np.floor(left + 0.5).astype('int32'))
    #         bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    #         right = min(image.size[0], np.floor(right + 0.5).astype('int32'))


    #         # 画框�?
    #         label = '{} {:.2f}'.format(predicted_class, score)
    #         draw = ImageDraw.Draw(image)
    #         label_size = draw.textsize(label, font)
    #         label = label.encode('utf-8')
    #         print(label)
            
    #         if top - label_size[1] >= 0:
    #             text_origin = np.array([left, top - label_size[1]])
    #         else:
    #             text_origin = np.array([left, top + 1])

    #         for i in range(thickness):
    #             draw.rectangle(
    #                 [left + i, top + i, right - i, bottom - i],
    #                 outline=self.colors[c])
    #         draw.rectangle(
    #             [tuple(text_origin), tuple(text_origin + label_size)],
    #             fill=self.colors[c])
    #         draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
    #         del draw

    #     end = timer()
    #     print(end - start)
    #     return image