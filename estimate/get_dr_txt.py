#----------------------------------------------------#
#   获取测试集的detection-result和images-optional
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
from yolo import YOLO
from PIL import Image
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from utils.utils import letterbox_image
from nets.yolo4_tiny import yolo_body,yolo_eval
import colorsys
import numpy as np
import os

##################### SETTING #########################
txt_pth = 'test_set/Raw_test.txt'
img_pth = 'test_set/Raw/'
INPUT_CHN = 8
exp = 'chn8'

TARGET_SIZE = 256
stride = TARGET_SIZE# - 2*op_size

# UPSAMPLE_RATIO = 2
def crop_patch(img, left_top, target_size = TARGET_SIZE):
    w = TARGET_SIZE
    xmin = left_top[0]
    ymin = left_top[1]
    xmax = left_top[0] + w 
    ymax = left_top[1] + w
    patch = img[xmin:xmax, ymin:ymax,:]
    return patch

#######################################################

class mAP_YOLO(YOLO):
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        self.score = 0.1
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes, INPUT_CHN)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
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

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_id, image, left_top, f):
        # 调整图片使其符合输入要求
        boxed_image = letterbox_image(image, self.model_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            score = str(out_scores[i])

            top, left, bottom, right = out_boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left_top[1] + left)), str(int(left_top[0] + top)), str(int(left_top[1] + right)),str(int(left_top[0] + bottom))))

        return 

yolo = mAP_YOLO()

image_ids = open(txt_pth).read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results/" + exp + '/'):
    os.makedirs("./input/detection-results/" + exp + '/')
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_id in image_ids:
    image_path = img_pth +image_id+".tif"
    f = open("./input/detection-results/" + exp + '/' + image_id + ".txt","w") 
    image = np.array(Image.open(image_path))
    W,H,C = image.shape
    w_num, h_num = W//stride + 1, H//stride + 1
    # 开启后在之后计算mAP可以可视化
    # image.save("./input/images-optional/"+image_id+".jpg")
    for ix in range(w_num):
        for iy in range(h_num):
            left_top = [int(ix*stride), int(iy*stride)]
            patch = crop_patch(image, left_top)
            patch = Image.fromarray(patch)
            yolo.detect_image(image_id,patch, left_top, f)
    f.close()
    print(image_id," done!")
print("Conversion completed!")
