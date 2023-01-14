import numpy as np 
import tiffile as tiff 
from tqdm import *
from skimage.color import lab2rgb, rgb2lab
from glob import *

samplelist = glob("data\\Pathology\\Airway_CLS_Prob_512\\*\\*.tif")

px = []
for samplepth in tqdm(samplelist, ncols= 50):
    img = tiff.imread(samplepth)
    img = rgb2lab(img)
    print(img.max(), img.min())
    px = img.reshape([-1,3]).tolist()

mean_value = np.mean(px, axis = 0)
std_value = np.std(px, axis = 0)

print('mean value:', mean_value)
print('std value:', std_value)

np.save('mean_std.npy', [mean_value, std_value])