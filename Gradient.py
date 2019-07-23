import numpy as np
import cv2
import dicom
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
import copy

ds = dicom.read_file('Pat_Erl_02d_ABVS.IMA')
img = ds.pixel_array[:,420,:]
img_3D = copy.deepcopy(ds.pixel_array)
print('img_origin',img.shape)
row, col = img.shape
sobel = cv2.Sobel(img,cv2.CV_64F,0,1,ksize = 21)
sobel_Y = cv2.Sobel(sobel,cv2.CV_64F,0,1,ksize=1)
sobel_X = cv2.Sobel(sobel,cv2.CV_64F,1,0,ksize=1)
output = np.zeros_like(img)
for i in range(row):
    for j in range(col):
        if sobel_Y[i,j] > sobel_X[i,j]:
            output[i,j] = img[i,j]



plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Orig')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(output, cmap='gray'),plt.title('output')
plt.xticks([]),plt.yticks([])
plt.show()
