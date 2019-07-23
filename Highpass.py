# -*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from scipy import ndimage
import dicom

def contrast(img, k,b):
    row,col = img.shape
    for i in range(row):
        for j in range(col):
            img[i,j] = k * img[i,j] + b
    return img



def HighPass(img,D1):
    dft = np.fft.fft2(img)  # do the fourier transform
    dft_shift = np.fft.fftshift(dft)
    M, N = dft_shift.shape
    x0 = np.floor(M / 2)
    y0 = np.floor(N / 2)
    h = np.zeros_like(dft_shift)
    print(x0)
    print(y0)
    for i in range(M):
        for j in range(N):
            D = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
            #D = np.sqrt((i - y0) ** 2)
            if (D > D1):
                h[i, j] = 1

    dst = dft_shift * h
    f_ishift = np.fft.ifftshift(dst)  # inverse shift
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    #img_back = contrast(img_back, 4, 10)
    #img_back = rescale_intensity(img_back, in_range=(0, 255))
    #img_back = (img_back * 255).astype("uint8")
    #cv2.imwrite('afterHighPass.jpg', img_back)
    return img_back

ds = dicom.read_file('Pat_Erl_02d_ABVS.IMA')
img_org = ds.pixel_array[:,420,:]
print('img_origin',img_org.shape)

D1 = 15
output = HighPass(img_org, D1)
plt.imsave('imageHighpass', output)
'''
cv2.imshow('image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

plt.subplot(121),plt.imshow(img_org, cmap='gray'),plt.title('AfterFilter')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(output, cmap='gray'),plt.title('output')
plt.xticks([]),plt.yticks([])
plt.show()
