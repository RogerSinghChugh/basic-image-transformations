import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from matplotlib.pyplot import imshow
from skimage import exposure


imagePath = 'resources/car.png'

# Reading the image
masterCarImage = cv2.imread(imagePath, 0)
img = masterCarImage.copy()


# print("Noisy Image")
# cv2.imshow('', masterCarImage)
# cv2.waitKey(0)

# apply fft2 that refers to 2D fft. fft2() provides us the frequency transform which will be a complex array. It first argument is a greyscale image.
f=np.fft.fft2(img)
# next, we apply ffshift() that essentially performs multiplication operation f(x,y)(-1)^(x+y) and then takes the FT of this product.
# we want to place the zero frequency component in the center. Otherwise, it will be at the top left corner. We shift the result in both directions.
fshift=np.fft.fftshift(f)
# calculate the magnitude of DFT and log scale for the purpose of visualization
magnitude_spectrum=20*np.log(np.abs(fshift))

# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input image'), plt.xticks([]),plt.yticks([])
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Magnitude'), plt.xticks([]),plt.yticks([])
# plt.show()


#Find peaks in the image
coordinates = peak_local_max(magnitude_spectrum, min_distance=18, threshold_abs=125.0)
# fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharex=True, sharey=True)
# ax = axes.ravel()
# ax[0].imshow(img, cmap=plt.cm.gray)
# ax[0].axis('off')
# ax[0].set_title('Original')
# ax[1].imshow(magnitude_spectrum, cmap=plt.cm.gray)
# ax[1].autoscale(False)
# ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
# ax[1].axis('off')
# ax[1].set_title('Peak local max')
# fig.tight_layout()
# plt.show()
print("The min_distance parameter is used to enforce a minimum distance between the detected peaks. When min_distance is set to a positive integer value d, the function only returns peaks that are separated by at least d pixels. This can be useful when detecting peaks in noisy images or in regions with a high density of peaks. By setting min_distance to a larger value, the function will only return the most significant peaks and ignore smaller peaks that are closer together.")



l=[7]
for L in l:
	dx,dy=np.shape(img)[0], np.shape(img)[1]
	new=magnitude_spectrum.copy()
	for coord in coordinates:
	    i=coord[0]
	    j=coord[1]
	    if i==dx//2 and j==dy//2:
	        continue
	    else:
	        for k1 in np.arange(-L,L,1):
	            for k2 in np.arange(-L,L,1):
	                if i+k1>=0 and j+k2>=0 and i+k1<dx and j+k2<dy:
	                    new[i+k1,j+k2]=0
	                    fshift[i+k1,j+k2]=0 # shifted DFT of car image
	imshow(new, cmap='gray')
	plt.title("The size of the neighbourhood is "+str(2*L+1)+"x"+str(2*L+1))
	plt.waitforbuttonpress()

	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = img_back.real

	# show the original image and the filtered image side by side
	fig, ax = plt.subplots(1, 2, figsize=(15, 15))
	ax[0].imshow(img, cmap='gray')
	ax[0].set_title('Original Image')
	ax[1].imshow(img_back, cmap='gray')
	ax[1].set_title('Filtered Image')
	plt.show()
print("To determine the value of L that removes the periodic noise while preserving the important features of the "
      "original image, we can experiment with different values and observe the resulting filtered image. We can start "
      "with a small value of L, such as 5, and gradually increase it until we find a value that produces a visually "
      "satisfactory result.")
# L = 8
# dx, dy = np.shape(img)[0], np.shape(img)[1]
# new = magnitude_spectrum.copy()
# for coord in coordinates:
# 	i = coord[0]
# 	j = coord[1]
# 	if i == dx // 2 and j == dy // 2:
# 		continue
# 	else:
# 		for k1 in np.arange(-L, L, 1):
# 			for k2 in np.arange(-L, L, 1):
# 				if i + k1 >= 0 and j + k2 >= 0 and i + k1 < dx and j + k2 < dy:
# 					new[i + k1, j + k2] = 0
# 					fshift[i + k1, j + k2] = 0  # shifted DFT of car image
# imshow(new, cmap='gray')
# plt.title("The size of the neighbourhood is " + str(2 * L + 1) + "x" + str(2 * L + 1))
# plt.waitforbuttonpress()


img_back = np.fft.ifft2(np.fft.ifftshift(fshift))
img_back = np.abs(img_back)
print("The modified magnitude (new) represents the Fourier spectrum of the filtered image, where the frequency components"
      " corresponding to the periodic noise have been set to zero. The shifted Fourier transform at each point (u,v), "
      "denoted as fshift, represents the complex value of the Fourier transform at that point after it has been shifted "
      "to the center of the image. It is used to update the Fourier transform during the filtering process.")
# show the original image and the filtered image side by side
fig, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(img_back, cmap='gray')
ax[1].set_title('Filtered Image')
plt.show()

print("Original image intensity range:", np.min(img), "-", np.max(img))
print("Restored image intensity range:", np.min(img_back), "-", np.max(img_back))
img_back=(img_back-np.min(img_back))*255.0/(np.max(img_back)-np.min(img_back))
imshow(img_back, cmap='gray')
plt.waitforbuttonpress()
print("Restored image intensity range after rescaling:", np.min(img_back), "-", np.max(img_back))

print("The filter that is known for its ability to remove localized noise in the Fourier domain is the bandpass filter."
      " In the given code snippet, the filter has been designed by setting the magnitudes of the Fourier transform to "
      "zero in a certain neighborhood around the detected noise peaks. This is equivalent to multiplying the Fourier "
      "transform by a rectangular function, which is the filter. More specifically, the mathematical operation that "
      "has been performed between the magnitude and the filter is a point-wise multiplication, also known as "
      "element-wise or Hadamard product, between the magnitude and the filter. By setting the magnitudes to zero in a "
      "certain neighborhood, the filter effectively suppresses the Fourier components associated with the noise, and "
      "only allows the remaining clean Fourier components to pass through.")


