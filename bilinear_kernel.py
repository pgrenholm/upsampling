#This is done in Keras 2.0, with tensorflow channel ordering (channels_last)
import numpy as np

#Use these weights on a Conv2D or a Conv2DTranspose layer in Keras,
#and you can perform bilinear interpolation by integer magnification (h,w),
#Arguments: 
#  '(h,w)' is the magnification factors for height and weight. These must be integers.
#   If the input image has size (in_height, in_width), the output image after scaling will have size
#   (h*(in_height-1) + 1,  w*(in_width-1) + 1)
#   'channels' is the number of channels, this must be the same in input and output image.
#   'use_bias' must have the same value as 'use_bias' in the Keras layer initialization. Default is True,
#   which means that the layer has a bias which is set to 0.0.
#   'dtype' is the datatype of the returned weights
def bilinear_kernel(h,w,channels, use_bias = True, dtype = "float32") :
    y = np.zeros((h,w,channels,channels), dtype = dtype)
    for i in range(0,h) :
        for j in range(0,w) :
            y[i,j,:,:] = np.identity(channels) / float(h*w*1)
    if use_bias : return [y,np.array([0.], dtype = dtype)]
    else : return [y]
    
#Examples:

from keras.models import Model
in_height, in_width = 32,32

#A convolutional layer is easiest:
from keras.layers import Input, Upsampling2D, Conv2D

(h,w,channels) = (2,2,3)
input_layer = Input([in_height, in_width, channels])
us = UpSampling2D(size = (h,w))(input_layer)
conv = Conv2D(filters = channels, kernel_size = (h,w), strides=(1,1), 
            activation = 'linear',
            padding = 'valid', name = 'conv', use_bias = False,
            weights = bilinear_kernel(h,w,channels, False))(us)
model = Model(input_layer, conv)

#A transposed convolutional layer will need extra cropping 
from keras.layers import Input, UpSampling2D, Conv2DTranspose, Cropping2D

(h,w,channels) = (3,5,1)
input_layer = Input([in_height, in_width, channels])
us = UpSampling2D(size = (h,w))(input_layer)
ct = Conv2DTranspose(filters = channels, kernel_size = (h,w),strides=(1,1), 
            activation = 'linear',
            padding = 'valid', name = 'ct', 
            weights = bilinear_kernel(h,w,channels))(us)
cr = Cropping2D(cropping = (h-1,w-1))(ct)
model = Model(input_layer, cr)

#We can then use either model model.predict to enlarge an image:
x = np.random.rand((1,in_height,in_width, channels), dtype = "float32")
y = model.predict(x)
