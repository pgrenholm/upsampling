
#Use these weights on a Conv2D or a Conv2DTranspose layer in Keras,
#and it will perform bilinear interpolation by integer magnification (h,w)
def bilinear_kernel(h,w,channels, use_bias = False) :
    y = np.zeros((h,w,channels,channels), dtype = "float32")
    for i in range(0,h) :
        for j in range(0,w) :
            y[i,j,:,:] = np.identity(channels) / float(h*w*1)
    if use_bias : return [y,np.array([0.], dtype = "float32")]
    else : return [y]
    
#Example:
