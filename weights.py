
def bilinear_kernel(h,w,channels=1, use_bias = False) :
    y = np.zeros((h,w,channels,channels), dtype = "float32")
    for i in range(0,h) :
        for j in range(0,w) :
            y[i,j,:,:] = np.identity(channels) / float(h*w*1)
    if use_bias : return [y,np.array([0.], dtype = "float32")]
    else : return [y]
