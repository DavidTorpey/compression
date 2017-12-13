import cv2
import numpy as np
import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, Input, Deconv2D, UpSampling2D
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

def imread(f):
    return cv2.imread(f, 0)

def resize(im, s):
    return cv2.resize(im, s)

def jp2(im):
    return cv2.imencode(".jp2", im)[1]    

im = imread('lena.jpg')
im_ = resize(im, (400, 224))
im = im_.reshape((1, im_.shape[0], im_.shape[1], 1)) / 255.0

input_img = Input(shape=im.shape[1:])
encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
encoded = MaxPooling2D()(encoded)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D()(encoded)

encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)

x1 = UpSampling2D()
x2 = Conv2D(8, (3, 3), activation='relu', padding='same')
x3 = UpSampling2D()
x4 = Conv2D(16, (3, 3), activation='relu', padding='same')
x5 = Conv2D(1, (3, 3), activation='relu', padding='same')

decoded = x1(encoded)
decoded = x2(decoded)
decoded = x3(decoded)
decoded = x4(decoded)
decoded = x5(decoded)

ae = Model(input_img, decoded)
ae.compile(loss='mean_squared_error', optimizer='adam')

ae.fit(im, im, epochs=1000, batch_size=1)

encoder = Model(input_img, encoded)

inp = Input(shape=K.int_shape(encoded)[1:])
dec = x1(inp)
dec = x2(dec)
dec = x3(dec)
dec = x4(dec)
dec = x5(dec)
decoder = Model(inp, dec)


"""compress"""
im_encoded = encoder.predict(im)
compressed = []
mmses = []
for i in range(4):
    curr_image = im_encoded[0, :, :, i] * 255.0
    compressed.append(jp2(curr_image))


"""decompress"""
a = np.zeros((1, K.int_shape(encoded)[1], K.int_shape(encoded)[2], K.int_shape(encoded)[3]))
for i, e in enumerate(compressed):
    decoded_image = cv2.imdecode(e, 0) / 255.0
    a[0, :, :, i] = decoded_image
    
im_decoded = decoder.predict(a) * 255.0
im_r = im_decoded.reshape((im_decoded.shape[1], im_decoded.shape[2])).astype('uint8')
cv2.imwrite('reconstructed.jpg', im_r)
decoder.save_weights('weights.h5py')