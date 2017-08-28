from __future__ import print_function

import time
from PIL import Image
import numpy as np

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

import loss_func

height = 512
width = 512

img_path = "images/image1"
content_img = Image.open(img_path)
content_img = content_img.resize((height, width))
#content_img.show()

img_path2 = "images/image2"
style_img = Image.open(img_path2)
style_img = style_img.resize((height, width))
#style_img.show()

content_arr = np.asarray(content_img, dtype='float32')
content_arr = np.expand_dims(content_arr, axis=0)
print(content_arr.shape)

style_arr = np.asarray(style_img, dtype='float32')
style_arr = np.expand_dims(style_arr, axis=0)
print(style_arr.shape)

content_arr[:, :, :, 0] -= 103.939
content_arr[:, :, :, 1] -= 116.779
content_arr[:, :, :, 2] -= 123.68
content_arr = content_arr[:, :, :, ::-1]

style_arr[:, :, :, 0] -= 103.939
style_arr[:, :, :, 1] -= 116.779
style_arr[:, :, :, 2] -= 123.68
style_arr = style_arr[:, :, :, ::-1]

content_image = backend.variable(content_arr)
style_image = backend.variable(style_arr)
resultant_image = backend.placeholder((1, height, width, 3))

input_arr = backend.concatenate([content_image, style_image, resultant_image], axis=0)

model = VGG16(input_tensor=input_arr, weights='imagenet', include_top=False)
layers = dict([(layer.name, layer.output) for layer in model.layers])
print(layers)

c_weight = 0.025
s_weight = 1.0
res_weight = 1.0
loss = backend.variable(0.)

content_features = layers['block2_conv2']
img_features = content_features[0,:,:,:]
resultant_features = content_features[2,:,:,:]

loss += c_weight*loss_func.content_loss(img_features, resultant_features)

style_features_layers = ['block1_conv2', 'block2_conv2','block3_conv3', 'block4_conv3', 'block5_conv3']

for name in style_features_layers:
    features = layers[name]
    style_features = features[1, :, :, :]
    resultant_features = features[2, :,:,:]
    s_loss = loss_func.style_loss(style_features, resultant_features, height, width, 3)
    loss += s_weight*s_loss
    
grads = backend.gradients(loss, resultant_image)
outputs = [loss]
outputs += grads
output_func = backend.function([resultant_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = output_func([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

iterations = 100

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
