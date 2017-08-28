from __future__ import print_function

import time
from PIL import Image
import numpy as np

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

def content_loss(content, resultant):
    c_loss =  backend.sum(backend.square(content-resultant))
    return c_loss/2

def style_loss(style, resultant, height, width, channels):
    s_loss = backend.sum(backend.square(gram_mat(style)-gram_mat(resultant)))
    return s_loss/(4*(channels**2)*(height**2)*(width**2))
    
def gram_mat(img):
    features = backend.batch_flatten(backend.permute_dimensions(img, (2,0,1)))
    mat = backend.sum(backend.dot(features, backend.transpose(features)))
    return mat