import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
  # load image
  img = skimage.io.imread(path)
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  return resized_img

conv_counter = 0
def _conv(inpOp, kH, kW, nOut, dH=1, dW=1, relu=True):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        nIn = int(inpOp.get_shape()[-1])
        stddev = 1.4e-2
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=(kH*kW*nIn)**0.5*stddev), name='weights')
        
        conv = tf.nn.conv2d(inpOp, kernel, [1, 1, 1, 1],
                         padding="SAME")

        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if relu:
          bias = tf.nn.relu(bias, name=scope)
        #parameters += [kernel, biases]
        # bias = tf.Print(bias, [tf.sqrt(tf.reduce_mean(tf.square(inpOp - tf.reduce_mean(inpOp))))], message=name)
        tf.histogram_summary(scope+"/output", bias)
        tf.image_summary(scope+"/output", bias[:,:,:,0:3])
        tf.image_summary(scope+"/kernel_weight", tf.expand_dims(kernel[:,:,0:3,0], 0))
        # tf.image_summary(scope+"/point_weight", pointwise_filter)
        
        return bias

deconv_counter = 0
def _deconv(inpOp, kH, kW, nOut, dH=1, dW=1, relu=True):
    global deconv_counter
    global parameters
    name = 'conv' + str(deconv_counter)
    deconv_counter += 1
    with tf.name_scope(name) as scope:
        nIn = int(inpOp.get_shape()[-1])
        in_shape = inpOp.get_shape()
        stddev = 7e-3
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nOut, nIn],
                                                 dtype=tf.float32,
                                                 stddev=(kH*kW*nIn)**0.5*stddev), name='weights')
        
        conv = tf.nn.deconv2d(inpOp, kernel, [int(in_shape[0]),int(in_shape[1]),int(in_shape[2]),nOut], [1, 1, 1, 1],
                         padding="SAME")
                         
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if relu:
          bias = tf.nn.relu(bias, name=scope)
        #parameters += [kernel, biases]
        #bias = tf.Print(bias, [tf.sqrt(tf.reduce_mean(tf.square(inpOp - tf.reduce_mean(inpOp))))], message=name)
        tf.histogram_summary(scope+"/output", bias)
        tf.image_summary(scope+"/output", bias[:,:,:,0:3])
        #tf.image_summary(scope+"/depth_weight", depthwise_filter)
        # tf.image_summary(scope+"/point_weight", pointwise_filter)
        
        return bias

        
fc_counter = 0
def _fc(inpOp, nOut):
    global fc_counter
    name = 'fc' + str(fc_counter)
    fc_counter += 1
    with tf.name_scope(name) as scope:
        shape = inpOp.get_shape()
        reshaped = tf.reshape(inpOp, [int(shape[0]), shape.num_elements()/int(shape[0])])
        nIn = int(reshaped.get_shape()[-1])
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=True)
        b = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        res = tf.matmul(reshaped, kernel) + b
        return res

def _doubledim(inp, dim):
    #a, b = tf.split(3, 2, inp)
    #return tf.concat(dim, [a, b])
    
    b = tf.expand_dims(inp,dim+1)
    c = tf.concat(dim+1, [b, tf.zeros(b.get_shape())])

    #blur_kernel_size = [1,1,1,1]
    #blur_kernel_size[dim-1]=2
    
    shape = inp.get_shape()
    
    new_shape = []
    for i in range(len(shape)):
      if i==dim:
        new_shape.append(int(shape[i])*2)
      else:
        new_shape.append(int(shape[i]))
    
    c = tf.reshape(c, new_shape)

    
    
    return c

def _doublesh(inp):
    with tf.name_scope("doublesh"):
      a = _double(inp)
      ndims = int(a.get_shape()[3])

      kernel = tf.constant([[0.25, -0.5, 0.25], [-0.5, 1.0, -0.5], [0.25, -0.5, 0.25]])
      kernel = tf.expand_dims(kernel,2)
      kernel = tf.expand_dims(kernel,3)
      kernel = tf.tile(kernel,[1,1,ndims,1])

      #a =  tf.nn.depthwise_conv2d(a, kernel, [1, 1, 1, 1],
      #                     padding="SAME", name="blur")
      return a
def _doublelp(inp):
    with tf.name_scope("doublelp"):
      a = _double(inp)
      ndims = int(a.get_shape()[3])

      kernel = tf.constant([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
      kernel = tf.expand_dims(kernel,2)
      kernel = tf.expand_dims(kernel,3)
      kernel = tf.tile(kernel,[1,1,ndims,1])

      a =  tf.nn.depthwise_conv2d(a, kernel, [1, 1, 1, 1],
                           padding="SAME", name="blur")
      return a


def _double(inp):
    with tf.name_scope("double"):
      a = _doubledim(inp, 1)
      a = _doubledim(a, 2)
      
      return a

def _half(inpOp):
    shape = inpOp.get_shape()
    inpOp = tf.nn.dropout(inpOp, 0.5);
    return tf.nn.avg_pool(inpOp, [1,2,2,1], [1,2,2,1], 'SAME')

