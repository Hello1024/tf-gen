import tensorflow as tf
import utils

def superres_layer(inp, name=None):
  results = []
  inp_depth = int(inp.get_shape()[-1])
  if inp_depth != 20:
    inp = tf.pad(inp, [[0,0], [0,0], [0,0], [0,20-inp_depth]])

  inp_x = int(inp.get_shape()[-2])

  if inp_x>=16:
    inp_small = tf.image.resize_images(inp, inp_x/2, inp_x/2)
    inp_small = superres_layer(inp_small, name='s'+name)
    inp = tf.concat(3, [inp, tf.image.resize_images(inp_small, inp_x, inp_x)])

  #results.append(inp[:,:,:,0:3])
  results.append(utils._deconv(inp, 3, 3, 3, relu=False, name="norelu"+name) + inp[:,:,:,0:3])
  results.append(utils._deconv(inp, 5, 5, 10, name=name))
  results.append(tf.nn.max_pool(inp[:,:,:,3:5], [1,8,8,1], [1,1,1,1], 'SAME'))
  results.append(tf.nn.max_pool(inp[:,:,:,5:7], [1,4,4,1], [1,1,1,1], 'SAME'))
  results.append(tf.nn.max_pool(inp[:,:,:,7:9], [1,2,2,1], [1,1,1,1], 'SAME'))
  results.append(tf.truncated_normal(inp[:,:,:,0:1].get_shape(), dtype=tf.float32, stddev=1))
  
  return tf.concat(3, results)


def superres_model(inp):
    inp = superres_layer(inp, name="1")
    #inp = superres_layer(inp, name="2")
    #inp = superres_layer(inp, name="3")
    return inp

