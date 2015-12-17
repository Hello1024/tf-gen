import tensorflow as tf
import utils
import time
from datetime import datetime
import matplotlib.pyplot as plt

BATCH_SIZE = 40

with tf.name_scope("input"):
  jpg = tf.image.decode_jpeg(open('brick.jpg', 'r').read())
  jpg.set_shape([100,100,3])  # Bodge to bypass checks later...
  jpg = tf.image.per_image_whitening(jpg)
  crops = []
  for i in range(BATCH_SIZE):
    crops.append(tf.image.random_crop(jpg, [128, 128]))
  jpg = tf.pack(crops)
  #jpg = tf.image.resize_bilinear(jpg, [128, 128])

with tf.name_scope("gen") as scope:
  #inp = tf.placeholder("float", [128])
  #inp2 = tf.Variable(tf.truncated_normal([128], dtype=tf.float32, stddev=1))
  inp = tf.truncated_normal([BATCH_SIZE, 128], dtype=tf.float32, stddev=1)
  
  dou = tf.reshape(inp, [BATCH_SIZE, 4, 4, 8])

  l10 = utils._deconv(dou, 4, 4, 15)
  res = utils._deconv(l10, 1, 1, 3, relu=False)
  res = utils._doublelp(res)
  dou = utils._double(l10)

  rand_l = tf.truncated_normal(res.get_shape(), dtype=tf.float32, stddev=1)
  dou = tf.concat(3, [rand_l, dou, res])
  l10 = utils._deconv(dou, 5, 5, 15)
  res = res + utils._deconv(l10, 1, 1, 3, relu=False)
  res = utils._doublelp(res)
  dou = utils._double(l10)

  rand_l = tf.truncated_normal(res.get_shape(), dtype=tf.float32, stddev=1)
  dou = tf.concat(3, [rand_l, dou, res])
  l10 = utils._deconv(dou, 5, 5, 15)
  res = res + utils._deconv(l10, 1, 1, 3, relu=False)
  res = utils._doublelp(res)
  dou = utils._double(l10)

  rand_l = tf.truncated_normal(res.get_shape(), dtype=tf.float32, stddev=1)
  dou = tf.concat(3, [rand_l, dou, res])
  l10 = utils._deconv(dou, 5, 5, 15)
  res = res + utils._deconv(l10, 1, 1, 3, relu=False)
  res = utils._doublelp(res)
  dou = utils._double(l10)

  rand_l = tf.truncated_normal(res.get_shape(), dtype=tf.float32, stddev=1)
  dou = tf.concat(3, [rand_l, dou, res])
  l10 = utils._deconv(dou, 5, 5, 15)
  res = res + utils._deconv(l10, 1, 1, 3, relu=False)
  res = utils._doublelp(res)
  dou = utils._double(l10)

  rand_l = tf.truncated_normal(res.get_shape(), dtype=tf.float32, stddev=1)
  dou = tf.concat(3, [rand_l, dou, res])
  l10 = utils._deconv(dou, 5, 5, 3)
  res = res + utils._deconv(l10, 1, 1, 3, relu=False)


tf.image_summary("gen", res, max_images=1)
tf.image_summary("real", jpg, max_images=1)

adv_inp = tf.concat(0, [res, jpg])
answers = tf.concat(0, [
  tf.tile(tf.constant([[0.0,1.0]]), [BATCH_SIZE,1]),  
  tf.tile(tf.constant([[1.0,0.0]]), [BATCH_SIZE,1])  
])

with tf.name_scope("adv") as scope:
  al = utils._conv(adv_inp, 3, 3, 15)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 15)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 15)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 15)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 15)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 15)
  al = utils._fc(al, 2)
  al = tf.nn.softmax(al)


cross_entropy = -tf.reduce_sum(answers*tf.log(tf.clip_by_value(al, 1e-10, 1.0))) / BATCH_SIZE

tf.scalar_summary("cross_entropy", cross_entropy)

opt = tf.train.AdagradOptimizer(1e-2)

# Compute the gradients for a list of variables.
grads_and_vars = opt.compute_gradients(cross_entropy)


#for g,v in grads_and_vars:
#  print "G: " + str(g)
#  print "V: " + str(v.name)

def normclip(i):
  return tf.clip_by_norm(i, 1.0)

grads_and_vars = [ (-normclip(g)*tf.exp(-cross_entropy*cross_entropy/4),v) if "gen" in v.name else ((tf.constant([1.0]) - tf.exp(-cross_entropy*cross_entropy/4)) * normclip(g),v) for g, v in grads_and_vars]

# grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# need to the 'gradient' part, for example cap them, etc.
capped_grads_and_vars = grads_and_vars # 

# Ask the optimizer to apply the capped gradients.
train_step = opt.apply_gradients(capped_grads_and_vars)

# another thinggy...
# train_step = tf.train.AdagradOptimizer(3e-2).minimize(tf.reduce_mean(tf.square(l10-jpg)))

saver = tf.train.Saver()

plt.ion()

with tf.Session() as sess:
  testin = tf.truncated_normal([BATCH_SIZE, 128], dtype=tf.float32, stddev=1).eval()
  
  log_path = '/tmp/mnist_logs/'+datetime.now().strftime("%Y/%m/%d/%H/%M/%S")
  merged_summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(log_path, sess.graph_def)

  init = tf.initialize_all_variables()
  sess.run(init)
  print "variables initialized"

  for i in range(1000000):
     print i
     # l10, cross_entropy, train_step
     # ddd = [g for g,v in grads_and_vars];
     #ddd = [tf.log(al13)]
     tr = sess.run([train_step])
     
     if not (i%30):
       summary_str, out, ce, gg = sess.run([merged_summary_op, res, cross_entropy, al], feed_dict={inp: testin})
       summary_writer.add_summary(summary_str, i)

       save_path = saver.save(sess, log_path+ "/model.ckpt")

       print "entropy: " + str(ce)
       print gg
       plt.imshow((out[0,:,:,0:3]/3+0.5).clip(min=0, max=1))
       plt.pause(0.1)


     
     #for j in range(len(gv)):
     #  print "-----------------"
     #  print ddd[j].name
     #  print gv[j]
     
     # summary_str = sess.run(merged_summary_op, feed_dict={inp: testin.eval()})
     # summary_writer.add_summary(summary_str, i)

#print prob
plt.imshow(res[0,:,:,:])
plt.show()

