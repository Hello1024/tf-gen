import tensorflow as tf
import utils
import time
from datetime import datetime
import matplotlib.pyplot as plt

with tf.name_scope("input"):
  jpg = tf.image.decode_jpeg(open('brick.jpg', 'r').read())
  jpg.set_shape([100,100,3])  # Bodge to bypass checks later...
  jpg = tf.image.per_image_whitening(jpg)
  jpg = tf.image.random_crop(jpg, [128, 128])
  jpg = tf.expand_dims(jpg,0)
  #jpg = tf.image.resize_bilinear(jpg, [128, 128])

with tf.name_scope("gen") as scope:
  #inp = tf.placeholder("float", [128])
  #inp2 = tf.Variable(tf.truncated_normal([128], dtype=tf.float32, stddev=1))
  inp = tf.truncated_normal([128], dtype=tf.float32, stddev=1)
  
  l10 = tf.reshape(inp, [1, 8, 8, 2])
  l10 = utils._deconv(l10, 3, 3, 20)
  l10 = utils._double(l10)
  l10 = utils._deconv(l10, 3, 3, 20)
  l10 = utils._double(l10)
  l10 = utils._deconv(l10, 3, 3, 20)
  l10 = utils._double(l10)
  l10 = utils._deconv(l10, 3, 3, 20)
  l10 = utils._double(l10)
  l10 = utils._deconv(l10, 3, 3, 3, relu=False)


adv_inp = tf.concat(0, [l10, jpg])
tf.image_summary("gen", adv_inp, max_images=1000000)
answers = tf.constant([[0.0,1.0], [1.0,0.0]])

with tf.name_scope("adv") as scope:
  al2 = utils._conv(adv_inp, 3, 3, 15)
  al3 = utils._half(al2)
  al4 = utils._conv(al3, 3, 3, 15)
  al5 = utils._half(al4)
  al6 = utils._conv(al5, 3, 3, 15)
  al7 = utils._half(al6)
  al8 = utils._conv(al7, 3, 3, 15)
  al9 = utils._half(al8)
  al10 = utils._conv(al9, 3, 3, 15)
  al = utils._half(al10)
  al = utils._conv(al, 3, 3, 15)
  al12 = utils._fc(al, 2)
  al13 = tf.nn.softmax(al12)


cross_entropy = -tf.reduce_sum(answers*tf.log(al13))

tf.scalar_summary("cross_entropy", cross_entropy)

opt = tf.train.AdagradOptimizer(1e-2)

# Compute the gradients for a list of variables.
grads_and_vars = opt.compute_gradients(cross_entropy)


for g,v in grads_and_vars:
  print "G: " + str(g)
  print "V: " + str(v.name)
  
grads_and_vars = [ (-g/cross_entropy*tf.exp(-cross_entropy),v) if "gen" in v.name else ((tf.constant([1.0]) - tf.exp(-cross_entropy)) * g,v) for g, v in grads_and_vars]

# grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# need to the 'gradient' part, for example cap them, etc.
capped_grads_and_vars = grads_and_vars # 

# Ask the optimizer to apply the capped gradients.
train_step = opt.apply_gradients(capped_grads_and_vars)

# another thinggy...
# train_step = tf.train.AdagradOptimizer(3e-2).minimize(tf.reduce_mean(tf.square(l10-jpg)))



plt.ion()

with tf.Session() as sess:
  testin = tf.truncated_normal([128], dtype=tf.float32, stddev=1)
  
  merged_summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs/'+str(datetime.now()), sess.graph_def)

  init = tf.initialize_all_variables()
  sess.run(init)
  testin2 = testin.eval()
  print "variables initialized"

  for i in range(1000000):
     print i
     # l10, cross_entropy, train_step
     # ddd = [g for g,v in grads_and_vars];
     #ddd = [tf.log(al13)]
     tr = sess.run([train_step])
     
     if not (i%10):
       summary_str, out, ce, gg = sess.run([merged_summary_op, l10, cross_entropy, al13], feed_dict={inp: testin.eval()})
       summary_writer.add_summary(summary_str, i)

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

