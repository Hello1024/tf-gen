import tensorflow as tf
import utils
import time
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import random

import superres_model

BATCH_SIZE = 1

with tf.name_scope("input"):
  jpg = tf.image.decode_jpeg(open('city.jpg', 'r').read())
  jpg.set_shape([100,100,3])  # Bodge to bypass checks later...
  jpg = tf.image.per_image_whitening(jpg)
  crops = []
  for i in range(BATCH_SIZE):
    crops.append(tf.image.random_crop(jpg, [512, 512]))
  jpg = tf.pack(crops)
  #jpg = tf.image.resize_bilinear(jpg, [128, 128])

mix = tf.random_uniform([1]);
net_in = mix*jpg + (1.0-mix)*tf.truncated_normal(jpg.get_shape(), dtype=tf.float32, stddev=1)
l10 = net_in


all_images = []
  
with tf.variable_scope("gen", reuse=None) as scope:
  with tf.name_scope("1"):
    l10 = superres_model.superres_model(l10)
  all_images.append(l10)
with tf.variable_scope("gen", reuse=True) as scope:
  for i in range(2,8):
    with tf.name_scope(str(i)):
      l10 = superres_model.superres_model(l10)
    all_images.append(l10)

measure_images = tf.concat(0, [image[:,:,:,0:3] for image in [all_images[i] for i in [0, 2, 4, 6]]])
feedback_images = tf.concat(0, [image[:,:,:,0:3] for image in [all_images[i] for i in [6]]])
show_images = tf.concat(0, [image[0:1,:,:,0:3] for image in all_images])


tf.image_summary("gen", measure_images, max_images=1)
tf.image_summary("real", jpg, max_images=1)

adv_inp = tf.concat(0, [measure_images, jpg])
answers = tf.concat(0, [
  tf.tile(tf.constant([[0.0,1.0]]), [BATCH_SIZE*4,1]),  
  tf.tile(tf.constant([[1.0,0.0]]), [BATCH_SIZE,1])  
])

with tf.name_scope("adv") as scope:
  al = utils._conv(adv_inp, 5, 5, 10)
  al = utils._half(al)
  al = utils._conv(al, 5, 5, 15)
  al = utils._half(al)
  al = utils._conv(al, 5, 5, 25)
  al = utils._half(al)
  al = utils._conv(al, 5, 5, 25)
  al = utils._half(al)
  al = utils._conv(al, 5, 5, 35)
  al = utils._half(al)
  al = utils._conv(al, 5, 5, 35)
  al = utils._half(al)
  al = utils._conv(al, 5, 5, 35)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 35)
  al = utils._fc(al, 2)
  al = tf.nn.softmax(al)


def normclip(grads_and_vars):
  #for g,v in grads_and_vars:
  #  print "G: " + str(g)
  #  print "V: " + str(v.name)
  #return grads_and_vars
  return [ (tf.clip_by_norm(g, 1.0),v) for g, v in grads_and_vars]


adv_entropy = -tf.reduce_sum(answers*tf.log(tf.clip_by_value(al, 1e-10, 1.0))) / BATCH_SIZE
tf.scalar_summary("adv_entropy", adv_entropy)
adv_opt = tf.train.AdagradOptimizer(2e-2)
adv_train_step = adv_opt.apply_gradients(normclip(adv_opt.compute_gradients(adv_entropy, var_list=[x for x in tf.trainable_variables() if "adv" in x.name])))


gen_entropy = -tf.reduce_sum((1.0-answers)*tf.log(tf.clip_by_value(al, 1e-10, 1.0))) / BATCH_SIZE
tf.scalar_summary("gen_entropy", gen_entropy)
gen_opt = tf.train.AdagradOptimizer(2e-2)
gen_train_step = gen_opt.apply_gradients(normclip(gen_opt.compute_gradients(gen_entropy, var_list=[x for x in tf.trainable_variables() if "gen" in x.name])))


saver = tf.train.Saver()

plt.ion()

grad_op = tf.gradients(gen_entropy, adv_inp)[0]*300

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

with tf.Session(config=config) as sess:
  log_path = '/tmp/superres/'+datetime.now().strftime("%Y/%m/%d/%H/%M/%S")
  merged_summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(log_path, sess.graph_def)

  init = tf.initialize_all_variables()
  sess.run(init)
  print "variables initialized"

  if len(sys.argv)>1:
    saver.restore(sess, sys.argv[1])


  ae=1.0

  feedback_images_res = None

  for i in range(1000000):

     if feedback_images_res is not None and random.random()<0.90:
       feed_dict={net_in: feedback_images_res}
     else:
       print "new"
       feed_dict={}
       
     if not (i%10):
       summary_str, out, ae, ge, gg, dif, show_images_res = sess.run([merged_summary_op, adv_inp, adv_entropy, gen_entropy, al, grad_op, show_images], feed_dict=feed_dict)
       summary_writer.add_summary(summary_str, i)

       save_path = saver.save(sess, log_path+ "/model.ckpt")

       print gg
       
       #print "adv entropy: " + str(ae)
       #print "gen entropy: " + str(ge)
       
       #for b in range(4):
       #  plt.subplot(4, 2, b*2+1)
       #  plt.imshow((out[b*BATCH_SIZE,:,:,0:3]/3+0.5).clip(min=0, max=1))
       #  plt.subplot(4, 2, b*2+2)
       #  plt.imshow((dif[b*BATCH_SIZE,:,:,0:3]+0.5).clip(min=0, max=1))
       for b in range(show_images_res.shape[0]):
         plt.subplot(2, 5, b+1)
         plt.imshow((show_images_res[b,:,:,:]/3+0.5).clip(min=0, max=1))
       
       plt.pause(0.1)


     if ae*5 < ge:
       chosen_step = gen_train_step
     else:
       chosen_step = adv_train_step

     ae, ge, feedback_images_res, _ = sess.run([adv_entropy, gen_entropy, feedback_images, chosen_step], feed_dict=feed_dict)
     print i, ae, ge

     
     #for j in range(len(gv)):
     #  print "-----------------"
     #  print ddd[j].name
     #  print gv[j]
     
     # summary_str = sess.run(merged_summary_op, feed_dict={inp: testin.eval()})
     # summary_writer.add_summary(summary_str, i)

#print prob


