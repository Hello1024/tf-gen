import tensorflow as tf
import utils
import time
import sys
from datetime import datetime
import matplotlib.pyplot as plt

BATCH_SIZE = 10

with tf.name_scope("input"):
  jpg = tf.image.decode_jpeg(open('brick.jpg', 'r').read())
  jpg.set_shape([100,100,3])  # Bodge to bypass checks later...
  jpg = tf.image.per_image_whitening(jpg)
  crops = []
  for i in range(BATCH_SIZE):
    crops.append(tf.image.random_crop(jpg, [128, 128]))
  jpg = tf.pack(crops)
  #jpg = tf.image.resize_bilinear(jpg, [128, 128])

mix = tf.random_uniform([1]);
l10 = mix*jpg + (1.0-mix)*tf.truncated_normal(jpg.get_shape(), dtype=tf.float32, stddev=1)

def superres_layer(inp, name=None):
  results = []
  inp_depth = int(inp.get_shape()[-1])
  if inp_depth != 20:
    inp = tf.pad(inp, [[0,0], [0,0], [0,0], [0,20-inp_depth]])
  #results.append(inp[:,:,:,0:3])
  results.append(utils._deconv(inp, 3, 3, 3, relu=False, name="norelu"+name) + inp[:,:,:,0:3])
  results.append(utils._deconv(inp, 5, 5, 10, name=name))
  results.append(tf.nn.max_pool(inp[:,:,:,3:5], [1,8,8,1], [1,1,1,1], 'SAME'))
  results.append(tf.nn.max_pool(inp[:,:,:,5:7], [1,4,4,1], [1,1,1,1], 'SAME'))
  results.append(tf.nn.max_pool(inp[:,:,:,7:9], [1,2,2,1], [1,1,1,1], 'SAME'))
  results.append(tf.truncated_normal(inp[:,:,:,0:1].get_shape(), dtype=tf.float32, stddev=1))
  
  return tf.concat(3, results)

with tf.variable_scope("gen", reuse=None) as scope:
  with tf.name_scope("1"):
    l10 = superres_layer(l10, name="1")
    l10 = superres_layer(l10, name="2")
    l10 = superres_layer(l10, name="3")
  res = l10[:,:,:,0:3]

with tf.variable_scope("gen", reuse=True) as scope:

  with tf.name_scope("2"):
    l10 = superres_layer(l10, name="1")
    l10 = superres_layer(l10, name="2")
    l10 = superres_layer(l10, name="3")
  with tf.name_scope("3"):
    l10 = superres_layer(l10, name="1")
    l10 = superres_layer(l10, name="2")
    l10 = superres_layer(l10, name="3")
  res = tf.concat(0, [res, l10[:,:,:,0:3]])
  with tf.name_scope("4"):
    l10 = superres_layer(l10, name="1")
    l10 = superres_layer(l10, name="2")
    l10 = superres_layer(l10, name="3")
  with tf.name_scope("5"):
    l10 = superres_layer(l10, name="1")
    l10 = superres_layer(l10, name="2")
    l10 = superres_layer(l10, name="3")
  with tf.name_scope("6"):
    l10 = superres_layer(l10, name="1")
    l10 = superres_layer(l10, name="2")
    l10 = superres_layer(l10, name="3")
  res = tf.concat(0, [res, l10[:,:,:,0:3]])


tf.image_summary("gen", res, max_images=1)
tf.image_summary("real", jpg, max_images=1)

adv_inp = tf.concat(0, [res, jpg])
answers = tf.concat(0, [
  tf.tile(tf.constant([[0.0,1.0]]), [BATCH_SIZE*3,1]),  
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
adv_opt = tf.train.AdagradOptimizer(7e-3)
adv_train_step = adv_opt.apply_gradients(normclip(adv_opt.compute_gradients(adv_entropy, var_list=[x for x in tf.trainable_variables() if "adv" in x.name])))


gen_entropy = -tf.reduce_sum((1.0-answers)*tf.log(tf.clip_by_value(al, 1e-10, 1.0))) / BATCH_SIZE
tf.scalar_summary("gen_entropy", gen_entropy)
gen_opt = tf.train.AdagradOptimizer(7e-2)
gen_train_step = gen_opt.apply_gradients(normclip(gen_opt.compute_gradients(gen_entropy, var_list=[x for x in tf.trainable_variables() if "gen" in x.name])))


saver = tf.train.Saver()

plt.ion()

grad_op = tf.gradients(gen_entropy, adv_inp)[0]*300

with tf.Session() as sess:
  log_path = '/tmp/superres/'+datetime.now().strftime("%Y/%m/%d/%H/%M/%S")
  merged_summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(log_path, sess.graph_def)

  init = tf.initialize_all_variables()
  sess.run(init)
  print "variables initialized"

  if len(sys.argv)>1:
    saver.restore(sess, sys.argv[1])


  ae=1.0

  for i in range(1000000):
     # l10, cross_entropy, train_step
     # ddd = [g for g,v in grads_and_vars];
     #ddd = [tf.log(al13)]
     
     if not (i%30):
       summary_str, out, ae, ge, gg, dif = sess.run([merged_summary_op, adv_inp, adv_entropy, gen_entropy, al, grad_op], feed_dict={})
       summary_writer.add_summary(summary_str, i)

       save_path = saver.save(sess, log_path+ "/model.ckpt")

       print gg
       print "adv entropy: " + str(ae)
       print "gen entropy: " + str(ge)
       for b in range(4):
         plt.subplot(4, 2, b*2+1)
         plt.imshow((out[b*BATCH_SIZE,:,:,0:3]/3+0.5).clip(min=0, max=1))
         plt.subplot(4, 2, b*2+2)
         plt.imshow((dif[b*BATCH_SIZE,:,:,0:3]+0.5).clip(min=0, max=1))
       plt.pause(0.1)


     if ae*5 < ge:
       chosen_step = gen_train_step
     else:
       chosen_step = adv_train_step
     ae = sess.run([adv_entropy, chosen_step])
     ae = ae[0]
     print i, ae

     
     #for j in range(len(gv)):
     #  print "-----------------"
     #  print ddd[j].name
     #  print gv[j]
     
     # summary_str = sess.run(merged_summary_op, feed_dict={inp: testin.eval()})
     # summary_writer.add_summary(summary_str, i)

#print prob
plt.imshow(res[0,:,:,:])
plt.show()

