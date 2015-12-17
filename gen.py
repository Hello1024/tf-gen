import tensorflow as tf
import utils
import time
from datetime import datetime
import matplotlib.pyplot as plt

BATCH_SIZE = 5

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
  al = utils._conv(adv_inp, 5, 5, 10)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 15)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 25)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 25)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 35)
  al = utils._half(al)
  al = utils._conv(al, 3, 3, 35)
  al = utils._fc(al, 2)
  al = tf.nn.softmax(al)


def normclip(grads_and_vars):
  #for g,v in grads_and_vars:
  #  print "G: " + str(g)
  #  print "V: " + str(v.name)
  return grads_and_vars
  return [ (tf.clip_by_norm(g, 1.0),v) for g, v in grads_and_vars]


adv_entropy = -tf.reduce_sum(answers*tf.log(tf.clip_by_value(al, 1e-10, 1.0))) / BATCH_SIZE
tf.scalar_summary("adv_entropy", adv_entropy)
adv_opt = tf.train.AdagradOptimizer(1e-2)
adv_train_step = adv_opt.apply_gradients(normclip(adv_opt.compute_gradients(adv_entropy, var_list=[x for x in tf.trainable_variables() if "adv" in x.name])))


gen_entropy = -tf.reduce_sum((1.0-answers)*tf.log(tf.clip_by_value(al, 1e-10, 1.0))) / BATCH_SIZE
tf.scalar_summary("gen_entropy", gen_entropy)
gen_opt = tf.train.AdagradOptimizer(1e-1)
gen_train_step = gen_opt.apply_gradients(normclip(gen_opt.compute_gradients(gen_entropy, var_list=[x for x in tf.trainable_variables() if "gen" in x.name])))



saver = tf.train.Saver()

plt.ion()

grad_op = tf.gradients(gen_entropy, res)[0]*100
print grad_op

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
     tr = sess.run([adv_train_step, gen_train_step])
     
     if not (i%30):
       summary_str, out, ae, ge, gg, dif = sess.run([merged_summary_op, res, adv_entropy, gen_entropy, al, grad_op], feed_dict={inp: testin})
       summary_writer.add_summary(summary_str, i)

       save_path = saver.save(sess, log_path+ "/model.ckpt")

       print gg
       print "adv entropy: " + str(ae)
       print "gen entropy: " + str(ge)
       plt.subplot(1, 2, 1)
       plt.imshow((out[0,:,:,0:3]/3+0.5).clip(min=0, max=1))
       plt.subplot(1, 2, 2)
       plt.imshow((dif[0,:,:,0:3]/3+0.5).clip(min=0, max=1))
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

