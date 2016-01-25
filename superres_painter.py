import tensorflow as tf
import utils
import time
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from random import randint

import numpy as np
import Tkinter
from Tkinter import *
from PIL import Image, ImageTk
from skimage.draw import polygon

import superres_model

b1 = "up"
xold, yold = None, None
color = 1.0
canvas = None
photo = None
WIDTH, HEIGHT = 512, 512




def b1down(event):
    global b1, color
    b1 = "down"           # you only want to draw when the button is down
                          # because "Motion" events happen -all the time-
    color = 1.0
    if event.num==3:
      color = -1.0

def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None           # reset the line when you let go of the button
    yold = None

def goframe():
    global  canvas, photo, data, window
    feed_dict={in_x: data}
      
    out_dat = sess.run([out_x], feed_dict=feed_dict)

    data_new = out_dat[0][:,:,:,0:3]
    out_dat = np.clip(data_new[0,:,:,:]*128+128, 0, 255)


    im=Image.fromstring('RGB', (out_dat.shape[1],\
                        out_dat.shape[0]), out_dat.astype('b').tostring())
    photo = ImageTk.PhotoImage(image=im)
    canvas.create_image(0,0,image=photo,anchor=Tkinter.NW)
    canvas.i = photo
    
    window.after(1000, goframe)


def motion(event):
    if b1 == "down":
        global xold, yold, photo, data, color
        if xold is not None and yold is not None and photo is not None:
          x = np.array([xold, xold+25, event.x+25, event.x])
          y = np.array([yold, yold+25, event.y+25, event.y])
          rr, cc = polygon(y, x, (WIDTH, HEIGHT))
          
          for i in range(3):
            data[0,rr, cc, i] += color
                          # here's where you draw it. smooth. neat.
        xold = event.x
        yold = event.y



data=np.array((np.random.random((1, HEIGHT, WIDTH, 3))-0.5)*4,dtype=float)

in_x = tf.placeholder(tf.float32, shape=(1, WIDTH, HEIGHT, 3))

with tf.variable_scope("gen", reuse=None) as scope:
  with tf.name_scope("1"):
    out_x = superres_model.superres_model(in_x)
with tf.variable_scope("gen", reuse=True) as scope:
  for i in range(6):
    with tf.name_scope(str(i+2)):
      out_x = superres_model.superres_model(out_x)

restore_list = [x for x in tf.trainable_variables() if "ssskkds" not in x.name]

saver = tf.train.Saver(var_list = restore_list)

sess = tf.Session()

saver.restore(sess, sys.argv[1])



window = Tk()
canvas = Canvas(window, width=WIDTH, height=HEIGHT, bg="#000000")
canvas.pack()


canvas.bind("<Motion>", motion)
canvas.bind("<ButtonPress>", b1down)
canvas.bind("<ButtonRelease>", b1up)
goframe()
mainloop()
 

     


