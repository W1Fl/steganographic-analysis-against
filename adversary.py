import tensorflow as tf
import numpy as np
from PIL import Image
import steganalysis
import codec
import os

k=0.0001


input_img='teststg/stg.png'
imagearray=np.array(Image.open(input_img))
img=tf.constant(imagearray.reshape((1,32,32,3)),tf.float32,name='input_img')

b=tf.get_variable('noise',[1,32,32,3],tf.float32,tf.zeros_initializer())
l=tf.nn.l2_loss(b)

adversary_sample=tf.add(img,b,'noise_add')

_,out=steganalysis.inference_op(adversary_sample,[[0]],1)

loss=tf.add(out,l*k,'sumloss')

vars=tf.contrib.framework.get_variables_to_restore()
print(vars)
saver = tf.train.Saver(vars[1:])
sess=tf.Session()

trainer=tf.train.AdamOptimizer(0.1).minimize(loss,var_list=b)
tf.summary.FileWriter("alogs", sess.graph)

sess.run(tf.global_variables_initializer())
saver.restore(sess, 'model/steganalysismodel-200')

for i in range(10000):
    _,outvalue,l2,lossvalue=sess.run([trainer,out,l,loss])
    if not i%10:
        print('第',i,'次迭代','分析器输出为',outvalue[0][0],'噪声范数为',l2,'损失为',lossvalue[0][0])

noise=sess.run(b)[0]
if noise.max()<2:
    k=1/noise.max()*2
    noise*=k
    newimage=np.uint8(imagearray+np.round(noise/2)*2)
    newimage=Image.fromarray(newimage)
    newimage.save('out/out.png')
    print('请运行work.py进行隐写分析')
    os._exit(0)
else:
    print('需要调整,噪声达到',noise.max())