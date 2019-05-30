import tensorflow as tf
from PIL import Image
import numpy as np
import codec
import dataset

saver = tf.train.import_meta_graph('model/steganalysismodel-200.meta')
sess=tf.Session()
saver.restore(sess, 'model/steganalysismodel-200')
graph=tf.get_default_graph()

x=graph.get_tensor_by_name('input:0')
y=graph.get_tensor_by_name('output:0')
keep_prob=graph.get_tensor_by_name('keep_prob:0')

def comput(img):
    img=np.array(img)
    res=sess.run(y,{x:[img],keep_prob:1})
    return res


def test():
    _,_,testlabel,testimage,_,_=dataset.splitdataset(None,None,True)
    res = sess.run(y, {x: testimage, keep_prob: 1})
    testacc=np.equal(res > 0.5, testlabel).mean()
    print('测试集正确率为',testacc*100,'%')

if __name__ == '__main__':
    test()