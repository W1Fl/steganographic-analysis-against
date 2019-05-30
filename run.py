from PIL import Image
import os
import codec
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

imgurl=os.path.join('test',input('请输入目标图片的编号--> ')+'.png')
img=Image.open(imgurl)
import work
sc=work.comput(np.array(img))[0][0]
print('分析器认为原图有{}%的概率被隐写过'.format(sc*100))
stgimg=codec.encodeDataInImage(img,input('请输入需要隐写的信息--> '))
sc=work.comput(np.array(stgimg))[0][0]
print('分析器认为被隐写后的图有{}%的概率被隐写过'.format(sc*100))
try:
    os.mkdir('teststg')
except:
    ...
stgimg.save('teststg/stg.png')
work.sess.close()
print('被隐写的图片保存到了teststg目录下,请运行adversary.py生成对抗样本')
os._exit(0)