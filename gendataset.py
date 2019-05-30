import numpy as np
from PIL import Image
import os
import codec
from tqdm import tqdm
import setting

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getimage(array:np.ndarray):
    img=array.reshape((3,32,32))
    img=img.transpose((1,2,0))
    return img

if __name__ == '__main__':
    datadir='dataset/cifar-10-batches-py'
    try:
        os.mkdir('dataset/stged')
        os.mkdir('dataset/unstged')
        os.mkdir('test')

    except:
        ...
    for i in range(1,6):
        datafile=os.path.join(datadir,'data_batch_'+str(i))
        data=unpickle(datafile)[b'data']
        for j,k in enumerate(data):
            print(j)
            imgarray=getimage(k)
            img=Image.fromarray(imgarray)
            stgimg=codec.encodeDataInImage(img,''.join((chr(i)for i in np.random.randint(97,123,[np.random.randint(setting.n)]))))
            stgimg.save(os.path.join('dataset/stged','%d.png'%((i-1)*10000+j)))
            img.save(os.path.join('dataset/unstged','%d.png'%((i-1)*10000+j)))
    datafile='dataset/cifar-10-batches-py/test_batch'
    data = unpickle(datafile)[b'data']
    for j, k in enumerate(data):
        print(j)
        imgarray = getimage(k)
        img = Image.fromarray(imgarray)
        img.save(os.path.join('test', '%d.png' % (j)))