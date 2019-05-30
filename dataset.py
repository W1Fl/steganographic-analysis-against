import os

import numpy as np
from PIL import Image
from scipy.io import savemat,loadmat
from tqdm import tqdm


def loadimage(fromfile: bool):
    if fromfile:
        labelarray = np.load('dataset/label.npy')
        imagearray = np.load('dataset/image.npy')
    else:
        url = 'dataset/'
        imagearray = []
        labelarray = []
        unstgurl = os.path.join(url, 'unstged')
        unstgeddir = os.listdir(unstgurl)
        stgurl = os.path.join(url, 'stged')
        stgeddir = os.listdir(stgurl)
        assert stgeddir == unstgeddir

        print('加载未隐写图片')
        for i in tqdm(stgeddir):
            try:
                img = np.array(Image.open(os.path.join(stgurl, i)))
                imagearray.append(img)
                labelarray.append([1])
                img = np.array(Image.open(os.path.join(unstgurl, i)))
                imagearray.append(img)
                labelarray.append([0])
            except:
                print(i)

        print('加载隐写图片')

        labelarray = np.array(labelarray)
        imagearray = np.array(imagearray)
        np.save('dataset/label.npy', labelarray)
        np.save('dataset/image.npy', imagearray)
    return labelarray, imagearray


def splitdataset(labelarray: np.ndarray, imagearray: np.ndarray,readmat:bool):
    getdictvalue=lambda d:(d['trainlabel'],d['trainimage'],d['testlabel'],d['testimage'],d['validlabel'], d['validimage'])
    if readmat:
        datadict=loadmat('dataset/splitedDataset.mat')
    else:
        testid = []
        trainid = []
        validid = []
        for i in range(labelarray.shape[0] // 2):
            rand = np.random.rand()
            if rand < 0.15:
                testid.append(i * 2)
                testid.append(i * 2 + 1)
            elif rand < 0.3:
                validid.append(i * 2)
                validid.append(i * 2 + 1)
            else:
                trainid.append(i * 2)
                trainid.append(i * 2 + 1)

        datadict = dict(trainlabel=labelarray[trainid], trainimage=imagearray[trainid], testlabel=labelarray[testid],
                    testimage=imagearray[testid], validlabel=labelarray[validid], validimage=imagearray[validid])
        savemat('dataset/splitedDataset.mat', datadict)
    return getdictvalue(datadict)


def next_batch(batch_size, trainlabel, trainimage, validlabel, validimage):
    def getdata(m):
        label = [trainlabel, validlabel][m]
        image = [trainimage, validimage][m]
        index = np.random.randint(0, label.shape[0], batch_size) // 2 * 2
        index = [*index, *(index + 1)]
        batchlabel = label[index]
        batchimage = image[index]

        return batchlabel, batchimage

    return [*getdata(0), *getdata(1)]

if __name__ == '__main__':
    a=loadimage(True)
    b=splitdataset(*a,readmat=True)