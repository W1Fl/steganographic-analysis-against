import numpy as np
import tensorflow as tf

import dataset
import setting


def conv_op(input_op, name, kh, kw, n_out, dh, dw):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.01))
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(1, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        return activation


def hpf_op(input_op, name):
    with tf.name_scope(name):
        hpf_kernel = np.array([
            [-1, 1, -1],
            [-1, 5, -1],
            [-1, 1, -1],
        ], np.float).reshape((3, 3, 1, 1))
        hpf_kernel = hpf_kernel * np.eye(3, 3)
        hpf_kernel = tf.constant(hpf_kernel, tf.float32)
        conv = tf.nn.conv2d(input_op, hpf_kernel, (1, 1, 1, 1), padding='SAME')
        return conv


def fc_op(input_op, name, n_out, actfunc):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.01))
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = actfunc(input_op @ kernel + biases, name=scope)
        return activation


def inference_op(input_op, labels, kp):
    hpf0 = hpf_op(input_op, 'hpf')

    sp = hpf0[:, :3, :, :]

    conv1_1 = conv_op(sp, name='conv1_1', kh=3, kw=5, n_out=32, dh=1, dw=2)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=32, dh=1, dw=2)

    conv2_1 = conv_op(conv1_2, name='conv2_1', kh=3, kw=5, n_out=64, dh=1, dw=2)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=64, dh=1, dw=2)

    shp = conv2_2.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh = tf.reshape(conv2_2, [-1, flattened_shape], name="resh")

    fc3 = fc_op(resh, name="fc3", n_out=1024, actfunc=tf.nn.relu)
    fc3_drop = tf.nn.dropout(fc3, kp, name='fc3_drop')

    fc4 = fc_op(fc3_drop, name="fc4", n_out=128, actfunc=tf.nn.relu)
    fc4_drop = tf.nn.dropout(fc4, kp, name='fc4_drop')
    out_op = fc_op(fc4_drop, name="output", n_out=1, actfunc=tf.nn.sigmoid)

    loss = tf.losses.mean_squared_error(out_op, labels)
    tf.summary.scalar('loss', loss)
    return loss, out_op


# 训练
def train():
    x = tf.placeholder(tf.float32, [None, *setting.size, 3], 'input')
    label = tf.placeholder(tf.float32, [None, 1], 'label')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    loss, y = inference_op(x, label, keep_prob)
    trainer = tf.train.AdamOptimizer(setting.learnning_rate).minimize(tf.log(loss))

    labelarray, imagearray = dataset.loadimage(setting.datasetfromfile)
    trainlabel, trainimage, testlabel, testimage, validlabel, validimage = dataset.splitdataset(labelarray, imagearray,setting.datasetfromfile)

    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    trainwriter = tf.summary.FileWriter("logs/train", sess.graph)
    testwriter = tf.summary.FileWriter("logs/test", sess.graph)
    for i in range(setting.epoch):
        for j in range(setting.step):
            trainbatchlabel, trainbatchimage,validbatchlabel,validbatchimage = dataset.next_batch(setting.batch_size,
                                                                                                  trainlabel,
                                                                                                  trainimage,
                                                                                                  validlabel,
                                                                                                  validimage)

            _, trainloss, out, rs = sess.run([trainer, loss, y, merged],
                                             {x: trainbatchimage, label: trainbatchlabel,
                                              keep_prob: setting.keep_prob, })
            if not j % 10:
                trainwriter.add_summary(rs, i*setting.step+j)
                trainwriter.flush()
                trainacc = np.equal(out > 0.5, trainbatchlabel).mean()
                validloss, rs,out = sess.run([loss, merged,y],
                                             {x:validbatchimage, label: validbatchlabel, keep_prob: 1})
                validacc = np.equal(out > 0.5, validbatchlabel).mean()
                testwriter.add_summary(rs, i*setting.step+j)
                testwriter.flush()
                print('第%d轮第%d批训练,训练损失为%f,验证损失为%f,训练集准确率为%f,验证集准确率为%f' % (i, j, trainloss, validloss, trainacc,validacc))
        saver.save(sess, 'model/steganalysismodel', i)


if __name__ == '__main__':
    train()
