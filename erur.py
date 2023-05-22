# encoding: utf-8
import tensorflow.compat.v1 as tf #原有的1.x的API整理到tensorflow.compat.v1这个包里了
tf.disable_v2_behavior() # 禁用2.0默认的的即时执行模式
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import dtypes

import numpy as np
from ExtractData import Dataset
from GetTest import get_test_set2, get_train_instance_batch_test_aux
from time import time
import os

def ini_word_embed(num_words, latent_dim):
    word_embeds = np.random.rand(num_words, latent_dim)
    return word_embeds

def word2vec_word_embed(num_words, latent_dim, path, word_id_dict):
    word2vect_embed_mtrx = np.zeros((num_words, latent_dim))
    with open(path, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            row_id = word_id_dict.get(arr[0])
            vect = arr[1].strip().split(" ")
            for i in range(len(vect)):
                word2vect_embed_mtrx[row_id, i] = float(vect[i])
            line = f.readline()

    return word2vect_embed_mtrx


def user_embeds_model(word_latent_factor, latent_dim, num_filters, windows_size, users_inputs, users_masks_inputs, word_embeddings):
    user_reviews_representation = tf.nn.embedding_lookup(word_embeddings, users_inputs)
    user_reviews_representation = user_reviews_representation * tf.expand_dims(users_masks_inputs, -1)
    user_reviews_representation_expnd = tf.expand_dims(user_reviews_representation, -1)
    # print "user_reviews_representation_expnd", user_reviews_representation_expnd.get_shape()

    #CNN layers
    W = tf.Variable(tf.truncated_normal([windows_size, word_latent_factor, 1, num_filters],stddev=0.3), name="user_W")
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="user_b")

    conv = tf.nn.conv2d(user_reviews_representation_expnd, W, strides=[1,1,1,1], padding="VALID", name="user_conv")
    # print "conv", conv.get_shape()

    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="user_relu")
    # print "h", h.get_shape()
    sec_dim = h.get_shape()[1]
    print(sec_dim)

    o = tf.nn.max_pool(
        h,
        ksize=[1, sec_dim, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="user_pool")
    # print "o", o.get_shape()
    o = tf.squeeze(o)
    # print "o", o.get_shape()

    W1 = tf.Variable(tf.truncated_normal([num_filters, latent_dim],stddev=0.3), name="user_W1")


    b1 = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="user_b")

    user_vector = tf.nn.relu_layer(o, W1, b1, name="user_layer1")
    # print "user_vector", user_vector.get_shape()

    return user_vector

def item_embeds_model(word_latent_factor, latent_dim, num_filters, windows_size, items_inputs, items_masks_inputs, word_embeddings):
    item_reviews_representation = tf.nn.embedding_lookup(word_embeddings, items_inputs)
    item_reviews_representation = item_reviews_representation * tf.expand_dims(items_masks_inputs, -1)
    item_reviews_representation_expnd = tf.expand_dims(item_reviews_representation, -1)

    #CNN layers
    W = tf.Variable(tf.truncated_normal([windows_size, word_latent_factor, 1, num_filters],stddev=0.3), name="item_W")
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="user_b")

    conv = tf.nn.conv2d(item_reviews_representation_expnd, W, strides=[1,1,1,1], padding="VALID", name="item_conv")
    #print "conv", conv.get_shape()

    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="item_relu")
    #print "h", h.get_shape()
    sec_dim = h.get_shape()[1]

    o = tf.nn.max_pool(
        h,
        ksize=[1, sec_dim, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="item_pool")
    #print "o", o.get_shape()
    o = tf.squeeze(o)
    #print "o", o.get_shape()

    W1 = tf.Variable(tf.truncated_normal([num_filters, latent_dim],stddev=0.3), name="item_W1")


    b1 = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="item_b")

    item_vector = tf.nn.relu_layer(o, W1, b1, name="item_layer1")
    # print "item_vector", item_vector.get_shape()

    return item_vector


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""

    init_range = np.sqrt(6 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def InnerProduct(inputs, dropout_rate):
    inputs = tf.nn.dropout(inputs, dropout_rate)
    x = tf.transpose(inputs)
    x = tf.matmul(inputs, x)
    # x = tf.reshape(x, [-1])
    # x = from_dense(x)

    InnerP = tf.nn.relu(x)

    features = InnerP

    zero = tf.constant(0, dtype=tf.float32)

    where = tf.not_equal(features, zero)

    indices = tf.where(where)

    indices = tf.transpose(indices)
    return indices


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / int(input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def from_dense(tensor, name=None):
  """Converts a dense tensor into a sparse tensor.

  Only elements not equal to zero will be present in the result. The resulting
  `SparseTensor` has the same dtype and shape as the input.

  Args:
    tensor: A dense `Tensor` to be converted to a `SparseTensor`.
    name: Optional name for the op.

  Returns:
    The `SparseTensor`.
  """
  with ops.name_scope(name, "dense_to_sparse"):
    tensor = ops.convert_to_tensor(tensor)
    indices = array_ops.where_v2(
        math_ops.not_equal(tensor, array_ops.constant(0, tensor.dtype)))
    values = array_ops.gather_nd(tensor, indices)
    shape = array_ops.shape(tensor, out_type=dtypes.int64)
    return sparse_tensor.SparseTensor(indices, values, shape)

def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # np.random.seed(1)
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


def GraphLearn(user_embeds, dropout_rate, num_nodes, users_inputs):
    # graph learning
    input_dim = 32
    output_dim = 32
    S = InnerProduct(user_embeds, dropout_rate)

    weights = glorot([input_dim, output_dim], name='weights')
    a = glorot([output_dim, 1], name='a')  # 对应公式中的a
    bias = zeros([output_dim], name='bias')

    x = user_embeds
    x = tf.nn.dropout(x, dropout_rate)
    x = from_dense(x)

    h = tf.sparse_tensor_dense_matmul(x, weights)
    edge = S
    N = num_nodes

    edge_v = tf.abs(tf.gather(h, edge[0]) - tf.gather(h, edge[1]))
    edge_v = tf.squeeze(tf.nn.relu(tf.matmul(edge_v, a)))
    sgraph = tf.SparseTensor(indices=tf.transpose(edge), values=edge_v, dense_shape=[N, N])
    sgraph = tf.sparse_tensor_to_dense(sgraph, default_value=0, validate_indices=True, name=None)
    sgraph = tf.nn.softmax(sgraph)

    #sgraph = tf.nn.softmax(edge_v)

    return h, sgraph



def GraphConvolution(user_embeds, sgraph, dropout_rate):
    input_dim = 32
    output_dim = 32
    x = user_embeds
    adj = sgraph
    print("adj.shape")
    print(adj.shape)
    weights = weight_variable_glorot(input_dim, output_dim, name="weights")
    x = tf.nn.dropout(x, dropout_rate)
    x = tf.matmul(x, weights)
    x = tf.matmul(adj, x)
    outputs = tf.nn.relu(x)
    return outputs

def Discriminator(inputs, reuse = False):
    with tf.variable_scope('Discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # np.random.seed(1)
        hidden = 32
        tf.set_random_seed(1)
        dc_den1 = tf.nn.relu(dense(inputs, hidden, hidden, name='dc_den1'))
        dc_den2 = tf.nn.relu(dense(dc_den1, hidden, hidden, name='dc_den2'))
        output = dense(dc_den2, hidden, 1, name='dc_output')
        return output



def get_train_instance(train):
    user_input, item_input, rates = [], [], []

    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        rates.append(train[u,i])
    return user_input, item_input, rates

def get_train_instance_batch_change(count, batch_size, user_input, item_input, ratings, user_reviews, item_reviews, user_aux_reviews, user_masks, item_masks, user_aux_masks):
    user_batch, item_batch, user_input_batch, item_input_batch, user_aux_input_batch, user_mask_batch, item_mask_batch, user_aux_mask_batch, labels_batch = [], [], [], [], [], [], [], [], []

    for idx in range(batch_size):
        index = (count*batch_size + idx) % len(user_input)
        user_batch.append(user_input[index])
        item_batch.append(item_input[index])
        user_input_batch.append(user_reviews.get(user_input[index]))
        user_mask_batch.append(user_masks.get(user_input[index]))
        item_input_batch.append(item_reviews.get(item_input[index]))
        item_mask_batch.append(item_masks.get(item_input[index]))
        user_aux_input_batch.append(user_aux_reviews.get(user_input[index]))
        user_aux_mask_batch.append(user_aux_masks.get(user_input[index]))
        labels_batch.append([ratings[index]])

    return user_batch, item_batch, user_input_batch, item_input_batch, user_aux_input_batch, user_mask_batch, item_mask_batch, user_aux_mask_batch, labels_batch

def highway_network(W_trans, b_trans, W, b, embed, drop_out):
    high_embed = tf.nn.relu_layer(embed, W_trans, b_trans)
    gate = tf.nn.sigmoid(tf.matmul(embed, W) + b)
    return tf.multiply(high_embed, gate) + tf.multiply((1 - gate), embed)


def train_model():
    users = tf.placeholder(tf.int32, shape=[None])
    items = tf.placeholder(tf.int32, shape=[None])
    users_inputs = tf.placeholder(tf.int32, shape=[None, len(user_reviews.get(0))])
    items_inputs = tf.placeholder(tf.int32, shape=[None, len(item_reviews.get(0))])
    users_masks_inputs = tf.placeholder(tf.float32, shape=[None, len(user_masks.get(0))])
    items_masks_inputs = tf.placeholder(tf.float32, shape=[None, len(item_masks.get(0))])
    users_aux_inputs = tf.placeholder(tf.int32, shape=[None, len(user_aux_reviews.get(0))])
    users_aux_masks_inputs = tf.placeholder(tf.float32, shape=[None, len(user_aux_masks.get(0))])
    rates_inputs = tf.placeholder(tf.float32, shape=[None, 1])
    drop_out_rate = tf.placeholder(tf.float32)

    num_nodes = tf.placeholder(tf.int32)
    z_real_dist = tf.placeholder(tf.float32)
    losslr1 = 0.01
    losslr2 = 0.0001

    word_embeddings = tf.Variable(word_embedding_mtrx, dtype=tf.float32)

    padding_embedding = tf.Variable(np.zeros([1, word_latent_dim]),dtype=tf.float32)

    word_embeddings = tf.concat([word_embeddings, padding_embedding], 0)

    user_bias = tf.Variable(tf.random_normal([num_user, 1], mean=0, stddev=0.02), name="user_bias")
    item_bias = tf.Variable(tf.random_normal([num_item, 1], mean=0, stddev=0.02), name="item_bias")
    user_bs = tf.nn.embedding_lookup(user_bias, users)
    item_bs = tf.nn.embedding_lookup(item_bias, items)

    user_embeds = user_embeds_model(word_latent_dim, num_factor, num_filters, cnn_windows_size, users_inputs, users_masks_inputs, word_embeddings)
    #user_embeds = tf.nn.dropout(user_embeds, drop_out_rate)

    item_embeds = item_embeds_model(word_latent_dim, num_factor, num_filters, cnn_windows_size, items_inputs, items_masks_inputs, word_embeddings)
    item_embeds = tf.nn.dropout(item_embeds, drop_out_rate)


    glx, sgraph = GraphLearn(user_embeds, drop_out_rate, num_nodes, users_inputs)

    user_embeds_1 = GraphConvolution(user_embeds, sgraph, drop_out_rate)
    user_embeds_2 = GraphConvolution(user_embeds_1, sgraph, drop_out_rate)
    user_embeds_3 = GraphConvolution(user_embeds_2, sgraph, drop_out_rate)
    user_embeds_4 = user_embeds_1 * 0.5 + user_embeds_2 * (1/3) + user_embeds_3 * (1/4)

    user_aux_embeds = user_embeds_4

    #user_aux_embeds = user_aux_embeds_model(word_latent_dim, num_factor, num_filters, cnn_windows_size, users_aux_inputs, users_aux_masks_inputs, word_embeddings)

    #highway network
    W_aux_trans = tf.Variable(tf.truncated_normal([num_factor, num_factor],stddev=0.3), name="user_aux_W_trans")
    b_aux_trans = tf.Variable(tf.constant(0.1, shape=[num_factor]), name="user_aux_b_trans")
    W_aux_high = tf.Variable(tf.truncated_normal([num_factor, num_factor], stddev=0.3), name="user_aux_W_high")
    b_aux_high = tf.Variable(tf.constant(-0.2, shape=[num_factor]), name="user_aux_b_high")

    user_aux_embeds = highway_network(W_aux_trans, b_aux_trans, W_aux_high, b_aux_high, user_aux_embeds, drop_out_rate)

    #gated mechanism
    W_u_gated = tf.Variable(tf.truncated_normal([num_factor, num_factor], stddev=0.3), name="user_W_gated")
    W_aux_gated = tf.Variable(tf.truncated_normal([num_factor, num_factor], stddev=0.3), name="user_aux_W_gated")
    W_i_gated = tf.Variable(tf.truncated_normal([num_factor, num_factor], stddev=0.3), name="item_W_gated")
    b_gated = tf.Variable(tf.constant(0.1, shape=[num_factor]), name="gated_b")
    W_gated = tf.nn.tanh(
        tf.matmul(user_embeds, W_u_gated) + tf.matmul(user_aux_embeds, W_aux_gated) + tf.matmul(item_embeds, W_i_gated) + b_gated)

    W_u = tf.Variable(tf.truncated_normal([2 * num_factor, num_factor], stddev=0.3), name="user_W_gated")
    b_u = tf.Variable(tf.constant(0.1, shape=[num_factor]), name="u_b")
    user_embeds_con = tf.concat([user_embeds, tf.multiply(user_aux_embeds, W_gated)], 1, name="user_concat_embed")
    user_embeds_con = tf.nn.relu_layer(user_embeds_con, W_u, b_u)

    user_embeds_con = tf.nn.dropout(user_embeds_con, drop_out_rate)
    item_embeds = tf.nn.dropout(item_embeds, drop_out_rate)

    embeds_sum = tf.concat([user_embeds_con, item_embeds], 1, name="concat_embed")

    w_0 = tf.Variable(tf.zeros(1), name="w_0")
    w_1 = tf.Variable(tf.truncated_normal([1, num_factor * 2],stddev=0.3), name="w_1")
    J_1 = w_0 + tf.matmul(embeds_sum, w_1, transpose_b=True)

    embeds_sum_1 = tf.expand_dims(embeds_sum, -1)
    embeds_sum_2 = tf.expand_dims(embeds_sum, 1)

    v = tf.Variable(tf.truncated_normal([num_factor * 2, 100],stddev=0.3), name="v")
    J_2 = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.matmul(embeds_sum_1, embeds_sum_2),tf.matmul(v, v, transpose_b=True)), 2), 1, keep_dims=True)
    J_3 = tf.trace(tf.multiply(tf.matmul(embeds_sum_1, embeds_sum_2),
                                 tf.matmul(v, v, transpose_b=True)))
    predict_rating = (J_1 + 0.5 * (J_2 - J_3)) + user_bs + item_bs
    loss = tf.reduce_mean(tf.squared_difference(predict_rating, rates_inputs)) + gama * tf.nn.l2_loss(tf.subtract(user_embeds, user_aux_embeds))

    # Loss
    # Graph Learning loss
    D = tf.matrix_diag(tf.ones(num_nodes)) * -1
    D = tf.add(D, sgraph) * -1
    D = tf.matmul(tf.transpose(glx), D)
    gl_loss = 0
    gl_loss += tf.trace(tf.matmul(D, glx)) * losslr1
    gl_loss -= tf.trace(tf.matmul(tf.transpose(sgraph), sgraph)) * losslr2

    # GAN loss


    d_real = Discriminator(z_real_dist)
    # d_real = Discriminator(user_embeds_1)
    # user_embed_grl = flip_gradient(user_embeds)
    d_fake = Discriminator(user_embeds)
    # GAN_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=d_real, logits=d_fake))

    # Discrimminator Loss
    dc_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_fake + dc_loss_real

    # Generator loss
    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

    GAN_loss = dc_loss + generator_loss


    total_loss = loss + gl_loss + GAN_loss





    train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(total_loss)
    init = tf.global_variables_initializer()

    user_val, item_val, user_input_val, item_input_val, user_aux_input_val, user_mask_input_val, item_mask_input_val, user_aux_mask_input_val, rating_input_val = get_test_set2(valRatings, user_reviews, item_reviews, user_aux_reviews, user_masks, item_masks, user_aux_masks)
    user_test, item_test, user_input_test, item_input_test, user_aux_input_test, user_mask_input_test, item_mask_input_test, user_aux_mask_input_test, rating_input_test = get_test_set2(testRatings, user_reviews, item_reviews, user_aux_reviews, user_masks, item_masks, user_aux_masks)

    with tf.Session() as sess:
        sess.run(init)

        best_mse, best_mae = 2, 2
        for e in range(epochs):
            t = time()
            loss_total = 0.0
            count = 0.0
            for i in range(int(len(user_input) / batch_size) + 1):
                user_batch, item_batch, user_input_batch, item_input_batch, user_aux_input_batch, user_mask_batch, item_mask_batch, user_aux_mask_batch, rates_batch = get_train_instance_batch_change(i, batch_size, user_input, item_input, rateings, user_reviews, item_reviews, user_aux_reviews, user_masks, item_masks, user_aux_masks)
                real_dist = np.random.randn(len(user_batch), 32)
                _, loss_val = sess.run([train_step, loss], feed_dict={users: user_batch, items: item_batch, users_inputs: user_input_batch, items_inputs: item_input_batch, users_aux_inputs: user_aux_input_batch,
                                                                      users_masks_inputs: user_mask_batch, items_masks_inputs: item_mask_batch,
                                                                      users_aux_masks_inputs: user_aux_mask_batch, rates_inputs: rates_batch, drop_out_rate: drop_rate, num_nodes: len(user_input_batch), z_real_dist:real_dist})
                loss_total += loss_val
                count += 1.0
            t1 = time()
            val_mse, val_mae = eval_model(users, items, users_inputs, items_inputs, users_aux_inputs, users_masks_inputs, items_masks_inputs, users_aux_masks_inputs, drop_out_rate, predict_rating, sess, user_val, item_val, user_input_val, item_input_val, user_aux_input_val, user_mask_input_val, item_mask_input_val, user_aux_mask_input_val, rating_input_val, num_nodes, z_real_dist)
            rmse, mae = eval_model(users, items, users_inputs, items_inputs, users_aux_inputs, users_masks_inputs, items_masks_inputs, users_aux_masks_inputs, drop_out_rate, predict_rating, sess, user_test, item_test, user_input_test, item_input_test, user_aux_input_test, user_mask_input_test, item_mask_input_test, user_aux_mask_input_test, rating_input_test, num_nodes, z_real_dist)
            print("epoch%d time: %.2fs, loss: %.2f, val_MSE: %.3f, MSE: %.3f, MAE: %.3f"%(e, t1 - t, loss_total/count, val_mse, rmse, mae))
            best_mse = rmse if rmse < best_mse else best_mse
            best_mae = mae if mae < best_mae else best_mae
        print("End. best_mse: %.6f" % (best_mse))
        print("End. best_mae: %.6f" % (best_mae))


def eval_model(users, items, users_inputs, items_inputs, users_aux_inputs, users_masks_inputs, items_masks_inputs, users_aux_masks_inputs, drop_out_rate, predict_rating, sess, user_test, item_test, user_input_test, item_input_test, user_aux_input_test, user_mask_input_test, item_mask_input_test, user_aux_mask_input_test, rating_input_test, num_nodes, z_real_dist):
    rmses, maes = [], []

    for i in range(int(len(user_input_test) / 100) + 1):
        user_batch_test, item_batch_test, user_input_batch_test, item_input_batch_test, user_aux_input_batch_test, user_mask_batch_test, item_mask_batch_test, user_aux_mask_batch_test, rates_batch_test = get_train_instance_batch_test_aux(i, 100, user_test, item_test, user_input_test,
                                                                                          item_input_test, user_aux_input_test, user_mask_input_test, item_mask_input_test, user_aux_mask_input_test, rating_input_test)
        real_dist = np.random.randn(len(user_batch_test), 32)
        predict = sess.run(predict_rating, feed_dict={users: user_batch_test, items: item_batch_test, users_inputs: user_input_batch_test, items_inputs: item_input_batch_test, users_aux_inputs: user_aux_input_batch_test,
                                                      users_masks_inputs: user_mask_batch_test, items_masks_inputs: item_mask_batch_test, users_aux_masks_inputs: user_aux_mask_batch_test, drop_out_rate:1.0, num_nodes: len(user_batch_test), z_real_dist: real_dist})
        row, col = predict.shape
        for r in range(row):
            rmses.append(pow((predict[r,0] - rates_batch_test[r]), 2))
            maes.append(abs(predict[r,0] - rates_batch_test[r]))
    rmse = np.array(rmses).mean()
    mae = np.array(maes).mean()
    return rmse, mae



if __name__ == "__main__":

    word_latent_dim = 300
    max_doc_length = 300
    cnn_windows_size = 3
    v_dim = 50
    learn_rate = 0.001
    lambda_1 = 0.05#normally we set from [0.05, 0.01, 0.005, 0.001]
    batch_size = 200
    epochs = 500
    gama=0.8

    

    for i in range(1):
        fpath=["./data/musical/"]
        dataset = Dataset(max_doc_length, fpath[i])
        num_factor = 32
        num_filters = 50
        drop_rate = 0.6  
        print("  latent_dim  ",num_factor,"  num_filters   ",num_filters,"   drop_out   ",drop_rate,"   dataset:  ",fpath[i])  
        word_dict, user_reviews, item_reviews, user_aux_reviews, user_masks, item_masks, user_aux_masks, train, valRatings, testRatings = dataset.word_id_dict, dataset.userReview_dict, dataset.itemReview_dict, dataset.userAuxReview_dict, dataset.userMask_dict, dataset.itemMask_dict, dataset.userAuxMask_dict, dataset.trainMtrx, dataset.valRatings, dataset.testRatings
        secTime = time()
        num_user, num_item = train.shape
        word_embedding_mtrx = ini_word_embed(len(word_dict), word_latent_dim)
        user_input, item_input, rateings = get_train_instance(train)
        train_model()

