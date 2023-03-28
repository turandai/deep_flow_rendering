import os
import sys
import pathlib
import math
import numpy as np
import imageio
import tensorflow as tf
import nvdiffrast.tensorflow as dr
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.python import pywrap_tensorflow
# import tensorflow_addons as tfa
import time
import random
from sklearn.cluster import KMeans
import joblib
import gc
import cv2
# import pycuda.autoinit
# import pycuda.driver as cuda
# from pycuda.compiler import SourceModule

from dataIO import *
from transform import *
from transformer import *


def init_vars():
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        # init vars
        # sess.run(tf.global_variables_initializer())
        global_vars = tf.global_variables()
        is_initialized = tf.get_default_session().run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if not f]
        if len(not_initialized_vars):
            tf.get_default_session().run(tf.variables_initializer(not_initialized_vars))
        # return sess


def gaussian_kernel(kernel_size, sigma, n_channels):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, tf.float32), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

def sobel_kernel():
    gx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]]).astype(np.float32)
    gy = np.array([[ 1,  2,  1], 
                   [ 0,  0,  0], 
                   [-1, -2, -1]]).astype(np.float32)
    return gx[:, :, tf.newaxis, tf.newaxis], gy[:, :, tf.newaxis, tf.newaxis]

def laplacian_kernel():
    k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]).astype(np.float32)
    return k[:, :, tf.newaxis, tf.newaxis] 

def avg_filter(img, size):
    k = np.full([size, size], 1/size**2).astype(np.float32)
    # k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.float32) / 9
    k = k[:, :, tf.newaxis, tf.newaxis]
    k = tf.pad(k, [[0, 0], [0, 0], [0, img.shape[-1] - 1], [0, 0]])
    img = tf.nn.depthwise_conv2d(img, k, [1, 1, 1, 1], 'SAME')
    return img


def detect_edge(img):

    img = tf.reduce_mean(img, axis=-1, keep_dims=True)
    # kernel = gaussian_kernel(3, 2, int(img.shape[-1]))
    # img = tf.abs(tf.nn.depthwise_conv2d(img, kernel, [1, 1, 1, 1], 'SAME'))
    gx, gy = sobel_kernel()
    Igx = tf.nn.depthwise_conv2d(img, gx, [1, 1, 1, 1], 'SAME')
    Igy = tf.nn.depthwise_conv2d(img, gy, [1, 1, 1, 1], 'SAME')
    img = tf.abs(Igx + Igy)

    return img

def conv_down(data_in, filters, kernel_size=4, strides=2, activation=tf.nn.relu, regularizer=None, name=None, reuse=None):
    res = tf.layers.conv2d(
        inputs=data_in,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        activation=activation,
        kernel_regularizer=regularizer,
        name = name,
        reuse = reuse
    )
    # print(data_in.shape, '-->', res.shape)
    return res

def conv_up(data_in, filters, kernel_size=4, strides=2, activation=tf.nn.relu, regularizer=None, name=None, reuse=None):
    # kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1)
    res = tf.layers.conv2d_transpose(
        inputs=data_in,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        activation=activation,
        kernel_regularizer=regularizer,
        name = name,
        reuse = reuse
    )
    # print(data_in.shape, '-->', res.shape)
    return res

def max_pool(data_in):
    res = tf.layers.max_pooling2d(
        data_in,
        data_in.shape[-1],
        1,
        padding='valid',
        data_format='channels_last',
        name=None
    )
    return res


def positional_encode(feat, L):
    # NHW*C -> NHW*2L
    def encode(i):
        return tf.concat([tf.sin(feat * 2**i * np.pi), tf.cos(feat * 2**i * np.pi)], axis=-1)
    return tf.concat([encode(i) for i in range(L)], axis=-1)


    





def median(data):
    n = tf.shape(data)[0]
    data = tf.transpose(data, [1, 2, 3, 0])
    data = tf.math.top_k(data, n // 2 + 1).values
    if n % 2 == 0:
        data = data[..., -1:]
    else:
        data = tf.reduce_mean(data[..., -2:], axis=-1, keep_dims=True)
    data = tf.transpose(data, [3, 0, 1, 2])
    return data

def instance_norm(data):
    # m, v = tf.nn.moments(data, [1, 2], keep_dims=True)
    # scale = tf.Variable(tf.constant(1, tf.float32, [data.shape[-1]]))
    # shift = tf.Variable(tf.constant(0, tf.float32, [data.shape[-1]]))
    # data = scale * (data - m) / (v**0.5 + 10**(-8)) + shift
    data = tf.contrib.layers.instance_norm(data)
    return data

def batch_norm(data):
    data = tf.contrib.layers.batch_norm(data)
    return data

def adaptive_instance_norm(data, ref):
    m, v = tf.nn.moments(data, [1, 2], keep_dims=True)
    mr, vr = tf.moments(ref, [1, 2], keep_dims=True)
    data = vr**0.5 * (data - m) / (v**0.5 + 10**(-8)) + mr
    return data

def loss_mse(l, r, m):
    mse = tf.reduce_mean((r - l)**2, axis=[1, 2, 3]) / tf.reduce_mean(m, axis=[1, 2, 3])
    return mse

def loss_l1(l, r, m):
    l1 = tf.reduce_mean(tf.abs(r - l), axis=[1, 2, 3]) / tf.reduce_mean(m, axis=[1, 2, 3])
    return l1

def loss_cos(l, r, m):
    cos = tf.reduce_sum(tf.nn.l2_normalize(l, axis=-1) * tf.nn.l2_normalize(r, axis=-1), axis=-1, keep_dims=True)
    loss = tf.reduce_mean(tf.exp(-cos), axis=[1, 2, 3]) / tf.reduce_mean(m, axis=[1, 2, 3])
    # loss = tf.reduce_mean(-tf.log(cos), axis=[1, 2, 3]) / tf.reduce_mean(m, axis=[1, 2, 3])
    return loss

def loss_kld(l, r, m, sig=True):
    # l = tf.nn.sigmoid(l)
    # r = tf.nn.sigmoid(r)
    l += 1
    r += 1
    l /= tf.reduce_sum(l, axis=[1, 2], keep_dims=True)
    r /= tf.reduce_sum(r, axis=[1, 2], keep_dims=True)
    kld = tf.abs(tf.reduce_sum(r * tf.log(r / l) * m, axis=[1, 2, 3])) / tf.reduce_mean(m, axis=[1, 2, 3])
    return kld

def loss_vgg(l, r, m):
    base_model = VGG19(weights='imagenet', include_top=False)
    # # print(full_vgg_model.summary())
    vgg_model = Model(base_model.input, 
                     [base_model.get_layer('block1_conv2').output,
                      base_model.get_layer('block2_conv2').output,
                      #base_model.get_layer('block3_conv4').output,
                      #base_model.get_layer('block4_conv4').output
                     ])

    l_feat = vgg_model(l, training=False, mask=None)
    r_feat = vgg_model(r, training=False, mask=None)

    loss = [tf.reduce_mean(tf.abs(l_feat[i] - r_feat[i]), axis=[1, 2, 3]) / tf.reduce_mean(m, axis=[1, 2, 3]) for i in range(len(l_feat))]

    # loss = tf.reduce_mean(tf.abs(l_feat - r_feat), axis=[1, 2, 3]) / tf.reduce_mean(m, axis=[1, 2, 3])
    # layers = len(l_feat)

    # mse
    # if rm is None:
    #     loss = tf.reduce_mean((l_feat - r_feat)**2, axis=[1, 2, 3]) / tf.reduce_mean(m, axis=[1, 2, 3])
    #     return loss
    # # cosine similarity
    # cos = tf.reduce_sum(tf.nn.l2_normalize(l_feat, axis=-1) * tf.nn.l2_normalize(r_feat, axis=-1), axis=-1, keep_dims=True)
    # reginal_max = rm
    # reginal_max = tf.reduce_max(cos, keep_dims=True)
    # loss = tf.reduce_mean(tf.exp(-cos / reginal_max), axis=[1, 2, 3]) / tf.reduce_mean(m, axis=[1, 2, 3])


    loss = tf.reduce_sum(loss, axis=0)

    return loss

def IBR_cost(vi, vn, di, dn, gama=0.1, sigma=0.33):
    ca = tf.reduce_sum((vi - vn)**2, axis=-1, keep_dims=True)
    cd = tf.clip_by_value(1 - dn / di, 0, 1)
    c = (1 - gama) * ca + gama * cd
    c = tf.exp(-c / sigma)
    return c


def sh_basis(view, o):
    '''
    view: a tensor shaped (N, H, W, 3)
    output: a tensor shaped (N, H, W, 9)
    '''
    coff_0 = 1 / (2.0*math.sqrt(np.pi))
    coff_1 = math.sqrt(3.0) * coff_0
    coff_2 = math.sqrt(15.0) * coff_0
    coff_3 = math.sqrt(1.25) * coff_0

    # l=0
    sh_bands = [tf.ones_like(view[..., 1]) * coff_0]
    # l=1
    sh_bands.append(view[..., 1] * coff_1)
    sh_bands.append(view[..., 2] * coff_1)
    sh_bands.append(view[..., 0] * coff_1)

    if o == 2:
        sh_bands = tf.stack(sh_bands, axis=-1)
        return sh_bands


    # l=2
    sh_bands.append(view[..., 0] * view[..., 1] * coff_2)
    sh_bands.append(view[..., 1] * view[..., 2] * coff_2)
    sh_bands.append((3.0 * view[..., 2] * view[..., 2] - 1.0) * coff_3)
    sh_bands.append(view[..., 2] * view[..., 0] * coff_2)
    sh_bands.append((view[..., 0] * view[..., 0] - view[..., 2] * view[..., 2]) * coff_2)

    sh_bands = tf.stack(sh_bands, axis=-1)

    return sh_bands



class ReFlow:
    def __init__(self):
        # self.mesh = mesh
        # self.result_path = result_path


        self.width_in = 1024
        self.height_in = 1024
        self.chanel_in = 12 
        self.chanel_out = 3 
        self.seq_len = 4


        self.max_seq = 3
        self.max_buf = 512



    def ras_graph(self, mesh, resolution, tex, w2p_in, view_pos_in, w2p_mid):
        # geometry
        world_pos = mesh['v']
        vert_idx = mesh['fv']
        normal = mesh['vn']
        normal_idx = mesh['fvn']
        uv = mesh['vt']
        uv_idx = mesh['fvt']
        vert_n = len(world_pos)

        # transform
        clip_pos = tf.matmul(world_pos[tf.newaxis], tf.transpose(w2p_in, perm=[0, 2, 1]), name='clip_pos')


        # rasterize
        rast_out, _ = dr.rasterize(clip_pos, vert_idx, resolution, ranges=None, tri_const=True, output_db=False, grad_db=False)
        mask = tf.clip_by_value(rast_out[..., 3:], 0, 1)

        # interpolate
        inter_world_pos, _ = dr.interpolate(world_pos, rast_out, vert_idx, rast_db=None, diff_attrs=None)

        inter_uv, _ = dr.interpolate(uv, rast_out, uv_idx, rast_db=None, diff_attrs=None)
        inter_normal, _ = dr.interpolate(normal, rast_out, normal_idx, rast_db=None, diff_attrs=None)

        inter_view_dir_unit = tf.nn.l2_normalize(inter_world_pos[..., :3] - view_pos_in[:,:,tf.newaxis], axis=-1)
        inter_normal_unit = tf.nn.l2_normalize(inter_normal, axis=-1)
        # reflect_dir = inter_view_dir_unit - 2 * inter_normal_unit * tf.reduce_sum(inter_normal_unit * inter_view_dir_unit, axis=-1, keep_dims=True)
        reflect_dir_unit = tf.nn.l2_normalize(inter_view_dir_unit - 2 * inter_normal_unit * tf.reduce_sum(inter_normal_unit * inter_view_dir_unit, axis=-1, keep_dims=True), axis=-1) * mask
        depth = tf.sqrt(tf.reduce_sum((view_pos_in[:,:,tf.newaxis] - inter_world_pos[..., :3])**2, axis=3, keep_dims=True))

        # texture
        # tex_feat = dr.texture(tex, inter_uv, uv_da, filter_mode='linear-mipmap-linear', boundary_mode='wrap', tex_const=False, max_mip_level=4)
        tex_feat = dr.texture(tex, inter_uv, filter_mode='linear', boundary_mode='wrap', tex_const=False, max_mip_level=None)
        tex_feat *= mask
        # env_map = dr.texture(env[tf.newaxis], reflect_dir_unit, filter_mode='linear', boundary_mode='cube', tex_const=False, max_mip_level=None)
        # env_map *= mask




        # 1-step flow
        iwp = tf.reshape(inter_world_pos[-1:], [1, -1, 4])
        mv = tf.matmul(iwp, tf.transpose(w2p_in, perm=[0, 2, 1]))
        mv = tf.reshape(mv, [-1, resolution[0], resolution[1], 4])
        mv = tf.clip_by_value(mv[..., :-1] / mv[..., -1:], -1, 1) # divide by w
        mv_flow = (mv[:-1] - mv[-1:]) / 2 # normalize
        mv_flow = mv_flow[..., 1::-1] # flip x and y, regard z
        mv_flow = tf.multiply(mv_flow, 1, name='move')

        # 1-step flow mask
        icp, _ = dr.interpolate(clip_pos, rast_out, vert_idx, rast_db=None, diff_attrs=None)
        icp = tf.clip_by_value(icp[..., :-1] / icp[..., -1:], -1, 1)
        n = tf.reshape(tf.range(tf.shape(mv)[0] - 1), [-1, 1, 1, 1])
        n = tf.broadcast_to(n, tf.shape(mv[:-1, ..., :1]))
        mv_xy = tf.cast((mv[:-1, ..., 1::-1] + 1) / 2 * resolution, tf.int32)
        idx = tf.concat([n, mv_xy], axis=-1)
        z_ori = tf.gather_nd(icp[:-1, ..., -1:], idx)
        z_warp = mv[:-1, ..., -1:]
        # tf.add(z_ori, 0, name='z_ori')
        # tf.add(z_warp, 0, name='z_warp')
        fuzzy = 0.01
        vis_mask = tf.less(z_warp, z_ori + fuzzy, name='vis_mask')
        bond_mask = tf.not_equal(tf.reduce_max(tf.abs(mv[:-1]), axis=-1, keep_dims=True), 1.0, name='bond_mask')
        warp_mask = tf.cast(tf.logical_and(vis_mask, bond_mask), tf.float32, name='warp_mask')


        # 2-step flow
        # w2p_mid = tf.concat([w2p_mid, w2p_in[-1:]], axis=-1)
        # iwp = tf.reshape(inter_world_pos[-1:], [1, -1, 4])




        clp2 = tf.matmul(iwp, tf.transpose(w2p_mid, perm=[0, 2, 1]))
        clp2 = tf.reshape(clp2, [-1, resolution[0], resolution[1], 4])
        clp2 = tf.clip_by_value(clp2[..., :-1] / clp2[..., -1:], -1, 1)
        flow2 = (clp2 - mv[-1:]) / 2
        flow2 = flow2[..., 1::-1]


        clip_pos_mid = tf.matmul(world_pos[tf.newaxis], tf.transpose(w2p_mid, perm=[0, 2, 1]), name='clip_pos')
        rast_out_mid,  _ = dr.rasterize(clip_pos_mid, vert_idx, resolution, ranges=None, tri_const=True, output_db=False, grad_db=False) 
        iwp_mid, _ = dr.interpolate(world_pos, rast_out_mid, vert_idx, rast_db=None, diff_attrs=None)

        clp1 = tf.matmul(iwp_mid, tf.transpose(w2p_in[:-1], perm=[0, 2, 1])[:, tf.newaxis])
        clp1 = tf.reshape(clp1, [-1, resolution[0], resolution[1], 4])
        clp1 = tf.clip_by_value(clp1[..., :-1] / clp1[..., -1:], -1, 1)

        clp0 = tf.matmul(iwp_mid, tf.transpose(w2p_mid, perm=[0, 2, 1])[:, tf.newaxis])
        clp0 = tf.reshape(clp0, [-1, resolution[0], resolution[1], 4])
        clp0 = tf.clip_by_value(clp0[..., :-1] / clp0[..., -1:], -1, 1)

        flow1 = (clp1 - clp0) / 2
        flow1 = flow1[..., 1::-1]







        return inter_world_pos, inter_normal_unit, inter_view_dir_unit, reflect_dir_unit, depth, inter_uv, tex_feat, mask, mv_flow, warp_mask, flow2, flow1






    
    def flow_net(self, feat, reuse=None):

        # reg = tf.contrib.layers.l2_regularizer(scale=0.01)
        reg = None



        feat = conv_down(feat, 32, 4, 2, regularizer=reg, name='fg0', reuse=reuse)
        feat = conv_down(feat, 64, 4, 2, regularizer=reg, name='fg1', reuse=reuse)
        feat = conv_down(feat, 128, 4, 2, regularizer=reg, name='fg2', reuse=reuse)
        # feat = conv_down(feat + view, 256, 4, 2, regularizer=reg)
        feat = conv_down(feat, 256, 4, 2, regularizer=reg, name='fg3', reuse=reuse)
        feat = conv_down(feat, 512, 4, 2, regularizer=reg, name='fg4', reuse=reuse)
        feat = conv_down(feat, 512, 4, 1, regularizer=reg, name='fg5', reuse=reuse)

        feat = conv_up(feat, 256, regularizer=reg, name='fg6', reuse=reuse)
        feat = conv_up(feat, 128, regularizer=reg, name='fg7', reuse=reuse)
        feat = conv_up(feat, 64, regularizer=reg, name='fg8', reuse=reuse)
        feat = conv_up(feat, 32, regularizer=reg, name='fg9', reuse=reuse)
        feat = conv_up(feat, 4, activation=None, name='fg10', reuse=reuse)

        return feat






    def fuse_net(self, feat, reuse=None):


        feat_len = int(feat.shape[-1])
        layers = [feat]
        # reg = tf.contrib.layers.l2_regularizer(scale=0.1)
        reg = None

        layers.append(instance_norm(conv_down(layers[-1], 64, regularizer=reg, name='rg0', reuse=reuse)))
        layers.append(instance_norm(conv_down(layers[-1], 128, regularizer=reg, name='rg1', reuse=reuse)))
        layers.append(instance_norm(conv_down(layers[-1], 256, regularizer=reg, name='rg2', reuse=reuse)))
        layers.append(instance_norm(conv_down(layers[-1], 512, regularizer=reg, name='rg3', reuse=reuse)))

        layers.append(instance_norm(conv_up(layers[-1], 256, regularizer=reg, name='rg4', reuse=reuse)))
        layers.append(instance_norm(conv_up(layers[-1] + layers[3], 128, regularizer=reg, name='rg5', reuse=reuse)))
        layers.append(instance_norm(conv_up(layers[-1] + layers[2], 64, regularizer=reg, name='rg6', reuse=reuse)))
        layers.append(instance_norm(conv_up(layers[-1] + layers[1], feat_len, regularizer=reg, name='rg7', reuse=reuse)))
        # layers.append(instance_norm(conv_up(layers[-1] + layers[0], feat_len, 4, 1, regularizer=reg, name='rg8', reuse=reuse)))
        layers.append(conv_up(layers[-1] + layers[0], 3, 4, 1, activation=tf.nn.sigmoid, regularizer=None, name='rg9', reuse=reuse))

        return layers[-1]




    def fuse_graph(self, tex_t, dif_t_fin, pos_t, mask_t, reuse):

        pos_pe = positional_encode(pos_t, 10)
        pos_pe = tf.multiply(pos_pe, 1, name='pos_pe')
        fuseNet_in = tf.concat([tex_t, dif_t_fin, pos_pe], axis=-1)
        fuseNet_out = tf.multiply(self.fuse_net(fuseNet_in, reuse=reuse), mask_t, name='lrn_t')

        return fuseNet_out

    def blend_test(self, ref_w, dif_w, tex_t, feat_w, feat_t, mask_w, mask_t):
        bias = (tf.zeros_like(mask_w) - 10**8) * (1 - mask_w)

        # out = feat_w[..., :3] * feat_t[..., :3]
        # brand = tf.reduce_min(out, axis=0, keep_dims=True) * 0.33
        # weight = tf.exp(-out / (brand + 0.1))
        # weight /= tf.reduce_sum(weight, axis=0, keep_dims=True)
        # weight = tf.multiply(weight, 1, name='weight')

        weight = tf.nn.softmax(tf.reduce_sum(feat_w[..., :3] * feat_t[..., :3] + bias, axis=-1, keep_dims=True), axis=0)
        ref_b = tf.multiply(tf.reduce_sum(ref_w * weight, axis=0, keep_dims=True), mask_t, name='ref_t_bld')
        dif_b = tf.multiply(tf.reduce_sum(dif_w * weight, axis=0, keep_dims=True), mask_t, name='dif_t_bld')
        mask_w_any = tf.clip_by_value(tf.reduce_sum(mask_w, axis=0, keep_dims=True), 0, 1)
        ref_bt = tf.multiply(ref_b + (1 - mask_w_any) * tex_t, mask_t, name='ref_t_bld_tex')
        return ref_b, ref_bt, dif_b



    def regist_graph(self):
        clip_pos = np.zeros([1, 1, 4])
        vert_idx = np.zeros([1, 3])
        resolution = [1024, 1024]
        rast_out,  rast_out_db = dr.rasterize(clip_pos, vert_idx, resolution, ranges=None, tri_const=True, output_db=True, grad_db=True)
        uv = np.zeros([1, 3])
        inter_out, _ = dr.interpolate(uv, rast_out, vert_idx, rast_db=rast_out_db, diff_attrs='all')
        tex = np.zeros([1, 1024, 1024, 3])
        tex_out = dr.texture(tex, inter_out, filter_mode='linear', boundary_mode='wrap', tex_const=False, max_mip_level=None)





    def temporal_graph(self, mesh, resolution):

        h, w = resolution
        crop_hw = tf.placeholder(tf.int32, [2], name='crop_temp')
        y0 = tf.random.uniform([], 0, resolution[0] - crop_hw[0] + 1, tf.int32)
        x0 = tf.random.uniform([], 0, resolution[1] - crop_hw[1] + 1, tf.int32)
        y1 = y0 + crop_hw[0]
        x1 = x0 + crop_hw[1]

        h_p = ((crop_hw[0] - 1) // 2**5 + 1) * 2**5
        w_p = ((crop_hw[1] - 1) // 2**5 + 1) * 2**5
        pad = [[0, 0], [0, h_p - crop_hw[0]], [0, w_p - crop_hw[1]], [0, 0]]

        tex = tf.get_default_graph().get_tensor_by_name('tex:0')

        view_pos_temp = tf.placeholder(tf.float32, [None, 1, 3], name='view_pos_temp')
        w2p_temp = tf.placeholder(tf.float32, [None, 4, 4], name='w2p_temp')
        w2p_mid0 = tf.placeholder(tf.float32, [None, 4, 4], name='w2p_mid0')
        w2p_mid1 = tf.placeholder(tf.float32, [None, 4, 4], name='w2p_mid1')
        ref_temp = tf.placeholder(tf.float32, [None, resolution[0], resolution[1], 3], name='ref_temp')


        w2p_0 = w2p_temp[:-1]
        view_pos_0 = view_pos_temp[:-1]
        ras = self.ras_graph(mesh, resolution, tex, w2p_0, view_pos_0, w2p_mid0)
        p, n, v, r, d, uv, t, m, mv, wm, f2, f1 = [tf.pad(i[:, y0:y1, x0:x1], pad) for i in ras]
        feat = tf.concat([r, d], axis=-1)
        ref_in = tf.pad(ref_temp[:, y0:y1, x0:x1], pad)
        ref_color = tf.multiply(ref_in, m[:-1])
        tex_color = tf.multiply(t[-1:, ..., :3], 1)

        _, ref_w, dif_w, _, _, dif_t_fin = self.flow_graph(feat[:-1], feat[-1:], ref_color, tex_color[-1:], mv * resolution, f2 * resolution, f1 * resolution, wm, m[-1:], p[-1:, ..., :3], reuse=True)
        lrn_color_0 = self.fuse_graph(tex_color[-1:], dif_t_fin, p[-1:, ..., :3], m[-1:], reuse=True)
#         lrn_color_0 = tf.add(tex_color[-1:], dif_t_fin)
        ref_b, ref_bt, dif_b = self.blend_test(ref_w, dif_w, tex_color[-1:], feat[:-1], feat[-1:], wm, m[-1:])

        w2p_1 = tf.concat([w2p_temp[:-2], w2p_temp[-1:]], axis=0)
        view_pos_1 = tf.concat([view_pos_temp[:-2], view_pos_temp[-1:]], axis=0)
        ras = self.ras_graph(mesh, resolution, tex, w2p_1, view_pos_1, w2p_mid1)
        p, n, v, r, d, uv, t, m, mv, wm, f2, f1 = [tf.pad(i[:, y0:y1, x0:x1], pad) for i in ras]
        feat = tf.concat([r, d], axis=-1)
        ref_color = tf.multiply(ref_in, m[:-1])
        tex_color = tf.multiply(t[-1:, ..., :3], 1)

        _, ref_w, dif_w, _, _, dif_t_fin = self.flow_graph(feat[:-1], feat[-1:], ref_color, tex_color[-1:], mv * resolution, f2 * resolution, f1*resolution, wm, m[-1:], p[-1:, ..., :3], reuse=True)
        lrn_color_1 = self.fuse_graph(tex_color[-1:], dif_t_fin, p[-1:, ..., :3], m[-1:], reuse=True)
#         lrn_color_1 = tf.add(tex_color[-1:], dif_t_fin)
        ref_b, ref_bt, dif_b = self.blend_test(ref_w, dif_w, tex_color[-1:], feat[:-1], feat[-1:], wm, m[-1:])

        loss_coh = tf.reduce_mean(tf.abs(lrn_color_0 - lrn_color_1))
        loss_tex = tf.reduce_mean(tf.abs(tex_color[-1:] - ref_bt))
        loss_opt = tf.add(loss_coh, loss_tex, name='loss_temp')

        lr = tf.placeholder(tf.float32, [], name='lr_temp')
        opt = tf.train.AdamOptimizer(lr, 0.9, 0.99, 10**(-6), name='opt_temp').minimize(loss_opt)


    def flow_graph(self, feat_r, feat_t, ref_r, tex_t, flow_rt, flow_mt, flow_rm, mask_w, mask_t, pos_t, reuse):
        feat_w = bilinear_sampler(feat_r, flow_rt) * mask_w #* mask_t
        feat_dif = feat_w - feat_t
        ref_w = tf.multiply(bilinear_sampler(ref_r, flow_rt, name='ref_w_unmask'), mask_w, name='ref_w') #* mask_t
        dif_w = tf.multiply(ref_w - tex_t, mask_w, name='dif_w') #* mask_t
        pos_brd = tf.broadcast_to(pos_t, tf.concat([tf.shape(flow_rt)[:-1], pos_t.shape[-1:]], axis=-1)) * mask_w

        flowNet_in = tf.concat([dif_w, feat_dif, pos_brd], axis=-1)
        flowNet_out = self.flow_net(flowNet_in, reuse=reuse) * mask_w * mask_t

        flow_len = (0.05 + 0.9 * tf.nn.sigmoid(flowNet_out[..., :2]))
        flow_p = tf.multiply(flow_len, (-flow_mt), name='flow_p')
        conf = tf.nn.relu(flowNet_out[..., 2:3], name='conf')
        bias = (tf.zeros_like(mask_w) - 10**8) * (1 - mask_w)
        weight = tf.nn.softmax(flowNet_out[..., -1:] + bias, axis=0, name='weight')

        dif_t = tf.multiply(bilinear_sampler(dif_w, flow_p) * conf, mask_t, name='dif_t')
        dif_t_fin = tf.multiply(tf.reduce_sum(dif_t * weight, axis=0, keep_dims=True), mask_t, name='dif_t_fin')


        # adaptive brandwith
        # out = tf.nn.relu(flowNet_out[..., -1:])
        # brand = tf.reduce_min(out, axis=0, keep_dims=True) * 0.33
        # weight = tf.exp(-out / (brand + 0.1))
        # weight /= tf.reduce_sum(weight, axis=0, keep_dims=True)
        # weight = tf.multiply(weight, 1, name='weight')

        return feat_w, ref_w, dif_w, flow_p, dif_t, dif_t_fin



    def construct_graph(self, mesh, resolution):

        h, w = resolution
        crop_hw = tf.placeholder(tf.int32, [2], name='crop_hw')
        y0 = tf.random.uniform([], 0, resolution[0] - crop_hw[0] + 1, tf.int32)
        x0 = tf.random.uniform([], 0, resolution[1] - crop_hw[1] + 1, tf.int32)
        y1 = y0 + crop_hw[0]
        x1 = x0 + crop_hw[1]

        h_p = ((crop_hw[0] - 1) // 2**5 + 1) * 2**5
        w_p = ((crop_hw[1] - 1) // 2**5 + 1) * 2**5
        pad = [[0, 0], [0, h_p - crop_hw[0]], [0, w_p - crop_hw[1]], [0, 0]]


        tex = tf.Variable(tf.constant(0, tf.float32, [1, 1024, 1024, 3]), name='tex')


        view_pos_in = tf.placeholder(tf.float32, [None, 1, 3], name='view_pos_in')
        w2p_in = tf.placeholder(tf.float32, [None, 4, 4], name='w2p_in')

        w2p_mid = tf.placeholder(tf.float32, [None, 4, 4], name='w2p_mid')

        ras = self.ras_graph(mesh, resolution, tex, w2p_in, view_pos_in, w2p_mid)
        p, n, v, r, d, uv, t, m, mv, mask_w, f2, f1 = [tf.pad(i[:, y0:y1, x0:x1], pad) for i in ras]
        r = tf.multiply(r, 1, name='r')
        p = tf.multiply(tf.nn.l2_normalize(p[..., :3], axis=3), 1, name='p')
        d = tf.multiply(d, 1, name='d')

        d_t = tf.multiply(d[-1:], 1, name='d_t')
        m_t = tf.multiply(m[-1:], 1, name='m_t')

        # visulization
        d_vis = tf.multiply(d_t, m_t)
        d_min = tf.reduce_min(d_vis[d_vis>0])
        d_max = tf.reduce_max(d_vis)
        d_vis = tf.multiply((d_vis - d_min) / (d_max - d_min), m_t, name='d_vis')
        p_vis = tf.multiply(p[-1:], m_t, name='p_vis')
        n_vis = tf.multiply((n[-1:]+1)/2, m_t, name='n_vis')
        n_vis = tf.multiply((v[-1:]+1)/2, m_t, name='v_vis')
        r_vis = tf.multiply((r[-1:]+1)/2, m_t, name='r_vis')
        uv_vis = tf.pad(uv[-1:], [[0,0],[0,0],[0,0],[0,1]], 'constant',constant_values=0.5)
        uv_vis = tf.multiply(uv_vis, m_t, name='uv_vis')
        t_vis = tf.multiply(t[-1:], m_t, name='t_vis')


        mask_r = m[:-1]
        mask_t = m[-1:]
        pos_t = p[-1:, ..., :3]

        feat = tf.concat([r, d], axis=-1)
        feat_r = feat[:-1]
        feat_t = feat[-1:]

        ref_in = tf.placeholder(tf.float32, [None, resolution[0], resolution[1], 3], name='ref_in')
        ref_in = tf.pad(ref_in[:, y0:y1, x0:x1], pad)
        ref_r = tf.multiply(ref_in[:-1], mask_r, name='ref_r')
        ref_t = tf.multiply(ref_in[-1:], mask_t, name='ref_t')

        flow_rt = mv * resolution
        flow_mt = f2 * resolution
        flow_rm = f1 * resolution

        tex_t = tf.multiply(t[-1:, ..., :3], 1, name='tex_t')


        feat_w, ref_w, dif_w, flow_p, dif_t, dif_t_fin = self.flow_graph(feat_r, feat_t, ref_r, tex_t, flow_rt, flow_mt, flow_rm, mask_w, mask_t, pos_t, reuse=None)

        lrn_t = self.fuse_graph(tex_t, dif_t_fin, pos_t, mask_t, reuse=None)
#         lrn_t = tf.add(tex_t, dif_t_fin, name='lrn_t')





        # optimize
        lrn_dif = tf.multiply(ref_t - lrn_t, mask_t, name='lrn_dif')

        mse_pix = loss_mse(lrn_t, ref_t, mask_t)
        l1_pix = tf.reduce_mean(loss_l1(lrn_t, ref_t, mask_t))
        vgg_pix = loss_vgg(lrn_t, ref_t, mask_t)
        loss_pix = 1 * tf.reduce_mean(l1_pix) + tf.reduce_mean(vgg_pix)


        l1_tex = loss_l1(tex_t, ref_t, m[-1:])
        loss_tex = tf.reduce_mean(l1_tex)


        dif_t_ref = tf.multiply(ref_t - tex_t, mask_w, name='dif_ref')
        loss_dif_t = tf.reduce_mean(loss_l1(dif_t_ref, dif_t, m[-1:]))
        loss_dif_t_fin = tf.reduce_mean(loss_l1(dif_t_ref, dif_t_fin, m[-1:]))

        # flow_mean = tf.reduce_mean(flow_p, axis=[1, 2], keep_dims=True)
        # reg_flow = loss_l1(flow_p, flow_mean, mask_t)
        # reg_flow = tf.reduce_mean(tf.exp(-flow_p))



        # guide = -f2 * resolution
        # reg_dif = tf.reduce_mean(loss_cos(flow, guide, m[-1:]))
        # reg_dif += tf.reduce_mean(loss_cos(flow - guide, -guide, m[-1:]))


        loss_dif = 1 * loss_dif_t + loss_dif_t_fin

        l1_warp = tf.reduce_mean(loss_l1(ref_w, ref_t, mask_t))

        # loss_reg = tf.losses.get_regularization_loss()

        self.test = [l1_pix, loss_tex, loss_dif, l1_warp]

        loss_opt = 1 * loss_pix + 1 * loss_dif + 10 * (loss_tex) 


        loss = tf.multiply(mse_pix[0], 255, name='loss')
        lr = tf.placeholder(tf.float32, [], name='lr_in')
        opt = tf.train.AdamOptimizer(lr, 0.9, 0.99, 10**(-6), name='opt').minimize(loss_opt)


        # # naive blend test
        ref_b, ref_bt, dif_b = self.blend_test(ref_w, dif_w, tex_t, feat_w, feat_t, mask_w, mask_t)
        
        # ref_wf = bilinear_sampler(ref_in[:-1], flow_rt)
        # bias = (tf.zeros_like(mask_w) - 10**8) * (1 - mask_w)
        # weight = tf.nn.softmax(tf.reduce_sum(feat_w[..., :3] * feat_t[..., :3] + bias, axis=-1, keep_dims=True), axis=0)
        # ref_bf = tf.reduce_sum(ref_wf * weight, axis=0, keep_dims=True, name='ref_bf')
        # lrn_b = tf.add(lrn_t, ref_bf * (1 - mask_t), name='lrn_b')

        # # 2-step warp test
        # warp1 = bilinear_sampler(ref_r, flow_rm, name='warp1')
        # warp2 = bilinear_sampler(warp1, flow_mt, name='warp2')

        # # visualiztion flow
        flow_dir = (flow_p + 0.01 * resolution) / (2 * 0.01 * resolution) * m[-1:]
        flow_dir = tf.concat([flow_dir, flow_dir[..., :1]], axis=-1, name='flow_dir')


        
        # ang = tf.ones_like(flow_rt[..., :1]) * (np.pi / 12)
        # rot = tf.concat([tf.cos(ang), tf.sin(ang), -tf.sin(ang), tf.cos(ang)], axis=-1)
        # rot = tf.reshape(rot, tf.concat([tf.shape(rot)[:-1], [2, 2]], axis=-1))
        # flow_rot = tf.reduce_sum(tf.matmul(rot, flow_rt[..., tf.newaxis]), axis=-1)
        # warp_tes = bilinear_sampler(ref_r, flow_rot, name='warp_test')







        
    def train(self, mesh, data, train_seq, test_seq, lr, iters, result_path, wr_img=True, wr_ckp=True, ow_reso=None):
        if ow_reso is None:
            resolution = np.array([data[0][1]['height'], data[0][1]['width']])
        else:
            resolution = np.array(ow_reso).astype(np.int32)
        self.construct_graph(mesh, resolution)
        # print('////////////////////////////////////////////////////////////////////', len(tf.trainable_variables()))
        self.temporal_graph(mesh, resolution)
        # print('////////////////////////////////////////////////////////////////////', len(tf.trainable_variables()))
        saver = tf.train.Saver(max_to_keep=None)
        init_vars()
        # train_views, train_idx, test_views, test_idx, train_group, train_group_idx, test_group, test_group_idx, al = arrange_data(data_arrange)
        loss_temp = []
        loss_log = []
        test_log = []
        # ref_buffer = dict()

        train_adj, test_adj = make_adj(4, train_seq, test_seq)

        train_grp = np.concatenate([train_adj, np.reshape(train_seq, [-1, 1])], axis=-1)
        test_grp = np.concatenate([test_adj, np.reshape(test_seq, [-1, 1])], axis=-1)

        for i in range(iters + 1):
            sample_idx = random.randint(0, len(train_seq) - 1)
            sample_seq = copy.deepcopy(train_grp[sample_idx])

            if np.random.rand() < 0.1:
                sample_seq[-2] = sample_seq[-1]

            # print(sample_seq)
            idx_seq = [data[s][0] for s in sample_seq]
            cmr_seq = [data[s][1] for s in sample_seq]
            ref_seq = [data[s][2] for s in sample_seq]



            # ref_seq = [np.zeros(i.shape) for i in ref_seq]


            # camera_seq = [camera]
            view_pos_seq = [c['origin'][np.newaxis] for c in cmr_seq]
            w2p_seq = [world_to_clip(c) for c in cmr_seq]
            # ref_seq = [ref]


            # mid stage
            cmr_mid = [copy.deepcopy(cmr_seq[-1]) for _ in range(len(cmr_seq[:-1]))]
            for c in range(len(cmr_seq[:-1])):
                move = cmr_seq[c]['origin'] - cmr_mid[c]['origin']
                cmr_mid[c]['origin'] += move
                cmr_mid[c]['target'] += move
            view_pos_mid = [c['origin'][np.newaxis] for c in cmr_mid]
            w2p_mid = [world_to_clip(c) for c in cmr_mid]

            # temp correct
            # cmr_temp = temporal_view(cmr_seq[-1], cmr_seq[random.randint(0, 1)])
            # view_pos_temp = [c['origin'][np.newaxis]]
            # w2p_temp = [world_to_clip(c)]

            # train
            feed = {
                'w2p_in:0': w2p_seq,
                'w2p_mid:0': w2p_mid,
                # 'w2p_temp:0': w2p_temp,
                'view_pos_in:0': view_pos_seq,
                # 'view_pos_temp:0': view_pos_temp,
                # 'tex_in:0': tex,
                'ref_in:0': ref_seq,
                'crop_hw:0': [256, 256],
                'lr_in:0': lr,
            }
            _, loss = tf.get_default_session().run(['opt', 'loss:0'], feed)
            loss_temp.append(loss)




            


            if i > 0 and i % 100 == 0:  # print
                loss_log.append([i, np.mean(loss_temp)])
                print(loss_log[-1], sample_seq.tolist())
                loss_temp = []


            if i % 10000 == 0 and wr_img:  # train
                feed = {
                    'w2p_in:0': w2p_seq,
                    'w2p_mid:0': w2p_mid,
                    'view_pos_in:0': view_pos_seq,
                    # 'tex_in:0': tex,
                    'ref_in:0': ref_seq,
                    'crop_hw:0': resolution,
                }
                run_list = ['loss:0', 'lrn_t:0', 'tex_t:0', 'ref_t:0']
                test_loss, lrn, tex, ref = tf.get_default_session().run(run_list, feed)


                write_image(f'{result_path}/{i}_train_lrn.png', lrn[-1])
                write_image(f'{result_path}/{i}_train_tex.png', tex[-1])
                write_image(f'{result_path}/{i}_train_ref.png', ref[-1])


                # test
                sample_idx = random.randint(0, len(test_seq) - 1)
                sample_seq = test_grp[sample_idx]

                idx_seq = [data[s][0] for s in sample_seq]
                cmr_seq = [data[s][1] for s in sample_seq]
                ref_seq = [data[s][2] for s in sample_seq]


                cmr_mid = [copy.deepcopy(cmr_seq[-1]) for _ in range(len(cmr_seq[:-1]))]
                for c in range(len(cmr_seq[:-1])):
                    move = cmr_seq[c]['origin'] - cmr_mid[c]['origin']
                    cmr_mid[c]['origin'] += move
                    cmr_mid[c]['target'] += move
                view_pos_mid = [c['origin'][np.newaxis] for c in cmr_mid]
                w2p_mid = [world_to_clip(c) for c in cmr_mid]

                view_pos_seq = [c['origin'][np.newaxis] for c in cmr_seq]
                w2p_seq = [world_to_clip(c) for c in cmr_seq]
                
                feed = {
                    'w2p_in:0': w2p_seq,
                    'w2p_mid:0': w2p_mid,
                    'view_pos_in:0': view_pos_seq,
                    # 'view_pos_mid:0': view_pos_mid,
                    # 'tex_in:0': tex,
                    'ref_in:0': ref_seq,
                    'crop_hw:0': resolution,
                }
                run_list = ['loss:0', 'lrn_t:0', 'tex_t:0', 'ref_t:0']
                test_loss, lrn, tex, ref = tf.get_default_session().run(run_list, feed)

                write_image(f'{result_path}/{i}_test_lrn.png', lrn[-1])
                write_image(f'{result_path}/{i}_test_tex.png', tex[-1])
                write_image(f'{result_path}/{i}_test_ref.png', ref[-1])


                print("test:", test_loss, sample_seq.tolist())
                print(tst)
                test_log.append([i, test_loss])
                # exit()
            

            # # temp
            for _ in range(1):
                n = 60
                r = random.randint(0, n - 3)
                [c0, c1] = inter_cmr(cmr_seq[0], cmr_seq[1], n)[r:r + 2]

                view_pos_temp = view_pos_seq[:-1] + [c0['origin'][np.newaxis], c0['origin'][np.newaxis]]
                w2p_temp = w2p_seq[:-1] + [world_to_clip(c0), world_to_clip(c1)]
                ref_temp = ref_seq[:-1]

                cmr_mid0 = [copy.deepcopy(c0) for _ in range(len(cmr_seq[:-1]))]
                for c in range(len(cmr_seq[:-1])):
                    move = cmr_seq[c]['origin'] - cmr_mid[c]['origin']
                    cmr_mid0[c]['origin'] += move
                    cmr_mid0[c]['target'] += move
                view_pos_mid0 = [c['origin'][np.newaxis] for c in cmr_mid0]
                w2p_mid0 = [world_to_clip(c) for c in cmr_mid0]

                cmr_mid1 = [copy.deepcopy(c1) for _ in range(len(cmr_seq[:-1]))]
                for c in range(len(cmr_seq[:-1])):
                    move = cmr_seq[c]['origin'] - cmr_mid[c]['origin']
                    cmr_mid1[c]['origin'] += move
                    cmr_mid1[c]['target'] += move
                view_pos_mid1 = [c['origin'][np.newaxis] for c in cmr_mid1]
                w2p_mid1 = [world_to_clip(c) for c in cmr_mid1]

                feed = {
                    'w2p_temp:0': w2p_temp,
                    'view_pos_temp:0': view_pos_temp,
                    'lr_temp:0': lr * 1,
                    'ref_temp:0': ref_temp,
                    'crop_temp:0': [256, 256],
                    'w2p_mid0:0': w2p_mid0,
                    'w2p_mid1:0': w2p_mid1,
                }
                _ = tf.get_default_session().run(['opt_temp'], feed)

                # time.sleep(60 * 50)

            if i % 10000 == 0 and wr_ckp:
                if i == 0:
                    saver.save(tf.get_default_session(), f'{result_path}/save/model')
                else:
                    saver.save(tf.get_default_session(), f'{result_path}/save/model', global_step=i, write_meta_graph=False)

                    


        write_loss(f'{result_path}/save/', loss_log, test_log)



    def render(self, mesh, data, ckp_path, ckp_idx, train_seq, render_seq, result_path, inter, wr_img, adj_n, ow_reso=None):
        if ow_reso is None:
            resolution = np.array([data[0][1]['height'], data[0][1]['width']])
        else:
            resolution = np.array(ow_reso).astype(np.int32)
        self.regist_graph()
        saver = tf.train.import_meta_graph(f'{ckp_path}/model.meta')
        saver.restore(tf.get_default_session(), f'{ckp_path}/model-{ckp_idx}')

        _, render_adj = make_adj(adj_n, train_seq, render_seq)
        render_grp = np.concatenate([render_adj, np.reshape(render_seq, [-1, 1])], axis=-1)

        time_list = []
        loss_list = []
        os.makedirs(result_path/'lrn', exist_ok=True)
        os.makedirs(result_path/'tex', exist_ok=True)
        os.makedirs(result_path/'ref', exist_ok=True)

        dmin = []
        dmax = []

        for i in render_seq:
            print(i)
            sample_idx = i
            sample_seq = copy.deepcopy(render_grp[sample_idx])
            if sample_seq[-1] in train_seq:
                sample_seq[-2] = sample_seq[-1]
            idx_seq = [data[s][0] for s in sample_seq]
            cmr_seq = [data[s][1] for s in sample_seq]
            ref_seq = [data[s][2] for s in sample_seq]
            # print(idx_seq, [np.mean(i) for i in ref_seq])
            view_pos_seq = [c['origin'][np.newaxis] for c in cmr_seq]
            w2p_seq = [world_to_clip(c) for c in cmr_seq]

            cmr_mid = [copy.deepcopy(cmr_seq[-1]) for _ in range(len(cmr_seq[:-1]))]
            for c in range(len(cmr_seq[:-1])):
                move = cmr_seq[c]['origin'] - cmr_mid[c]['origin']
                cmr_mid[c]['origin'] += move
                cmr_mid[c]['target'] += move
            view_pos_mid = [c['origin'][np.newaxis] for c in cmr_mid]
            w2p_mid = [world_to_clip(c) for c in cmr_mid]

            t = time.time()

            feed = {
                'w2p_in:0': w2p_seq,
                'w2p_mid:0': w2p_mid,
                'view_pos_in:0': view_pos_seq,
                # 'view_pos_mid:0': view_pos_mid,
                'ref_in:0': ref_seq,
                'crop_hw:0': resolution,
            }
            loss, lrn, ref = tf.get_default_session().run(['loss:0', 'lrn_t:0', 'ref_t:0'], feed)
            

            time_list.append(time.time() - t)
            loss_list.append(loss)

            if wr_img:
                write_image(f'{result_path}/lrn/{i}_lrn.png', lrn[-1])
                # write_image(f'{result_path}/lrn/{i}_tex.png', m_t[-1])
                # write_image(f'{result_path}/lrn/{i}_dif.png', dif[-1])
                # write_image(f'{result_path}/lrn/{i}_depth.png', d_t[-1])
                if not inter:
                    write_image(f'{result_path}/ref/{i}_ref.png', ref[-1])
        # print(np.max(dmax), np.min(dmin))
        print(f'ckp: {ckp_idx}\nspd: {np.mean(time_list)}')
        if not inter:
            print(f'loss: {np.mean(loss_list)}')



    def debug(self, camera, ref, x, y, result_path):
        init_vars()
        camera = rotate_camera(camera, x, y)
        w2p = world_to_clip(camera)
        view_pos = camera['origin'][np.newaxis]
        feed = {
            self.w2p: [w2p],
            self.view_pos: [view_pos],
            self.ref: [ref]
        }
        dif = tf.get_default_session().run(self.dif_ref, feed)
        write_image(f'{result_path}/{x}_{y}_absdif.png', np.abs(dif[0]))

