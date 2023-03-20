import numpy as np

from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pathlib
from dataIO import *
# import tensorflow as tf




def model_to_world(vert):
    # normalize model into a unit sphere at origin
    n = vert.shape[0]
    # print(n)
    dist = cdist(vert, vert)
    # print(dist.shape)
    max_idx = np.argmax(dist)
    max_idx = [max_idx // n, max_idx % n]
    # print(dist[max_idx[0]][max_idx[1]])
    # print(vert[max_idx[0]], vert[max_idx[1]])
    model_center = np.mean([vert[max_idx[0]], vert[max_idx[1]]], axis=0)
    model_scale = 2 / dist[max_idx[0]][max_idx[1]]
    # print(model_center, model_scale)

    
    x, y, z = -model_center[:3]
    M = np.matmul(
            scale(model_scale),
            translate(x, y, z)
    )
    return M


def get_rotation(quaterunion):
    w, x, y, z = quaterunion
    R = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
    ])
    return R


def set_camera_colmap(q, t, c):
    camera = dict()
    r = get_rotation(q)
    # flip y and z
    r = np.array([r[0], -r[1], -r[2]])
    t = np.array([t[0], -t[1], -t[2]])
    w, h, _, f = c
    camera['width'] = w
    camera['height'] = h
    camera['origin'] = np.matmul(t, -r)
    camera['target'] = -r[2] + camera['origin']
    camera['fov'] = 2 * np.arctan(0.5 * (camera['height']**2)**0.5 / f) * 180 / np.pi
    camera['up'] = r[1]

    camera['near'] = 1
    camera['far'] = 100

    return camera

def set_camera_mitsuba(camera, r, t):
    rx, ry, rz = r
    tx, ty, tz = t
    camera_temp = copy.deepcopy(camera)
    camera_rotation = np.matmul(rotate_camera_y(ry / 180 * np.pi), rotate_camera_x(rx / 180 * np.pi))
    camera_temp['origin'] = np.matmul(camera_temp['origin'], np.transpose(camera_rotation))
    camera_temp['up'] = np.matmul(camera_temp['up'], np.transpose(camera_rotation))

    return camera_temp

def get_grid(shape):
    H, W = shape[1:3]
    N = 1
    N_i = tf.range(N)
    H_i = tf.range(H)
    W_i = tf.range(W)
    n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
    n = tf.expand_dims(n, axis=3) # [N, H, W, 1]
    h = tf.expand_dims(h, axis=3) # [N, H, W, 1]
    w = tf.expand_dims(w, axis=3) # [N, H, W, 1]
    n = tf.cast(n, tf.float32) # [N, H, W, 1]
    h = tf.cast(h, tf.float32) # [N, H, W, 1]
    w = tf.cast(w, tf.float32) # [N, H, W, 1]
    return tf.concat([w, h], axis=-1)



def make_adj(n, train_seq, test_seq):
    def nearest(lst, v, n):
        lst = np.abs(np.array(lst) - v)
        # print(lst)
        min_i = np.argsort(lst)

        if lst[min_i[0]] == 0:
            min_i = min_i[1:]

        min_i = min_i[:min(n, lst.shape[0])]
        # print(min_i)
        min_v = train_seq[min_i]
        # print(min_v)
        return min_v

    train_seq = np.array(train_seq)
    test_seq = np.array(test_seq)
    adj_train_seq = np.array([nearest(train_seq, i, n) for i in train_seq])
    adj_test_seq = np.array([nearest(train_seq, i, n) for i in test_seq])

    return adj_train_seq, adj_test_seq


def temporal_view(co, ct, k):
    ct = copy.deepcopy(ct)
    ct['origin'] = 1 / k * ct['origin'] + (k - 1) / k * co['origin']
    ct['target'] = 1 / k * ct['target'] + (k - 1) / k * co['target']
    ct['fov'] = 1 / k * ct['for'] + (k - 1) / k * co['fov']
    ct['up'] = 1 / k * ct['up'] + (k - 1) / k * co['up']
    return ct

def inter_cmr(c0, c1, n):
    ori = np.linspace(c0['origin'], c1['origin'], n + 1)[1:-1]
    tar = np.linspace(c0['target'], c1['target'], n + 1)[1:-1]
    fov = np.linspace(c0['fov'], c1['fov'], n + 1)[1:-1]
    up = np.linspace(c0['up'], c1['up'], n + 1)[1:-1]
    cis = []
    for i in range(n - 1):
        ci = copy.deepcopy(c0)
        ci['origin'] = ori[i]
        ci['target'] = tar[i]
        ci['fov'] = fov[i]
        ci['up'] = up[i]
        cis.append(ci)
    return cis


def inter_view(data, train_seq, n):
    # data: [idx, cmr, ref]


    data_inter = []
    train_seq_inter = []
    render_seq_inter = []
    count = 0
    for i in range(len(train_seq) - 1):
        i0 = train_seq[i]
        i1 = train_seq[i + 1]
        data[i0][0] = count
        data_inter.append(data[i0])
        train_seq_inter.append(count)
        render_seq_inter.append(count)
        count += 1
        cmr = inter_cmr(data[i0][1], data[i1][1], n)
        for j in range(n - 1):
            data_inter.append([count, cmr[j], data[i0][2] * 0])
            render_seq_inter.append(count)
            count += 1
    data[train_seq[-1]][0] = count
    data_inter.append(data[train_seq[-1]])
    train_seq_inter.append(count)
    return data_inter, train_seq_inter, render_seq_inter



def world_to_clip(camera):
    ori = camera['origin']
    tar = camera['target']
    up = camera['up']
    D = normalize(tar - ori)
    U = -normalize(up)
    R = -normalize(np.cross(D, U))
    x, y, z = -ori
    V = np.matmul(
            np.array([[R[0], R[1], R[2], 0], 
                      [U[0], U[1], U[2], 0], 
                      [D[0], D[1], D[2], 0], 
                      [0,    0,    0,    1]]),
            translate(x, y, z)
        )
    r = np.array([[R[0], R[1], R[2]], 
                  [U[0], U[1], U[2]], 
                  [D[0], D[1], D[2]]])
    t = [x, y, z]
    # print(-t)
    # print(camera['origin'])
    # vert_view = np.matmul(vert_world, np.transpose(V))

    near = camera['near']
    far = camera['far']
    fov = camera['fov'] / 180 * np.pi
    asp = camera['width'] / camera['height']
    P = np.array([
            [1/np.tan(fov/2)/asp, 0,               0,                      0                       ], 
            [0,                   1/np.tan(fov/2), 0,                      0                       ], 
            [0,                   0,               (far+near)/(far-near), -(2*far*near)/(far-near)], 
            [0,                   0,               1,                     0                       ]
        ])

    # vert_clip = np.matmul(vert_view, np.transpose(P))

    VP = np.matmul(P, V)
    # vert_clip = np.matmul(vert, np.transpose(MVP))


    return VP






def normalize(v, axis=1):
    import tensorflow as tf
    t = type(v)
    if t == np.ndarray or t == list:
        return v / np.sum(v**2)**0.5
    elif t == tf.Tensor:
        return tf.nn.l2_normalize(v, axis=axis)

def rotate_camera_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s], 
                     [ 0, 1, 0], 
                     [-s, 0, c]]).astype(np.float32)

def rotate_camera_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0], 
                     [0,  c, s], 
                     [0, -s, c]]).astype(np.float32)

def rotate_camera_z(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c,  s, 0], 
                     [-s, c, 0], 
                     [0,  0, 1]]).astype(np.float32)

def rotate_2d(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c,  s], 
                     [-s, c]]).astype(np.float32)



def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0], 
                     [  0, n/-x,            0,              0], 
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                     [  0,    0,           -1,              0]]).astype(np.float32)
                    
def translate(x, y, z):
    return np.array([[1, 0, 0, x], 
                     [0, 1, 0, y], 
                     [0, 0, 1, z], 
                     [0, 0, 0, 1]]).astype(np.float32)



def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0], 
                     [0,  c, s, 0], 
                     [0, -s, c, 0], 
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0], 
                     [ 0, 1, 0, 0], 
                     [-s, 0, c, 0], 
                     [ 0, 0, 0, 1]]).astype(np.float32)

def scale(s):
    return np.array([[ s, 0, 0, 0], 
                     [ 0, s, 0, 0], 
                     [ 0, 0, s, 0], 
                     [ 0, 0, 0, 1]]).astype(np.float32)

def lookAt(eye, at, up):
    a = eye - at
    b = up                      # U
    w = a / np.linalg.norm(a)   # D
    u = np.cross(b, w)          # -R
    u = u / np.linalg.norm(u)   # -R
    v = np.cross(w, u)          # U
    translate = np.array([[1, 0, 0, -eye[0]], 
                          [0, 1, 0, -eye[1]], 
                          [0, 0, 1, -eye[2]], 
                          [0, 0, 0, 1]]).astype(np.float32)
    rotate =  np.array([[u[0], u[1], u[2], 0], 
                        [v[0], v[1], v[2], 0], 
                        [w[0], w[1], w[2], 0], 
                        [0, 0, 0, 1]]).astype(np.float32)
    return np.matmul(rotate, translate)

def preview_points(v):
    plt.scatter(v[:, 0], v[:, 1], v[:, 2])
    # plt.scatter(10, 10)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()

def preview1_tris(v, fv):
    for i in fv:
        tri = np.array([v[j] for j in i])
        tri = np.vstack([tri, v[fv[0]]])
        plt.plot(tri[:, 0], tri[:, 1], tri[:, 2], c='r')
    plt.axis('equal')
    plt.xlim((-1, 1))
    plt.ylim((-2, 0))

    plt.show()

    # points = []
    # for i in fv:
    #     for j in i:
    #         points.append(v[j])
    # points = np.array(points)
    # print(points.shape)
    # plt.scatter(points[:, 0], points[:, 1], points[:, 2])
    # plt.show()

def rotate_camera(camera, x, y):
    camera_temp = copy.deepcopy(camera)
    camera_rotation = np.matmul(rotate_camera_y(y / 180 * np.pi), rotate_camera_x(x / 180 * np.pi))
    view_pos = np.matmul(camera_temp['origin'], np.transpose(camera_rotation))
    camera_temp['origin'] = view_pos
    camera_temp['up'] = np.matmul(camera_temp['up'], np.transpose(camera_rotation))
    return camera_temp



def bilinear_sampler(x, v, normalize=False, name=None):
    """
    Args:
      x - Input tensor [N, H, W, C]
      v - Vector flow tensor [N, H, W, 2], tf.float32

      (optional)
      resize - Whether to resize v as same size as x
      normalize - Whether to normalize v from scale 1 to H (or W).
                  h : [-1, 1] -> [-H/2, H/2]
                  w : [-1, 1] -> [-W/2, W/2]
      crop - Setting the region to sample. 4-d list [h0, h1, w0, w1]
      out  - Handling out of boundary value.
             Zero value is used if out="CONSTANT".
             Boundary values are used if out="EDGE".
    """

    def _get_grid_array(N, H, W, h, w):
        N_i = tf.range(N)
        H_i = tf.range(h+1, h+H+1)
        W_i = tf.range(w+1, w+W+1)
        n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
        n = tf.expand_dims(n, axis=3) # [N, H, W, 1]
        h = tf.expand_dims(h, axis=3) # [N, H, W, 1]
        w = tf.expand_dims(w, axis=3) # [N, H, W, 1]
        n = tf.cast(n, tf.float32) # [N, H, W, 1]
        h = tf.cast(h, tf.float32) # [N, H, W, 1]
        w = tf.cast(w, tf.float32) # [N, H, W, 1]
        return n, h, w

    import tensorflow as tf
    shape = tf.shape(x) # TRY : Dynamic shape
    N = shape[0]

    H_ = H = shape[1]
    W_ = W = shape[2]
    h = w = 0

    out = "CONSTANT"
    if out == "CONSTANT":
        x = tf.pad(x, ((0,0), (1,1), (1,1), (0,0)), mode='CONSTANT')
    elif out == "EDGE":
        x = tf.pad(x, ((0,0), (1,1), (1,1), (0,0)), mode='REFLECT')

    vy, vx = tf.split(v, 2, axis=3)
    if normalize :
        vy *= tf.cast(H / 2, tf.float32)
        vx *= tf.cast(W / 2, tf.float32)

    n, h, w = _get_grid_array(N, H, W, h, w) # [N, H, W, 3]

    vx0 = tf.floor(vx)
    vy0 = tf.floor(vy)
    vx1 = vx0 + 1
    vy1 = vy0 + 1 # [N, H, W, 1]

    H_1 = tf.cast(H_+1, tf.float32)
    W_1 = tf.cast(W_+1, tf.float32)
    iy0 = tf.clip_by_value(vy0 + h, 0., H_1)
    iy1 = tf.clip_by_value(vy1 + h, 0., H_1)
    ix0 = tf.clip_by_value(vx0 + w, 0., W_1)
    ix1 = tf.clip_by_value(vx1 + w, 0., W_1)

    i00 = tf.concat([n, iy0, ix0], 3)
    i01 = tf.concat([n, iy1, ix0], 3)
    i10 = tf.concat([n, iy0, ix1], 3)
    i11 = tf.concat([n, iy1, ix1], 3) # [N, H, W, 3]
    i00 = tf.cast(i00, tf.int32)
    i01 = tf.cast(i01, tf.int32)
    i10 = tf.cast(i10, tf.int32)
    i11 = tf.cast(i11, tf.int32)

    x00 = tf.gather_nd(x, i00)
    x01 = tf.gather_nd(x, i01)
    x10 = tf.gather_nd(x, i10)
    x11 = tf.gather_nd(x, i11)
    w00 = tf.cast((vx1 - vx) * (vy1 - vy), tf.float32)
    w01 = tf.cast((vx1 - vx) * (vy - vy0), tf.float32)
    w10 = tf.cast((vx - vx0) * (vy1 - vy), tf.float32)
    w11 = tf.cast((vx - vx0) * (vy - vy0), tf.float32)
    output = tf.add_n([w00*x00, w01*x01, w10*x10, w11*x11], name=name)

    return output


def bilinear_sampler_(x, v, normalize=False, out="CONSTANT"):
    """
    Args:
      x - Input tensor [N, H, W, C]
      v - Vector flow tensor [N, H, W, 2], tf.float32

      (optional)
      resize - Whether to resize v as same size as x
      normalize - Whether to normalize v from scale 1 to H (or W).
                  h : [-1, 1] -> [-H/2, H/2]
                  w : [-1, 1] -> [-W/2, W/2]
      crop - Setting the region to sample. 4-d list [h0, h1, w0, w1]
      out  - Handling out of boundary value.
             Zero value is used if out="CONSTANT".
             Boundary values are used if out="EDGE".
    """

    def _get_grid_array(N, H, W, h, w, pad):
        N_i = tf.range(N)
        H_i = tf.range(h+pad, h+H+pad)
        W_i = tf.range(w+pad, w+W+pad)
        n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
        n = tf.expand_dims(n, axis=3) # [N, H, W, 1]
        h = tf.expand_dims(h, axis=3) # [N, H, W, 1]
        w = tf.expand_dims(w, axis=3) # [N, H, W, 1]
        n = tf.cast(n, tf.float32) # [N, H, W, 1]
        h = tf.cast(h, tf.float32) # [N, H, W, 1]
        w = tf.cast(w, tf.float32) # [N, H, W, 1]
        return n, h, w


    H_ = H = shape[1]
    W_ = W = shape[2]
    h = w = 0


    pad=1

    if out == "CONSTANT":
        x = tf.pad(x, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='CONSTANT')
    elif out == "EDGE":
        x = tf.pad(x, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='REFLECT')

    vy, vx = tf.split(v, 2, axis=3)
    if normalize :
        vy *= tf.cast(H / 2, tf.float32)
        vx *= tf.cast(W / 2, tf.float32)

    n, h, w = _get_grid_array(N, H, W, h, w, pad) # [N, H, W, 3]

    vx0 = tf.round(vx)
    vy0 = tf.round(vy)
    vx1 = vx0 + 1
    vy1 = vy0 + 1 # [N, H, W, 1]
    vx2 = vx0 - 1
    vy2 = vy0 - 1

    H_0 = tf.cast(pad, tf.float32)
    W_0 = tf.cast(pad, tf.float32)
    H_1 = tf.cast(H_+pad, tf.float32)
    W_1 = tf.cast(W_+pad, tf.float32)
    iy0 = tf.clip_by_value(vy0 + h, H_0, H_1)
    iy1 = tf.clip_by_value(vy1 + h, H_0, H_1)
    iy2 = tf.clip_by_value(vy2 + h, H_0, H_1)
    ix0 = tf.clip_by_value(vx0 + w, W_0, W_1)
    ix1 = tf.clip_by_value(vx1 + w, W_0, W_1)
    ix2 = tf.clip_by_value(vx2 + w, W_0, W_1)

    i00 = tf.concat([n, iy0, ix0], 3)
    i01 = tf.concat([n, iy0, ix1], 3)
    i02 = tf.concat([n, iy0, ix2], 3)
    i10 = tf.concat([n, iy1, ix0], 3)
    i11 = tf.concat([n, iy1, ix1], 3) # [N, H, W, 3]
    i12 = tf.concat([n, iy1, ix2], 3)
    i20 = tf.concat([n, iy2, ix0], 3)
    i21 = tf.concat([n, iy2, ix1], 3)
    i22 = tf.concat([n, iy2, ix2], 3)
    i00 = tf.cast(i00, tf.int32)
    i01 = tf.cast(i01, tf.int32)
    i02 = tf.cast(i02, tf.int32)
    i10 = tf.cast(i10, tf.int32)
    i11 = tf.cast(i11, tf.int32)
    i12 = tf.cast(i12, tf.int32)
    i20 = tf.cast(i20, tf.int32)
    i21 = tf.cast(i21, tf.int32)
    i22 = tf.cast(i22, tf.int32)

    x00 = tf.gather_nd(x, i00)
    x01 = tf.gather_nd(x, i01)
    x02 = tf.gather_nd(x, i02)
    x10 = tf.gather_nd(x, i10)
    x11 = tf.gather_nd(x, i11)
    x12 = tf.gather_nd(x, i12)
    x20 = tf.gather_nd(x, i20)
    x21 = tf.gather_nd(x, i21)
    x22 = tf.gather_nd(x, i22)
    w00 = tf.cast((1 - tf.abs(vy0 - vy)) * (1 - tf.abs(vx0 - vx)), tf.float32)
    w01 = tf.cast((1 - tf.abs(vy0 - vy)) * (1 - tf.abs(vx1 - vx)), tf.float32)
    w02 = tf.cast((1 - tf.abs(vy0 - vy)) * (1 - tf.abs(vx2 - vx)), tf.float32)
    w10 = tf.cast((1 - tf.abs(vy1 - vy)) * (1 - tf.abs(vx0 - vx)), tf.float32)
    w11 = tf.cast((1 - tf.abs(vy1 - vy)) * (1 - tf.abs(vx1 - vx)), tf.float32)
    w12 = tf.cast((1 - tf.abs(vy1 - vy)) * (1 - tf.abs(vx2 - vx)), tf.float32)
    w20 = tf.cast((1 - tf.abs(vy2 - vy)) * (1 - tf.abs(vx0 - vx)), tf.float32)
    w21 = tf.cast((1 - tf.abs(vy2 - vy)) * (1 - tf.abs(vx1 - vx)), tf.float32)
    w22 = tf.cast((1 - tf.abs(vy2 - vy)) * (1 - tf.abs(vx2 - vx)), tf.float32)
    output = tf.add_n([w00*x00, w01*x01, w02*x02, w10*x10, w11*x11, w12*x12, w20*x20, w21*x21, w22*x22])
    output /= tf.add_n([w00, w01, w02, w10, w11, w12, w20, w21, w22])
    # output = tf.add_n([w00*x00, w01*x01, w10*x10, w11*x11])

    return output


def gaussian_sampler(x, v, size, sigma, normalize=False, out="CONSTANT"):

    def _get_grid_array(N, H, W, h, w, pad):
        N_i = tf.range(N)
        H_i = tf.range(h+pad, h+H+pad)
        W_i = tf.range(w+pad, w+W+pad)
        n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
        n = tf.expand_dims(n, axis=3) # [N, H, W, 1]
        h = tf.expand_dims(h, axis=3) # [N, H, W, 1]
        w = tf.expand_dims(w, axis=3) # [N, H, W, 1]
        n = tf.cast(n, tf.float32) # [N, H, W, 1]
        h = tf.cast(h, tf.float32) # [N, H, W, 1]
        w = tf.cast(w, tf.float32) # [N, H, W, 1]
        return n, h, w

    shape = tf.shape(x) # TRY : Dynamic shape
    N = shape[0]
    H = shape[1]
    W = shape[2]
    h = w = 0


    pad = (size - 1) // 2
    if out == "CONSTANT":
        x = tf.pad(x, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='CONSTANT')
    elif out == "EDGE":
        x = tf.pad(x, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='REFLECT')

    vy, vx = tf.split(v, 2, axis=3)
    if normalize :
        vy *= tf.cast(H / 2, tf.float32)
        vx *= tf.cast(W / 2, tf.float32)

    n, h, w = _get_grid_array(N, H, W, h, w, pad) # [N, H, W, 3]


    vy_abs_l = [tf.round(vy) + i for i in range(-pad, pad + 1)]
    vx_abs_l = [tf.round(vx) + i for i in range(-pad, pad + 1)]


    H_0 = tf.cast(pad, tf.float32)
    W_0 = tf.cast(pad, tf.float32)
    H_1 = tf.cast(H+pad, tf.float32)
    W_1 = tf.cast(W+pad, tf.float32)

    vy_l = [tf.clip_by_value(h + i, H_0, H_1) for i in vy_abs_l]
    vx_l = [tf.clip_by_value(w + i, W_0, W_1) for i in vx_abs_l]


    v_l = [tf.cast(tf.concat([n, y_i, x_i], 3), tf.int32) for y_i in vy_l for x_i in vx_l]


    x_l = [tf.gather_nd(x, i) for i in v_l]

    # guassian weights:
    h_l = [i - vy for i in vy_abs_l]
    w_l = [i - vx for i in vx_abs_l]
    W_l = [tf.exp(-(h_i**2 + w_i**2) / (2 * sigma**2)) for h_i in h_l for w_i in w_l]
    # W_l = [(1 - tf.abs(h_i)) * (1 - tf.abs(w_i)) for h_i in h_l for w_i in w_l]
    output = tf.add_n([x_l[i] * W_l[i] for i in range(size**2)])
    output /= tf.add_n(W_l)

    # reginal_max = tf.reduce_max(tf.concat(x_l, axis=3), axis=3, keep_dims=True)

    return output




if __name__ == '__main__':
    pwd = pathlib.Path(__file__).absolute().parent
    obj_file = 'spot.obj'
    obj_data = load_obj(f'{pwd}/../data/{obj_file}')
    # transform to clip space
    camera_config = {
        'origin': np.array([0, 0, 5]),
        'target': np.array([0, 0, 0]),
        'up': np.array([0, 1, 0]),
        'near': 1,
        'far': 200,
        'fov': 20,
        'height': 512,
        'width': 512,
    }

    # camera_config = get_camera_config(get_camera_tensor(camera_config))

    model_pos = np.hstack([obj_data['v'], np.ones((obj_data['v'].shape[0], 1))])

    # camera_config['origin'] = np.dot(rotate_y(0 / 180 * np.pi), np.hstack([camera_config['origin'], 1]).T)[:3]
    # MVP = model_to_clip(model_pos, camera_config)
    MVP = np.matmul(world_to_clip(camera_config), model_to_world(model_pos))
    MVP = world_to_clip(camera_config)
    clip_pos = np.matmul(model_pos, np.transpose(MVP))
    NDC_pos = np.array([i / i[3] for i in clip_pos])
    print(clip_pos)
    # print(NDC_pos)
    # preview_points(NDC_pos)

    # preview1_tris(clip_pos, obj_data['fv'])

    print()

    a_rot = np.matmul(rotate_x(0), rotate_y(0))
    proj  = projection(x=np.tan(20 / 360 * np.pi), n=1.0, f=200.0)
    a_mv  = np.matmul(translate(0, 0, -5), a_rot)
    a_mvp = np.matmul(proj, a_mv).astype(np.float32)

    clip_pos1 = np.matmul(model_pos, np.transpose(a_mvp))
    

    NDC_pos1 = np.array([i / i[3] for i in clip_pos1])


    print(clip_pos1)
    # preview_points(clip_pos1)
    # print(NDC_pos1)
    # preview_points(NDC_pos1)
    
