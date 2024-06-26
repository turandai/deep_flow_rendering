import os
import sys
import pathlib
import numpy as np
import imageio
import tensorflow as tf
import nvdiffrast.tensorflow as dr
import time
import sys
import random
import argparse

from dataIO import *
from transform import *
from utils import *




def prepare_data_colmap():
    # read mesh
    mesh = read_obj(mesh_path)
    # read cameras and images
    data = read_data_colmap(img_path, cmrlist_path, imglist_path)

    # m2w
    # mesh['v'] = np.hstack([mesh['v'], np.ones((mesh['v'].shape[0], 1))]).astype(np.float32)
    mesh['v'] = np.pad(mesh['v'], [[0, 0], [0, 1]], 'constant', constant_values=1).astype(np.float32)
    mesh['v'] = np.matmul(mesh['v'], m2w.T)
    mesh['vn'] = np.matmul(mesh['vn'], m2w.T[:3, :3])
    mesh['vn'] /= np.linalg.norm(mesh['vn'], ord=2, axis=-1, keepdims=True)

    return mesh, data
    

def prepare_data_mitsuba():
    mesh = read_obj(mesh_path)
    mesh['v'] = np.pad(mesh['v'], [[0, 0], [0, 1]], 'constant', constant_values=1).astype(np.float32)
    mesh['v'] = np.matmul(mesh['v'], m2w.T)
    mesh['vn'] = np.matmul(mesh['vn'], m2w.T[:3, :3])
    mesh['vn'] /= np.linalg.norm(mesh['vn'], ord=2, axis=-1, keepdims=True)

    data = read_data_mitsuba(img_path, camera, views)

    return mesh, data






def learn_dgi():
    if data_type == 'real':
        mesh, data = prepare_data_colmap()
    else: # 'synth'
        mesh, data = prepare_data_mitsuba()

    # print(len(data))

    c = ReFlow()
    iters = 20 * 10**4
    c.train(mesh, data, train_seq, test_seq, 5 * 10**-4, iters, result_path, wr_img=True, wr_ckp=True, ow_reso=ow_reso)


    # c = NVDR()
    # iters = 5 * 10**4
    # c.train(mesh, data, train_seq, test_seq, 10 * 10**-4, iters, result_path, wr_img=True, wr_ckp=False, ow_reso=ow_reso)


    # n = 60
    # data_inter, train_seq_inter, render_seq_inter = inter_view(data, train_seq, n)
    # ckp_path = f'{result_path}/save'
    ckp_idx = iters
    # ckp_idx = 200000

    # c.render(mesh, data_inter, ckp_path, ckp_idx, train_seq_inter, render_seq_inter, result_path, inter=True, wr_img=True, adj_n=4, ow_reso=ow_reso)
    # ckp_path = f'{result_path}/save'

    # for ckp_idx in [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]:
    c.render(mesh, data, ckp_path, ckp_idx, train_seq, render_seq, result_path, inter=False, wr_img=True, adj_n=4, ow_reso=ow_reso)






# set path
pwd = pathlib.Path(__file__).absolute().parent
data_path = pwd.parent/'data'
result_path = pwd.parent/'result'/str(time.time())
# final_path = result_path/'final'
if result_path:
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path/'save')
    # backup
    os.system(f'cp {result_path}/../../code/utils.py {result_path}/save/utils_bk.py')
    os.system(f'cp {result_path}/../../code/main.py {result_path}/save/main_bk.py')
# if final_path:
#     os.makedirs(final_path, exist_ok=True)



# read data
model = 'xipot'
# obj_file = 'spot.obj'
camera_file = 'camera_config.json'
# np.savez()

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default=model)
args = parser.parse_args()

if model == 'ball':
    data_type = 'synth'
    data_path = data_path/'ball'
    mesh_path = data_path/'ball.obj'
    m2w = np.matmul(translate(0, -0.5, -0.15), rotate_y(-np.pi / 4))  # ball
    camera = read_camera_config(f'{data_path}/../{camera_file}')
    camera['origin'] = [0, 0, -3.9]
    camera['fov'] = 30
    camera['height'] = 512
    camera['width'] = 512
    ow_reso=None
    img_path = data_path/'images'
    train_seq = [i for i in range(0, 181, 10)]
    test_seq = [i for i in range(0, 181, 1) if i % 10 != 0]
    render_seq = [i for i in range(0, 181, 1)]
    views = [[x, y] for x in range(0, 1, 1) for y in range(0, 361, 2)]
elif model == 'brownpot':
    data_type = 'real'
    data_path = data_path/'brownpot'
    mesh_path = data_path/'mesh.obj'
    ref_path = data_path/'images'
    imglist_path = data_path/'images.txt'
    cmrlist_path = data_path/'cameras.txt'
    m2w = translate(0, 0, 0)
    ow_reso = None
    ref_seq = [i for i in range(0, 371, 5)]
    img_path = [f'{ref_path}/img_{i}.jpg' for i in ref_seq] 
    train_seq = [i for i in range(0, 75, 4)]
    test_seq = [i for i in range(0, 75, 2) if i % 4 != 0]
    render_seq = [i for i in range(0, 75, 1)]
elif model == 'teacup':
    data_type = 'real'
    data_path = data_path/'teacup'
    mesh_path = data_path/'mesh.obj'
    ref_path = data_path/'images'
    imglist_path = data_path/'images.txt'
    cmrlist_path = data_path/'cameras.txt'
    m2w = translate(0, 0, 0)
    ow_reso = None
    total = 261
    train_gap = 20
    test_gap = 5
    ref_seq = [i for i in range(0, total, 1)]
    img_path = [f'{ref_path}/img_{i}.jpg' for i in ref_seq] 
    train_seq = [i for i in range(0, total, train_gap)]
    test_seq = [i for i in range(0, total, test_gap) if i % train_gap != 0]
    render_seq = [i for i in range(0, total, 1)]
elif model == 'xipot':
    data_type = 'real'
    data_path = data_path/'xipot'
    mesh_path = data_path/'mesh.obj'
    ref_path = data_path/'images'
    imglist_path = data_path/'images.txt'
    cmrlist_path = data_path/'cameras.txt'
    m2w = translate(0, 0, 0)
    ow_reso = None
    ref_seq = [i for i in range(0, 401, 5)]
    img_path = [f'{ref_path}/img_{i}.jpg' for i in ref_seq] 
    train_seq = [i for i in range(0, 81, 4)]
    test_seq = [i for i in range(0, 81, 1) if i % 4 != 0]
    render_seq = [i for i in range(0, 81, 1)]
elif model == 'bluevase':
    data_type = 'real'
    data_path = data_path/'bluevase'
    mesh_path = data_path/'mesh.obj'
    ref_path = data_path/'images'
    imglist_path = data_path/'images.txt'
    cmrlist_path = data_path/'cameras.txt'
    m2w = translate(0, 0, 0)
    ow_reso = [640, 640]
    ref_seq = [i for i in range(0, 571, 5)]
    img_path = [f'{ref_path}/img_{i}.jpg' for i in ref_seq] 
    train_seq = [i for i in range(0, 115, 6)]
    test_seq = [i for i in range(0, 115, 1) if i % 6 != 0]
    render_seq = [i for i in range(0, 115, 1)]
    # colmap missing #21, #22, #31
    for i in range(len(train_seq)):
        if train_seq[i] >= 21:
            train_seq[i] -= 1
        if train_seq[i] >= 22:
            train_seq[i] -= 1
        if train_seq[i] >=31:
            train_seq[i] -= 1
    for i in range(len(test_seq)):
        if test_seq[i] >= 21:
            test_seq[i] -= 1
        if test_seq[i] >= 22:
            test_seq[i] -= 1
        if test_seq[i] >=31:
            test_seq[i] -= 1
    for i in range(len(render_seq)):
        if render_seq[i] >= 21:
            render_seq[i] -= 1
        if render_seq[i] >= 22:
            render_seq[i] -= 1
        if render_seq[i] >=31:
            render_seq[i] -= 1


# create tf session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess._default_session = sess.as_default()
sess._default_session.__enter__()


learn_dgi()




