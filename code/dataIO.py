import numpy as np
import scipy as sp
import imageio
import json
import pathlib
import copy
import xml
import matplotlib.pyplot as plt
from transform import *

# todo:
# load_obj(): materials, normalize
# load_material():

def read_data_colmap(img_path, cmrlst_path, imglst_path):
    img_dic = {img_path[i].split('/')[-1]: [i, img_path[i]] for i in range(len(img_path))}

    cmr = {}
    with open(cmrlst_path) as file:
        while True:
            l = file.readline()
            if not l:
                break
            l = l.split()
            if l[0].isdigit():
                cmr[int(l[0])] = [int(l[2]), int(l[3]), float(l[4]), float(l[5])]

    data = []
    with open(imglst_path) as file:
        while True:
            l = file.readline()
            if not l:
                break
            l = l.split()
            if l[0] == '#':
                continue
            n, qw, qx, qy, qz, tx, ty, tz, cmr_id, img_name = l
            q = [float(qw), float(qx), float(qy), float(qz)]
            t = [float(tx), float(ty), float(tz)]
            cmr_id = int(cmr_id)
            c = cmr[cmr_id]
            img_info = img_dic[img_name]
            # [img_idx, camera, ref_img]
            data.append([img_info[0], set_camera_colmap(q, t, c), read_image(img_info[1])])
            # print(f'load img: {img_name} with camera {cmr_id}')

            file.readline()

    data.sort(key=lambda x: x[0])
    print(f'images and cameras loaded: total {len(data)}')
    return data

def read_data_mitsuba(img_path, camera, views):
    data = []
    for i in range(len(views)):
        x, y = views[i]
        img = read_image(f'{img_path}/{str(x)}_{str(y)}_scene.png')
        c_t = set_camera_mitsuba(camera, [x, y, 0], [0, 0, 0])
        data.append([i, c_t, img])
    return data






def read_obj(path):
    with open(path) as file:
        lines = file.readlines()

    obj_data = {
        'v': [],    # vertices
        'vt': [],   # texcoords
        'vn': [],   # normals
    }


    vert_types = ['v', 'vt', 'vn']
    for i in vert_types:
        obj_data[i] = []

    face_types = ['fv', 'fvt', 'fvn']
    for i in face_types:
        obj_data[i] = []

    for l in lines:
        l = l.split()
        if len(l) == 0:
            continue
        l_type = l[0].lower()
        # vertices
        if l_type in vert_types:
            l_data = np.array([float(i) for i in l[1:]]).astype(np.float32)[:3]
            if l_type == 'vt':
                l_data = np.array([l_data[0], 1.0 - l_data[1]])
            # if l_type == 'v' and len(l_data == 3):
            #     l_data = np.hstack([l_data, 1])
            obj_data[l_type].append(l_data)
        # faces
        elif l_type == 'f':
            l_data = np.array([i.split('/') for i in l[1:]]).astype(np.int)
            v_count = l_data.shape[0]
            for i in range(v_count - 2):  # use triangle
                if len(l_data[0]) < 2:
                    continue
                obj_data['fv'].append(l_data[[0, i + 1, i + 2], 0] - 1)
                obj_data['fvt'].append(l_data[[0, i + 1, i + 2], 1] - 1)

                if len(l_data[0]) < 3: # no normal
                    vp = [obj_data['v'][i] for i in obj_data['fv'][-1]]
                    vn = np.cross(vp[1] - vp[0], vp[2] - vp[0])
                    vn /= np.sum(vn**2)**0.5
                    obj_data['vn'].append(vn)
                    obj_data['fvn'].append([len(obj_data['vn']) - 1 for _ in range(3)])
                else: 
                    # print(l_data, l_data[[0, i + 1, i + 2], 2])
                    obj_data['fvn'].append(l_data[[0, i + 1, i + 2], 2] - 1)

        # materials
        # refer to nvdiffmd obj.py


    msg = 'obj file loaded: '
    for k in obj_data.keys():
        obj_data[k] = np.array(obj_data[k])
        msg += k + str(obj_data[k].shape) + '; '
    print(msg)

    return obj_data



def vert_nt(obj_data):

    import tensorflow as tf
    uv = np.zeros([obj_data['v'].shape[0], 2])
    normal = np.zeros([obj_data['v'].shape[0], 3])
    for i in range(len(obj_data['fv'])):
        for j in range(len(obj_data['fv'][i])):
            vert_idx = obj_data['fv'][i][j]
            normal_idx = obj_data['fvn'][i][j]
            uv_idx = obj_data['fvt'][i][j]
            normal[vert_idx] += obj_data['vn'][normal_idx]
            uv[vert_idx] = obj_data['vt'][uv_idx]

    normal = tf.nn.l2_normalize(normal, axis=-1)
    with tf.Session() as sess:
        normal = sess.run(normal)
    return normal, uv



# def create_texture(w, h):
#     return np.random.random((w, h, 3)).astype(np.float32)
#     return np.full((w, h, 3), 0.5)

def read_image(path):
    return np.array(imageio.imread(path).astype(np.float32)/255.0).astype(np.float32)



def write_image(path, pic, ext=True):
    if ext:
        pic = np.rint(np.array(pic) * 255.0)
    pic = np.clip(pic, 0, 255).astype(np.uint8)
    # print(pic)
    imageio.imsave(path, pic)


def write_gif(path, imgs, fps):
    imgs = [np.clip(np.rint(np.array(i) * 255.0), 0, 255).astype(np.uint8) for i in imgs]
    imageio.mimsave(path, imgs, 'GIF', duration=1 / fps)


def write_camera_config(path, camera):
    caemra = copy.deepcopy(camera)
    for (k, v) in camera.items():
        if type(v) == np.ndarray:
            camera[k] = [str(i) for i in list(v)]
        else:
            camera[k] = str(v)
    with open(path, 'w') as file:
        file.write(json.dumps(camera))

def read_camera_config(path):
    with open(path, 'r') as file:
        camera = json.loads(file.read())
    for (k, v) in camera.items():
        if type(v) == list:
            camera[k] = np.array(v).astype(np.float32)
        else:
            camera[k] = np.float32(v)
    return camera

def read_mitsuba_xml(path):
    with open(path, 'r') as file:
        return file.read()

def update_camera_config(xml, camera):
    for (k, v) in camera.items():
        if type(v) == np.ndarray:
            camera[k] = '"' + ', '.join([str(i) for i in list(v)]) + '"'
        else:
            camera[k] = '"' + str(float(v)) + '"'
        print(k, camera[k])


def replace_origin(xml, pos):
    pos = ', '.join([str(i) for i in list(pos)])
    s = "origin"
    lx = len(xml)
    ls = len(s)
    flag = True
    i = b = e = 0
    while i < (lx - ls):
        if flag and xml[i:i+ls] == s:
            flag = False
            i += 2 + ls
            b = i
        elif xml[i] == '"' and not flag:
            e = i
            break
        i += 1
    return xml[:b] + pos + xml[e:]


def read_bundle(path):
    with open(path) as file:
        file.readline()
        n = int(file.readline().split()[0])
        views = []
        for _ in range(n):
            f = np.array(file.readline().split()[0]).astype(np.float32)
            r = np.reshape([float(i) for i in ' '.join([file.readline() for _ in range(3)]).split()], [3, 3]).astype(np.float32)
            t = np.array([float(i) for i in file.readline().split()]).astype(np.float32)
            views.append([f, r, t])
    return views

def read_imglst(path):
    with open(path) as file:
        lst = []
        while True:
            l = file.readline()
            if not l:
                break
            n, w, h = l.split()
            n = path.parent/n
            w = int(w)
            h = int(h)
            lst.append([n, h, w])
    return lst



            


def write_loss(path, train, test=None):
    train = np.array(train)
    plt.plot(train[:, 0], train[:, 1], label="train", color="#FF0000")
    if test is not None:
        test = np.array(test)
        plt.plot(test[:, 0], test[:, 1], label="test", color="#0000FF")
    plt.ylim(0, 0.5)
    plt.xlabel("iters")
    plt.ylabel("loss")
    # plt.show()
    plt.savefig(f'{path}/loss_curve.png')
    with open(f'{path}/loss_log.txt', 'a+') as file:
        file.write('iters\t\tloss\ntrain\n')
        file.writelines(['\t\t'.join([str(j) for j in i]) + '\n' for i in train])
        if test is not None:
            file.write('test\n')
            file.writelines(['\t\t'.join([str(j) for j in i]) + '\n' for i in test])









if __name__ == '__main__':

    pwd = pathlib.Path(__file__).absolute().parent
    data_path = pwd.parent/'data'

    xml = read_mitsuba_xml(f'{data_path}/scene.xml')
    # print(xml)
    camera = read_camera_config(f'{data_path}/camera_config.json')

    # xml = update_camera_config(xml, camera)
    print(replace_origin(np.array([1,2,3])))





    # print(camera_config)