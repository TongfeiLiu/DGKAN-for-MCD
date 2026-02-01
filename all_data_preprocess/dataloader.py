from skimage import io
import os
import numpy as np
import torch
from skimage.segmentation import slic, mark_boundaries
from all_data_preprocess.preprocess import process_images
from all_data_preprocess.preprocess_1 import process_images_1
from all_data_preprocess.Graph_construct import build_graphs
from PIL import Image
import torchvision.transforms as transforms
#定义硬件
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 数据集路径设置
root_dir = '/opt/data/private/xj/CGSL'
load_data_dir = '/data/dataset'


def load_images_Italy(N_SEG, Com):

    load_data = os.path.join(root_dir + load_data_dir)
    a = io.imread(os.path.join(load_data, 'Italy_1.bmp'))
    b = io.imread(os.path.join(load_data, 'Italy_2.bmp'))
    ref = io.imread(os.path.join(load_data, 'Italy_gt.bmp'))
    unique_values = np.unique(ref)
    if not np.array_equal(unique_values, [0, 255]):
        ref[ref < 200] = 0
        ref[ref >= 200] = 255
    image_t1 = torch.from_numpy(a).to(device)
    image_t2 = torch.from_numpy(b).to(device)

    objects = slic(image_t1.cpu().numpy(), n_segments=N_SEG, compactness=Com, start_label=0)
    objects = torch.from_numpy(objects).to(device)
    di_sp_b = mark_boundaries(image_t1.cpu().numpy(), objects.cpu().numpy())

    # 删除文件夹的创建代码
    img_t1, img_t2 = process_images(image_t1, image_t2, type_a='sar', type_b='opt')
    graph = build_graphs(img_t1, img_t2, objects, device)

    # 返回处理后的 graph、图像和参考数据
    return graph, image_t1, ref, objects, di_sp_b


def load_images_yellow(N_SEG, Com):

    load_data = os.path.join(root_dir + load_data_dir)
    a = io.imread(os.path.join(load_data, 'yellow1.png'))
    b = io.imread(os.path.join(load_data, 'yellow2.png'))
    ref = io.imread(os.path.join(load_data, 'yellow_C_gt.bmp'))
    unique_values = np.unique(ref)
    if not np.array_equal(unique_values, [0, 255]):
        ref[ref < 200] = 0
        ref[ref >= 200] = 255
    else:
        ref = ref
    image_t1 = torch.from_numpy(a).to(device)
    image_t2 = torch.from_numpy(b).to(device)

    objects = slic(image_t2.cpu().numpy(), n_segments=N_SEG, compactness=Com, start_label=0,channel_axis=None)
    objects = torch.from_numpy(objects).to(device)
    di_sp_b = mark_boundaries(image_t2.cpu().numpy(), objects.cpu().numpy())

    img_t1, img_t2 = process_images_1(image_t1, image_t2, type_a='opt', type_b='sar')
    graph = build_graphs(img_t1, img_t2, objects, device)

    return graph, image_t1, ref, objects, di_sp_b

def load_images_Dawn(N_SEG, Com):
    load_data = os.path.join(root_dir + load_data_dir)
    a = io.imread(os.path.join(load_data, 'T1.png'))
    b = io.imread(os.path.join(load_data, 'T2.png'))
    ref = io.imread(os.path.join(load_data, 'GT.png'))
    unique_values = np.unique(ref)
    if not np.array_equal(unique_values, [0, 255]):
        ref[ref < 200] = 0
        ref[ref >= 200] = 255
    else:
        ref=ref
    image_t1=torch.from_numpy(a).to(device)
    image_t2 = torch.from_numpy(b).to(device)
    objects = slic(image_t2.cpu().numpy(), n_segments=N_SEG, compactness=Com, start_label=0)
    objects = torch.from_numpy(objects).to(device)
    di_sp_b = mark_boundaries(image_t2.cpu().numpy(), objects.cpu().numpy())
    img_t1, img_t2 = process_images(image_t1, image_t2, type_a='sar', type_b='opt')
    graph = build_graphs(img_t1, img_t2, objects, device)
    return graph, image_t1, ref, objects, di_sp_b

def load_images_Gloucester2(N_SEG, Com):
    load_data = os.path.join(root_dir + load_data_dir)
    a = io.imread(os.path.join(load_data, 'T2-Img17-A.png'))
    b = io.imread(os.path.join(load_data, 'T1-Img17-Bc.png'))
    ref = io.imread(os.path.join(load_data, 'Img17-C.png'))
    unique_values = np.unique(ref)
    if not np.array_equal(unique_values, [0, 255]):
        ref[ref < 200] = 0
        ref[ref >= 200] = 255
    else:
        ref = ref
    image_t1 = torch.from_numpy(a).to(device)
    image_t2 = torch.from_numpy(b).to(device)

    objects = slic(image_t1.cpu().numpy(), n_segments=N_SEG, compactness=Com, start_label=0)
    objects = torch.from_numpy(objects).to(device)
    di_sp_b = mark_boundaries(image_t1.cpu().numpy(), objects.cpu().numpy())
    img_t1, img_t2 = process_images(image_t1, image_t2, type_a='opt', type_b='opt')
    graph = build_graphs(img_t1, img_t2, objects, device)

    return graph, image_t1,ref, objects, di_sp_b

def load_images_Gloucester1(N_SEG, Com):

    load_data = os.path.join(root_dir + load_data_dir)
    a = io.imread(os.path.join(load_data, 'img_t1.png'))
    b = io.imread(os.path.join(load_data, 'img_t2.png'))
    ref = io.imread(os.path.join(load_data, 'img_gt.png'))
    unique_values = np.unique(ref)
    if not np.array_equal(unique_values, [0, 255]):
        ref[ref < 200] = 0
        ref[ref >= 200] = 255
    else:
        ref = ref
    image_t1 = torch.from_numpy(a).to(device)
    image_t2 = torch.from_numpy(b).to(device)

    objects = slic(image_t1.cpu().numpy(), n_segments=N_SEG, compactness=Com, start_label=0)
    objects = torch.from_numpy(objects).to(device)
    di_sp_b = mark_boundaries(image_t1.cpu().numpy(), objects.cpu().numpy())
    img_t1, img_t2 = process_images(image_t1, image_t2, type_a='sar', type_b='opt')
    graph = build_graphs(img_t1, img_t2, objects, device)

    return graph, image_t1, ref, objects, di_sp_b

def load_images_California(N_SEG, Com):

    load_data = os.path.join(root_dir + load_data_dir)
    a = io.imread(os.path.join(load_data, 'C_t1.png'))
    b = io.imread(os.path.join(load_data, 'C_t2.png'))
    ref = io.imread(os.path.join(load_data, 'C_gt.png'))
    unique_values = np.unique(ref)
    if not np.array_equal(unique_values, [0, 255]):
        ref[ref < 200] = 0
        ref[ref >= 200] = 255
    else:
        ref = ref
    image_t1 = torch.from_numpy(a).to(device)
    image_t2 = torch.from_numpy(b).to(device)

    objects = slic(image_t1.cpu().numpy(), n_segments=N_SEG, compactness=Com, start_label=0,channel_axis=None)
    objects = torch.from_numpy(objects).to(device)
    di_sp_b = mark_boundaries(image_t1.cpu().numpy(), objects.cpu().numpy())
    img_t1, img_t2 = process_images_1(image_t1, image_t2, type_a='sar', type_b='opt')
    graph = build_graphs(img_t1, img_t2, objects, device)

    return graph, image_t1,ref, objects, di_sp_b

def load_images_France(N_SEG, Com):
    load_data = os.path.join(root_dir + load_data_dir)
    a = io.imread(os.path.join(load_data, 'Img7-Ac.png'))
    b = io.imread(os.path.join(load_data, 'Img7-Bc.png'))
    ref = io.imread(os.path.join(load_data, 'Img7-C.png'))
    # # 将 ndarray 转换为 PILImage
    pil_a = Image.fromarray(a)
    pil_b = Image.fromarray(b)

    # # 创建一个 transform 对象，用于缩小图像
    transform = transforms.Compose([
        transforms.Resize((800, 800)),  # 将图像缩小到指定的尺寸
    ])
    #
    #  应用 transform
    a= transform(pil_a)
    b = transform(pil_b)
    a = np.array(a)
    b = np.array(b)
    unique_values = np.unique(ref)
    if not np.array_equal(unique_values, [0, 255]):
        ref[ref < 200] = 0
        ref[ref >= 200] = 255
    else:
        ref = ref
    image_t1 = torch.from_numpy(a).to(device)
    image_t2 = torch.from_numpy(b).to(device)

    objects = slic(image_t1.cpu().numpy(), n_segments=N_SEG, compactness=Com, start_label=0)
    objects = torch.from_numpy(objects).to(device)
    di_sp_b = mark_boundaries(image_t1.cpu().numpy(), objects.cpu().numpy())
    img_t1, img_t2 = process_images(image_t1, image_t2, type_a='opt', type_b='opt')
    graph = build_graphs(img_t1, img_t2, objects, device)
    return graph, image_t1, ref,objects, di_sp_b
if __name__ == '__main__':
    load_images_France(N_SEG=1200, Com=23)