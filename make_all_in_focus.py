from asyncore import read
import os
from turtle import down
import cv2
import torch

torch.cuda.current_device()
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob
import time
import skimage
import matplotlib as mpl

from model import AddNet, MergeStackNet
from utils import Options,  Dataset_Test, Dataset_Test_Real, downsample_3D

def normalize_uint8(arr):
    maxval = np.max(arr)
    minval = np.min(arr)
    return np.uint8((arr - minval)*255.0 / (maxval-minval))

def colorRB(img):
    temp = np.zeros([img.shape[0], img.shape[1], 3]) + 255
    channel_r = temp[:, :, 0]
    channel_g = temp[:, :, 1]
    channel_b = temp[:, :, 2]
    factor = 1
    channel_r[img > 0] = 255 - img[img > 0]*(255)*factor
    channel_g[img > 0] = 255 - img[img > 0]*(255)*factor
    channel_b[img > 0] = 255

    channel_r[img < 0] = 255
    channel_g[img < 0] = 255 + img[img < 0]*(255)*factor
    channel_b[img < 0] = 255 + img[img < 0]*(255)*factor
    temp[:, :, 0] = channel_r
    temp[:, :, 1] = channel_g
    temp[:, :, 2] = channel_b
    return temp

def normalize(matrix):
    minv = np.min(matrix)
    maxv = np.max(matrix)
    matrix[matrix>0] = matrix[matrix>0]/max(maxv, 0.001)
    matrix[matrix<0] = matrix[matrix<0]/min(abs(minv), -0.001)
    return matrix

def ev2img(matrix):
    # input matrix size is [N,H,W]
    # Convert to RGB visualization events
    matrix = np.float32(np.sum(matrix, axis=0))
    matrix = normalize(matrix)
    matrix = colorRB(matrix)
    return np.uint8(matrix)

def ev2img_zxy(matrix, cl=3):
    norm = mpl.colors.Normalize(vmin=-cl, vmax=cl, clip=True)
    color = mpl.cm.get_cmap("bwr")
    n = norm(matrix)
    c = color(n)
    return np.uint8(c*255)[:,:,0:3]

def get_first_index_after(t, events):
    left = 0
    right = events.shape[0]
    min_ans = right - 1
    while left < right:
        mid = (right + left) // 2
        if events[mid][0] <= t:
            left = mid + 1
            continue
        else:
            min_ans = mid
            right = mid - 1
    return min_ans

def stack_events(h, w, events, stacksize):
    event_cnt = events.shape[0]
    cnt_per_frame = event_cnt // stacksize
    stack = np.zeros((stacksize, h, w))
    i_col = events[:,2]
    j_col = events[:,1]
    p_col = np.where(events[:,3]==1, 1, -1)
    for i in range(stacksize):
        np.add.at(stack[i], (i_col[i*cnt_per_frame:(i+1)*cnt_per_frame], j_col[i*cnt_per_frame:(i+1)*cnt_per_frame]), p_col[i*cnt_per_frame:(i+1)*cnt_per_frame])
    return stack


# Reconstruct an image and evaluate the sharpness.
def make_image_and_eval(image, events, evrefocusnet, device, voltmeter):
    h, w, c = image.shape
    STACK_SIZE = 64
    evstack = stack_events(h, w, events, STACK_SIZE)
    # In the training data, all event polarities were flipped because there was a bug in the Voltmeter simulator.
    # So we need to flip all correct data to the wrong polarity to get correct results......
    if voltmeter != True:
        evstack = - evstack
    pd = evrefocusnet(torch.Tensor(image.transpose(2,0,1)).to(device).unsqueeze(0), \
        torch.Tensor(evstack).to(device).unsqueeze(0))
    pd_y = torch.mean(pd, dim=1) # get rid of color channels
    sharpness = torch.var(pd_y)
    pd_np = np.uint8(torch.squeeze(pd.cpu()).numpy()).transpose((1,2,0))
    sharpness_np = torch.squeeze(sharpness.cpu()).numpy()
    return pd_np, sharpness_np


# Do the golden rate search to find the refocus timestamps, and reconstruct the refocused images using EvRefocusNet.
def make_stack_images(imgpath, evpath, stackpath, patch_N, evrefocusnet, device, voltmeter=False, output_patches=False):
    img = cv2.imread(imgpath)
    h, w, c = img.shape
    events = np.int32(np.load(evpath))
    events_binned = np.zeros((patch_N, patch_N), dtype=object)
    
    patch_h = ((h // patch_N) // 4 + 1) * 4
    patch_w = ((w // patch_N) // 4 + 1) * 4

    def patch_borders(i, j):
        mi = patch_h*i
        mj = patch_w*j
        Mi = patch_h*(i+1)
        Mj = patch_w*(j+1)
        if Mi > h:
            mi = h - patch_h
            Mi = h
        if Mj > w:
            mj = w - patch_w
            Mj = w
        return mi, Mi, mj, Mj
    # Split one (n, 4) event list into N*N sublists, such that events_binned[i,j] contains all events in patch (i,j).
    for i in range(patch_N):
        for j in range(patch_N):
            mi, Mi, mj, Mj = patch_borders(i, j)
            in_patch = np.logical_and( \
                np.logical_and(events[:,2]>=mi, events[:,2]<Mi), \
                np.logical_and(events[:,1]>=mj, events[:,1]<Mj))
            events_binned[i,j] = events[in_patch]
            events_binned[i,j][:,2] -= mi
            events_binned[i,j][:,1] -= mj
    
    # For each patch, search for in-focus moment with golden-rate-search.
    # This could be done in parallel.
    golden_ratio = (1+5**0.5)/2
    event_cnt_thres = 1000
    
    found_times = []

    for i in range(patch_N):
        for j in range(patch_N):
            mi, Mi, mj, Mj = patch_borders(i, j)
            imgpatch = img[mi:Mi, mj:Mj]
            evpatch = events_binned[i,j]
            if evpatch.shape[0] < event_cnt_thres*2:
                # Not enough events to run algorithm. Means that the patch doesn't change and probably doesn't have useful texture.
                # But we still need N*N frames
                found_times.append(0)
                continue

            left = 0
            right = evpatch.shape[0]
            while right - left > event_cnt_thres:
                t1 = max(left+1, int(right - (right-left)/golden_ratio))
                t2 = min(right-1, int(left + (right-left)/golden_ratio))
                # left < t1 < t2 < right

                img1, val1 = make_image_and_eval(imgpatch, evpatch[0:t1], evrefocusnet, device, voltmeter)
                img2, val2 = make_image_and_eval(imgpatch, evpatch[0:t2], evrefocusnet, device, voltmeter)

                if (val1 < val2):
                    left = t1
                else:
                    right = t2
            found_time = (left + right) // 2
            t = evpatch[found_time,0]
            ti_global = get_first_index_after(t, events)
            found_times.append(ti_global)

            if output_patches:
                # 整张图
                my_img, my_val = make_image_and_eval(img, events[0:ti_global], evrefocusnet, device, voltmeter)
                border_color = np.array([255, 255, 255])
                bwidth = 3
                # 描四条边
                my_img[mi:Mi, mj:mj+bwidth] = border_color
                my_img[mi:Mi, Mj-bwidth:Mj] = border_color
                my_img[mi:mi+bwidth, mj:Mj] = border_color
                my_img[Mi-bwidth:Mi, mj:Mj] = border_color
                cv2.imwrite(stackpath+"patch_img_%02d_%02d.png"%(i,j), my_img)
    
    selected_times = sorted(found_times)
    event_stack = np.zeros((64, h, w))

    frame_cnt = 0
    output_stack = np.zeros((len(selected_times), h, w, c))
    for t in selected_times:
        big_img, big_val = make_image_and_eval(img, events[0:t], evrefocusnet, device, voltmeter)

        output_stack[frame_cnt] = big_img

        init_time = events[0][0]
        max_time = np.max(events[:, 0])
        time_bin_cnt = 64

        time_interval = (max_time - init_time) // time_bin_cnt + 1
        cur_time = events[t,0]
        s_min = get_first_index_after(cur_time - time_interval, events)
        s_max = get_first_index_after(cur_time + time_interval, events)
        pols = np.where(events[s_min:s_max, 3] == 0, -1, 1)
        np.add.at(event_stack[frame_cnt], (events[s_min:s_max, 2], events[s_min:s_max, 1]), pols)

        frame_cnt += 1

    #np.save(stackpath+"event_stack.npy", event_stack)
    return output_stack, event_stack

# Merge with gradients. Used for ablation studies.
def merge_stack_gradient(stack):
    # Stack: (stacksize, h, w, c)
    ss, h, w, c = stack.shape

    weights = np.zeros((ss, h, w))
    for x in range(ss):
        dx = cv2.Sobel(stack[x], ddepth=-1, dx=1, dy=0).sum(axis=2)
        dy = cv2.Sobel(stack[x], ddepth=-1, dx=0, dy=1).sum(axis=2)
        gradient = dx*dx + dy*dy
        weights[x] = cv2.GaussianBlur(gradient, (41,41), sigmaX=7, sigmaY=7)
    
    weight_sum = weights.sum(axis=0).reshape((1, h, w))
    weights_norm = np.where(weight_sum < 1e-4, 0, weights / weight_sum).reshape((ss, h, w, 1))

    val = (stack * weights_norm).sum(axis=0)

    weight_indexes = np.indices(stack.shape)[0]
    weight_avg_index = (weights_norm*weight_indexes).sum(axis=0)
    weight_viz = cv2.applyColorMap(normalize_uint8(weight_avg_index.sum(axis=2)), 19)
    
    return val, weight_viz

# Use EvMergeNet to merge stack.
def merge_stack_with_model(imgstack, evstack, evmergenet, device):
    ss, h, w, c = imgstack.shape
    pd, weights, _ = evmergenet(torch.Tensor(imgstack.transpose(0, 3, 1, 2)).to(device).unsqueeze(0), \
        torch.Tensor(evstack).to(device).unsqueeze(0))
    pd_np = np.uint8(torch.squeeze(pd.cpu()).numpy()).transpose((1,2,0))
    weights_np = torch.squeeze(weights.cpu())

    weight_sum = weights_np.sum(axis=0).reshape((1, h, w))
    weights_norm = np.where(weight_sum < 1e-4, 0, weights_np / weight_sum).reshape((ss, h, w, 1))
    weight_indexes = np.indices(imgstack.shape)[0]
    weight_avg_index = (weights_norm*weight_indexes).sum(axis=0)
    weight_viz = cv2.applyColorMap(normalize_uint8(weight_avg_index.sum(axis=2)), 19)
    
    return pd_np, weight_viz



def test_all_data_in_path(base_path, model_add, model_merge, device):
    os.makedirs(base_path + "aif_result/", exist_ok=True)
    os.makedirs(base_path + "weights/", exist_ok=True)
    patch_N = 8

    datanames = [x.split("/")[-1].split("\\")[-1].split(".")[0] for x in sorted(glob.glob(base_path+"events/*.npy"))]
    for dataname in datanames:
        print(dataname)
        impath = base_path+"blurry/"+dataname+".png"
        evpath = base_path+"events/"+dataname+".npy"
        start_time = time.time()

        OUTPUT_PATCHES = False
        stackpath = base_path + "stacks/" + dataname + "/"
        if OUTPUT_PATCHES:
            os.makedirs(stackpath, exist_ok=True)
        imgstack, evstack = make_stack_images(impath, evpath, stackpath, patch_N, model_add, device, voltmeter=False, output_patches=OUTPUT_PATCHES)

        mergepath = base_path + "aif_result/" + dataname + ".png"
        aif, weights = merge_stack_with_model(imgstack, evstack, model_merge, device)
        cv2.imwrite(mergepath, aif)
        
        weightpath = base_path + "weights/" + dataname + ".png"
        cv2.imwrite(weightpath, weights)

        end_time = time.time()
        print("Used seconds: %d" % int(end_time-start_time))


if __name__ == "__main__":
    args = Options().parse()
    if args.use_gpus:
        device = torch.device("cuda")
        device_ids = [Id for Id in range(torch.cuda.device_count())]
    else:
        device = torch.device("cpu")

    model_add = AddNet(args.StackSize)
    model_merge = MergeStackNet(args.StackSize)

    if args.use_gpus:
        model_add = torch.nn.DataParallel(model_add.cuda(), device_ids=[device_ids[0]], output_device=device_ids[0])
        model_merge = torch.nn.DataParallel(model_merge.cuda(), device_ids=[device_ids[0]], output_device=device_ids[0])
        
    model_add.load_state_dict(torch.load("pretrained/addnet_best.pth", map_location='cpu'), strict=False)
    model_merge.load_state_dict(torch.load("pretrained/mergenet_best.pth", map_location='cpu'), strict=False)
        
    model_add.eval()
    model_merge.eval()

    with torch.no_grad():
        #test_all_data_in_path("example_data/real_evfocus/", model_add, model_merge, device)
        test_all_data_in_path("example_data/SMLFD-test/", model_add, model_merge, device)

