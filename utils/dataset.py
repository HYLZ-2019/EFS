import os
import os.path
import numpy as np
import random
import torch
import cv2
import glob
import torch.utils.data as udata

def RandomCrop(begin_img, end_img, event_stack, cropsize):
    c, h, w = begin_img.shape
    ri = random.randint(0, h-cropsize-1)
    rj = random.randint(0, w-cropsize-1)
    begin_img = begin_img[:,ri:ri+cropsize,rj:rj+cropsize]
    end_img = end_img[:,ri:ri+cropsize,rj:rj+cropsize]
    event_stack = event_stack[:,ri:ri+cropsize, rj:rj+cropsize]
    return begin_img, end_img, event_stack


def downsample_2D(img, r):
    h, w = img.shape
    sum = np.zeros((h//r, w//r))
    for i in range(r):
        for j in range(r):
            sum += img[i::r, j::r]
    return sum / (r*r)

def downsample_3D(img, r):
    c, h, w = img.shape
    sum = np.zeros((c, h//r, w//r))
    for i in range(r):
        for j in range(r):
            sum += img[:, i::r, j::r]
    return sum / (r*r)

# 把彩图转换为黑白
def read_img_y(img_path):
    img = cv2.imread(img_path)
    #return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    return y

def read_img(img_path):
    img = cv2.imread(img_path)
    img = np.transpose(img, (2,0,1)) # from (h, w, c) to (c, h, w)
    return img

def get_first_index_after(t, events):
    left = 0
    right = events.shape[0]
    min_ans = right
    while left < right:
        mid = (right + left) // 2
        if events[mid][0] <= t:
            left = mid + 1
            continue
        else:
            min_ans = mid
            right = mid - 1
    return min_ans

def make_stack(h, w, events, begin_i, end_i, stack_size):
    es = events[begin_i:end_i]
    event_cnt = end_i - begin_i
    cnt_per_frame = event_cnt // stack_size
    stack = np.zeros((stack_size, h, w))

    i_col = es[:,2]
    j_col = es[:,1]
    p_col = np.where(es[:,3]==1, 1, -1)

    for i in range(stack_size):
        np.add.at(stack[i], (i_col[i*cnt_per_frame:(i+1)*cnt_per_frame], j_col[i*cnt_per_frame:(i+1)*cnt_per_frame]), p_col[i*cnt_per_frame:(i+1)*cnt_per_frame])

    return stack
    

class Dataset_Train(udata.Dataset):
    def __init__(self, DataPath, CropSize, StackSize):
        super(Dataset_Train, self).__init__()
        self.DataPath = DataPath
        self.eventList = sorted(glob.glob(DataPath + 'events_random/*.npy'))

        self.CropSize = CropSize
        self.StackSize = StackSize
        self.samples_per_data = 20
        self.frames_per_data = 480

    def __len__(self):
        return len(self.eventList)*self.samples_per_data

    def __getitem__(self, index):
        eventpath = self.eventList[index % self.samples_per_data]
        events = np.load(eventpath)

        dataname = eventpath.split(".")[-2].split("\\")[-1]

        begin = random.randint(0, self.frames_per_data - 30)
        end = random.randint(begin+10, self.frames_per_data - 1)
        
        begin_path = self.DataPath + "frames/" + dataname + "/%04d"%begin + ".png"
        end_path = self.DataPath + "frames/" + dataname + "/%04d"%end + ".png"

        begin_img = read_img(begin_path)
        end_img = read_img(end_path)
        c, h, w = begin_img.shape

        begin_time = begin * 1000000 / 120 # 120fps
        begin_i = get_first_index_after(begin_time, events)
        end_time = end * 1000000 / 120
        end_i = get_first_index_after(end_time, events)

        event_stack = make_stack(h, w, events, begin_i, end_i, self.StackSize)

        begin_img, end_img, event_stack = RandomCrop(begin_img, end_img, event_stack, self.CropSize)
        return torch.Tensor(begin_img), torch.Tensor(end_img), torch.Tensor(event_stack)


class Dataset_Val(udata.Dataset):
    def __init__(self, ImgPath, EvePath):
        super(Dataset_Val, self).__init__()
        self.list = []
        # TODO

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        return self.list[index]


class Dataset_Test(udata.Dataset):
    def __init__(self, DataPath, StackSize):
        super(Dataset_Test, self).__init__()
        self.DataPath = DataPath
        self.eventList = sorted(glob.glob(DataPath + 'events_random/*.npy'))[::20]

        self.StackSize = StackSize
        self.samples_per_data = 3
        self.frames_per_data = 480

    def __len__(self):
        return len(self.eventList)*self.samples_per_data

    def __getitem__(self, index):
        eventpath = self.eventList[index // self.samples_per_data]
        events = np.load(eventpath)

        dataname = eventpath.split(".")[-2].split("\\")[-1]

        begin = 0
        end = self.frames_per_data // (self.samples_per_data+1) * (index % self.samples_per_data+1) - 1
        
        begin_path = self.DataPath + "frames/" + dataname + "/%04d"%begin + ".png"
        end_path = self.DataPath + "frames/" + dataname + "/%04d"%end + ".png"

        begin_img = read_img(begin_path)
        end_img = read_img(end_path)
        c, h, w = begin_img.shape

        begin_time = begin * 1000000 / 120 # 120fps
        begin_i = get_first_index_after(begin_time, events)
        end_time = end * 1000000 / 120
        end_i = get_first_index_after(end_time, events)

        event_stack = make_stack(h, w, events, begin_i, end_i, self.StackSize)

        return torch.Tensor(begin_img), torch.Tensor(end_img), torch.Tensor(event_stack)

class Dataset_Test_Real(udata.Dataset):
    def __init__(self, DataPath, StackSize):
        super(Dataset_Test_Real, self).__init__()
        self.DataPath = DataPath
        self.eventList = sorted(glob.glob(DataPath + 'events/[c]*.npy'))

        self.StackSize = StackSize
        self.samples_per_data = 5

    def __len__(self):
        return len(self.eventList)*self.samples_per_data

    def __getitem__(self, index):
        eventpath = self.eventList[index // self.samples_per_data]
        events = np.load(eventpath)

        dataname = eventpath.split(".")[-2].split("\\")[-1].split("_")[-1]

        begin = 0
        begin_path = self.DataPath + "firstFrame/" + dataname + ".png"
        begin_img = read_img(begin_path)
        c, h, w = begin_img.shape

        begin_i = 0
        end_i = events.shape[0] // (self.samples_per_data + 1) * (index % self.samples_per_data + 1) - 1

        # TMD, Voltmeter生成的event和真实相机是反的
        event_stack = - make_stack(h, w, events, begin_i, end_i, self.StackSize)

        if h == 260 and w == 346:
            return torch.Tensor(begin_img[:,0:260,0:340]), torch.Tensor(event_stack[:,0:260,0:340])
        
        if h == 2160 and w == 3840:
            img = downsample_3D(begin_img, 4)
            stack = downsample_3D(event_stack, 4)
            return torch.Tensor(img), torch.Tensor(stack)

        if h == 752 and w == 1082:
            return torch.Tensor(begin_img[:,0:752,0:1080]), torch.Tensor(event_stack[:,0:752,0:1080])

        if h % 4 != 0 or w % 4 != 0:
            new_h = 4 * int(h//4)
            new_w = 4 * int(w//4)
            return torch.Tensor(begin_img[:,0:new_h,0:new_w]), torch.Tensor(event_stack[:,0:new_h,0:new_w])


        return torch.Tensor(begin_img), torch.Tensor(event_stack)


