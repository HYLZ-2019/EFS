# All-in-focus Imaging from Event Focal Stack, CVPR 2023

Hanyue Lou, Minggui Teng, Yixin Yang, and Boxin Shi.

## How to test code:

1. Clone repository and create environment.

```
git clone https://github.com/HYLZ-2019/EFS.git
cd EFS
conda create --name efs --file env.txt
```

2. Download pretrained models and unzip, so that `addnet_best.pth` and `mergenet_best.pth` are in `pretrained/`.

3. Download example data and unzip, so that there are `example_data/real/blurry/*.png` & `example_data/real/events/*.npy`.

4. Run the test code:

```
conda activate efs
python make_all_in_focus.py
```