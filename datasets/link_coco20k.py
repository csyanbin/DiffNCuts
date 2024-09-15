import os

for name in open('coco_20k_filenames.txt').readlines():
    name = name.strip()
    cmd = "ln -s /data/yanbin/TokenCut/datasets/COCO/images/{}  COCO/images/train2014_20k".format(name)
    print(cmd)
    os.system(cmd)
