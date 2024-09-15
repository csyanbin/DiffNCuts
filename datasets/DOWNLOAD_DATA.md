## Download datasets
Adapted from TokenCut Github: https://github.com/YangtaoWANG95/TokenCut/blob/master/DOWNLOAD_DATA.md

### ImageNet
Download from ImageNet website and put into subfolders as below:
```
ILSVRC2012/imagenet/train
ILSVRC2012/imagenet/val
```

### ECSSD
To download [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html):
```
cd ../datasets/ECSSD
bash download_data.sh
```

The dataset should be organized as:

```
../datasets/ECSSD/
                  ├── img
                  ├── gt
```

### DUTS
To download [DUTS](http://saliencydetection.net/duts/#org3aad434):
```
cd ../datasets/DUTS_Test
bash download_data.sh
```

The dataset should be organized as:
```
../datasets/DUTS_Test/
                     ├── img
                     ├── gt
```


### DUT-OMRON
To downlaod [DUT_OMRON](http://saliencydetection.net/dut-omron/#org96c3bab):
```
cd ../datasets/DUT-OMRON
bash download_data.sh
```

The dataset should be organized as:
```
../datasets/DUT-OMRON/
                     ├── img
                     ├── gt
```




### VOC2007
To download [Pascal VOC 2007 train and validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/): 
```
cd datasets/VOC2007/
bash download_voc07.sh
```
The dataset should be organized as : 

```
./datasets/VOC2007/VOCdevkit
                    ├── VOC2007/
                        ├── JPEGImages
                        ├── Annotations
                        ...
```
### VOC2012

To download [Pascal VOC 2012 train and validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/): 
```
cd datasets/VOC2012/
bash download_voc12.sh
```
The dataset should be organized as : 

```
./datasets/VOC2012/VOCdevkit
                    ├── VOC2012/
                        ├── JPEGImages
                        ├── Annotations
                        ...
```

### COCO
To download [COCO dataset](https://cocodataset.org/#home): 
```
cd datasets/COCO/
bash download_coco.sh
```
The dataset should be organized as : 
```
./datasets/COCO/
               ├── images/
               ├── annotations/
```


