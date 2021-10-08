# Big Transfer (BiT): General Visual Representation Learning

本项目基于官方复现代码[google-research-big_transfer](https://github.com/google-research/big_transfer)，有关论文请查阅[Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)

*Paddle framework recurrence by Quanhao Guo*


## 环境准备

**为了方便起见以及基于`ImageNet`数据集的大小，本项目的测试数据为Cifar10和Cifar100，读取方式采用Paddle自带API，本项目未挂载任何数据集，感兴趣的同学可以测试`ImageNet`**

**本项目基于`paddlepaddle-2.0.2`框架复现，如果使用~AIStudio`环境，你可能需要更换`Paddle`版本**
```
python -m pip install paddlepaddle-gpu==2.0.2.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
```

## 微调BiT

下载BiT模型。 工提供针对5种不同架构在ILSVRC-2012(BiT-S)或ImageNet-21k(BiT-M)上进行预训练的模型：ResNet-50x1，ResNet-101x1，ResNet-50x3，ResNet-101x3和ResNet-152x4
```
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz
```
**注意本项目仅支持`.npz`格式读取，不支持`.h5`格式**，默认情况下，你需要将模型权重存储在此项目的根文件夹中，即`big_transfer`文件夹下

例如，如果要下载在ImageNet-21k上经过预训练的ResNet-152x2，请运行以下命令：
```
python -m bit_paddle.train --name cifar10_`date +%F_%H%M%S` --model BiT-M-R152x2 --logdir bit_logs --dataset cifar10 --datadir cifar10
```
其中`--name`代表训练过程中存储模型以及日志的文件夹名，cifar10_\`date +%F_%H%M%S\`格式表示时间后缀，你可以使用更具区别的名称代替如`cifar10_BiT-M-R152x2`

**注意到：本项目暂不支持`few-shot`训练方式`--examples_per_class <K>`选项无效**

### BiT-M models在ILSVRC-2012进行微调

为了方便起见，同时提供了已经在ILSVRC-2012数据集上进行过微调的BiT-M模型。 可以通过添加-ILSVRC2012后缀，例如下载模型
```
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1-ILSVRC2012.npz
```

作者发布了本文提到的所有架构，因此您可以在精度或速度之间进行选择：R50x1，R101x1，R50x3，R101x3，R152x4。 在上述模型文件的路径中，只需将R50x1替换为您选择的体系结构即可

### BiT-M models在19 VTAB-1k tasks进行微调

作者还针对VTAB-1k基准测试中包含的19个任务中的每一个发布了经过微调的模型。 我们对每个模型运行了三次，并释放了每个运行。 这意味着我们总共发布了`5x19x3=285`个模型，并希望这些模型可用于进一步分析迁移学习。

可以通过以下模式下载文件：
```
wget https://storage.googleapis.com/bit_models/vtab/BiT-M-{R50x1,R101x1,R50x3,R101x3,R152x4}-run{0,1,2}-{caltech101,diabetic_retinopathy,dtd,oxford_flowers102,oxford_iiit_pet,resisc45,sun397,cifar100,eurosat,patch_camelyon,smallnorb-elevation,svhn,dsprites-orientation,smallnorb-azimuth,clevr-distance,clevr-count,dmlab,kitti-distance,dsprites-xpos}.npz
```
其中我提供了下载脚本，你可以使用`python command.py`进行下载，以及[百度网盘链接](https://pan.baidu.com/s/16P7zh3EZ7U32asm17tiaUA)，提取码：qe29，有需求的可以自行下载

## 测试结果

### CIFAR测试结果

#### BiT-M-R101x3

本结果仅展示了作者测定的最高分以及AIStudio平台训练得分
| Dataset  | Ex/cls |TF2|Jax|PyTorch|**Paddle**|
|:---|:---:|:---:|:---:|:---:|:---:|
| CIFAR10  |full|**98.5**|**98.4**|**98.6**|**98.62**|
| CIFAR100 |full|**90.8**|**91.2**|**91.2**|**91.55**|

#### BiT-M-R152x2

| Dataset  | Ex/cls |Jax|PyTorch|**Paddle**|
|:---|:---:|:---:|:---:|:---:|
|CIFAR10|full|**98.5**|**98.5**|**98.45**|
|CIFAR100|full|**91.2**|**91.3**|**90.98**|

(TF2 models not yet available.)

#### BiT-M-R50x1

|Dataset| Ex/cls |TF2|Jax|PyTorch|**Paddle**|
|:---|:---:|:---:|:---:|:---:|:---:|
|CIFAR10|full|**97.2**|**97.3**|**97.4**|**97.32**|
|CIFAR100|full|**86.5**|**86.4**|**86.6**|**86.76**|

### IMAGENET2012测试结果

**本结果在4卡TeslaV100(32G)训练结果**

|Dataset|Model|BatchSize/GPU|**Paddle**|
|:---|:---:|:---:|:---:|
|IMAGENET2012|BiT-M-R152x4|1|**top1 60.02% , top5 85.57%**|
|IMAGENET2012|BiT-M-R50x1|64|**top1 81.29% , top5 96.07%**|
|IMAGENET2012|BiT-M-R101x3|8|**top1 79.87%, top5 95.67%**|

**注意到，由于原始论文在大规模集群TPU下训练，其batchsize设定为128/256，由于模型越重，越难以重现论文中的batchsize参数，这对于训练来说是致命的。本测试结果仅证明本项目能够胜任ImageNet2012数据集的训练**

**所有结果的训练日志都在`bit_logs`下**

<img src="https://ai-studio-static-online.cdn.bcebos.com/b1401e8651c444d4aac5f148c57411cc8871afb1676345f69a37735143ed8427" width="200"/>

**有关IMAGENET2012训练的任何细节参考[脚本任务BigTransfer](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2076715)**

# **关于作者**

<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| 主页        | [Deep Hao的主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
