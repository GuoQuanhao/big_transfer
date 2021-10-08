# wget https://storage.googleapis.com/bit_models/vtab/BiT-M-{R50x1,R101x1,R50x3,R101x3,R152x4}-run{0,1,2}-{caltech101,diabetic_retinopathy,dtd,oxford_flowers102,oxford_iiit_pet,resisc45,sun397,cifar100,eurosat,patch_camelyon,smallnorb-elevation,svhn,dsprites-orientation,smallnorb-azimuth,clevr-distance,clevr-count,dmlab,kitti-distance,dsprites-xpos}.npz
import urllib.request
import os


https = 'https://storage.googleapis.com/bit_models/vtab/BiT-M-'
first = 'R50x1,R101x1,R50x3,R101x3,R152x4'
run = '0,1,2'
classes = 'caltech101,diabetic_retinopathy,dtd,oxford_flowers102,oxford_iiit_pet,resisc45,sun397,cifar100,eurosat,patch_camelyon,smallnorb-elevation,svhn,dsprites-orientation,smallnorb-azimuth,clevr-distance,clevr-count,dmlab,kitti-distance,dsprites-xpos'

for i in first.split(','):
    for j in run.split(','):
        for k in classes.split(','):
            # print(https+i+'-run'+j+'-'+k+'.npz')
            url = https+i+'-run'+j+'-'+k+'.npz'
            if not os.path.exists(i+'-run'+j+'-'+k+'.npz'):
                try:
                    print('downloading from %s' % url)
                    operation = urllib.request.urlretrieve(url, i+'-run'+j+'-'+k+'.npz')
                except:
                    print('No File')
            else:
                print(i+'-'+j+'-'+k+'.npz' + ' is exists!')
