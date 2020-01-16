

import argparse
import subprocess


ckp = 'pretrained/mobilenetv2_1.0-0c6065bc.pth'

for i in range(1):
    p = subprocess.Popen(' '.join(['python',
                                   'imagenet.py',
                                   '-a mobilenetv2',
                                   '-d /home/dl/DATA/ImageNet/ILSVRC/Data/CLS-LOC',
                                   '--weight '+ckp,
                                   '--width-mult 1.0',
                                   '--input-size 224',
                                   '--status prune',
                                   '--method min_weight',
                                   '--ckp_out ./checkpoints/mw-nm-'+str(i)+'.pth']), shell=True)
    p.wait()
    ckp = './checkpoints/mw-nm-'+str(i)+'.pth'
    p = subprocess.Popen(' '.join(['python',
                                   'imagenet.py',
                                   '-a mobilenetv2',
                                   '-d /home/dl/DATA/ImageNet/ILSVRC/Data/CLS-LOC',
                                   '--weight ' + ckp,
                                   '--width-mult 1.0',
                                   '--input-size 224',
                                   '--status train',
                                   '--ckp_out ' + ckp]), shell=True)
    p.wait()
