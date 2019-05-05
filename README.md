# crack-segmentation-keras
Pave-road crack image segmentation using SegNet and Unet architectures.

## Dataset
<a href="https://drive.google.com/drive/folders/1y9SxmmFVh0xdQR-wdchUmnScuWMJ5_O-" target="_blank">CRACK500</a>

## Train
Adjust the paths of training and validation set in train.sh, and then
```
  sh train.sh
```

## Test
Adjust the paths of testing set in test.sh, and then
```
  sh test.sh
```
to calculate mIOU of model.
