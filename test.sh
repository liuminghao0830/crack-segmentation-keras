#!/bin/bash

python predict.py  --save_weights_path="FCN_Vgg16.h5"  \
		    --test_images="../CRACK500/testcrop/"  \
                    --output_path="test" \
		    --img_height=224  \
		    --img_width=224   \
		    --model="FCN_Vgg16" 
