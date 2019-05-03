#!/bin/bash

python test.py  --save_weights_path="segnet_basic.h5"  \
		--test_images="../CRACK500/testcrop/"  \
                --output_path="test" \
		--img_height=224  \
		--img_width=224   \
		--model="segnet_basic" 
