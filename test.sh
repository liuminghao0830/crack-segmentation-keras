#!/bin/bash

python3 predict.py  --save_weights_path="segnet.h5"  \
		    --test_images="../CRACK500/testcrop/"  \
                    --output_path="test" \
		    --img_height=224  \
		    --img_width=224 
