#!/bin/bash

python3 train.py  --train_images="../CRACK500/traincrop/"  \
		  --train_annotations="../CRACK500/traincrop/"  \
		  --val_images="../CRACK500/valcrop/"  \
		  --val_annotations="../CRACK500/valcrop/"  \
		  --img_height=224  \
		  --img_width=224  \
