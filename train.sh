
python train.py --phase A  --train_dir dataset/dummy/train --val_dir dataset/dummy/val --epochsA 2

# python train.py --phase B  --ckpt checkpoints/phaseA_best.h5 --train_dir dataset/dummy/train --val_dir dataset/dummy/val --epochsB 2

# python train.py --phase AB --train_dir dataset/dummy/train --val_dir dataset/dummy/val --epochsA 2 --epochsB 2