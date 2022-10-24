## MT-KGNN
python run_mtkgnn.py --name MTKGNN --init_dim 50  --tolerance 100 --lr 0.001 --epoch 2000 --lbl_smooth 0 --data FB15k-237 --gpu 0

## KBLN
python run.py --name KBLN --data Comp7 --n_layer 0 --hid_drop 0.8 --gpu 3 --literal

