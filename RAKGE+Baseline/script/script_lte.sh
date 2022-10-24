## TransE
python run.py --score_func transe --opn mult --gpu 3 --epoch 3000 --x_ops "d" --hid_drop 0.7 --n_layer 0 --init_dim 200 --name lte --data Comp7 --scale 0 --input_drop 0.7 --hid_drop 0.7

## DistMult
python run.py --score_func distmult --opn mult --gpu 3 --x_ops "d" --n_layer 0 --init_dim 200 --name lte --data Sparse237 --hid_drop 0.2 --scale 0

## ConvE
python run.py --score_func conve --opn mult --gpu 2 --x_ops "d" --n_layer 0 --init_dim 200 --name lte --data Sparse237 --scale 0 --hid_drop 0.2 --feat_drop 0.3 --conve_hid_drop 0.2

## TransMS
python run.py --score_func transms --opn mult --gpu 3 --epoch 3000 --x_ops "d" --hid_drop 0.0 --n_layer 0 --init_dim 200 --name lte --data Comp7 --scale 0 --input_drop 0.0

