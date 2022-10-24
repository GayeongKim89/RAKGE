
# REPRO
## TransE
python run.py --score_func transe --opn mult --gpu 3 --gamma 9 --hid_drop 0.8 --gcn_dim 150 --init_dim 150 --epoch 500 --batch 256 --num_base 5 --n_layer 1 --encoder rgcn --name repro --data Comp7

## DistMult
python run.py --score_func distmult --opn mult --gpu 3 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 150 --init_dim 150 --encoder rgcn --num_base 5 --name repro --data comp3

## ConvE
python run.py --score_func conve --opn mult --gpu 3 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 128 --init_dim 128 --embed_dim 128 --k_w 16 --k_h 8 --num_base 8 --encoder rgcn --name repro --data Comp7

