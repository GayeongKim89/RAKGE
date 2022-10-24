# TransE
python run.py --gpu 2 --n_layer 0 --init_dim 200 --literal --att_dim 200 --epoch 3000 --head_num 5 --name TransE_Gate_att --lbl_smooth 0 --scale 0 --data Comp7 --x_ops "d" --hid_drop 0.7 --input_drop 0.7
# TransE + loss
python run.py --gpu 3 --n_layer 0 --init_dim 200 --literal --att_dim 200 --epoch 3000 --head_num 5 --name TransE_Gate_att --lbl_smooth 0 --scale 0.25 --data Comp7 --x_ops "d" --hid_drop 0.7 --input_drop 0.7

# DistMult
python run.py --gpu 2 --n_layer 0 --init_dim 200 --literal --att_dim 200 --epoch 3000 --head_num 5 --name DistMult_Gate_att --lbl_smooth 0.0 --scale 0 --data Sparse-FB15k-237 --x_ops "d" --hid_drop 0.4

# DistMult + loss
python run.py --gpu 0 --n_layer 0 --init_dim 200 --literal --att_dim 200 --epoch 3000 --head_num 5 --name DistMult_Gate_att --lbl_smooth 0.1 --scale 0.1 --data Comp7 --x_ops "d" --hid_drop 0.4

# ConvE
python run.py --gpu 3 --n_layer 0 --init_dim 200 --literal --att_dim 200 --hid_drop 0.4 --input_drop 0.4 --feat_drop 0.5 --conve_hid_drop 0.4  --epoch 3000 --head_num 5 --name ConvE_Gate_att --lbl_smooth 0 --scale 0.0 --data Sparse-FB15k-237

# ConvE + loss
python run.py --gpu 2 --n_layer 0 --init_dim 200 --literal --att_dim 200 --x_ops "p.d" --input_drop 0.4 --feat_drop 0.5 --conve_hid_drop 0.4  --epoch 1500 --head_num 5 --name ConvE_Gate_att --lbl_smooth 0 --scale 0 --data YAGO15k

