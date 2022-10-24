# TransE
python run.py --gpu 3 --n_layer 0 --init_dim 200 --literal --epoch 3000 --name TransELiteral_gate --lbl_smooth 0 --data Comp7 --hid_drop 0.7 --input_drop 0.7 --scaler 0

# DistMult
python run.py --gpu 0 --n_layer 0 --init_dim 200 --literal --epoch 3000 --name DistMultLiteral_gate --lbl_smooth 0.1 --data comp3 --hid_drop 0.2


# ConvE
python run.py --gpu 3 --n_layer 0 --init_dim 200 --literal --input_drop 0.2 --feat_drop 0.3 --conve_hid_drop 0.2  --epoch 1500 --name ConvELiteral_gate --lbl_smooth 0 --data Sparse-FB15k-237

