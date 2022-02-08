## TransE
python run.py --score_func transe --opn mult --n_layer 0 --init_dim 100 --gcn_dim 100 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --x_ops "p" --name wn_lte --data wn18rr

## DistMult
python run.py --score_func distmult --opn mult --n_layer 0 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --x_ops "p.b.d" --name wn_lte --data wn18rr

## ConvE
python run.py --score_func conve --opn mult --n_layer 0 --init_dim 200 --gcn_drop 0.2 --feat_drop 0.1 --conve_hid_drop 0.4 --bias --batch 256 --num_filt 250 --gpu 0 --x_ops "p.b.d" --name wn_lte --data wn18rr