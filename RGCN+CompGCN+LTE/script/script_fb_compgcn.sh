# REPRO
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name repro

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --name repro

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name repro



# RAT
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name rat --rat

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --name rat --rat

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name rat --rat



# WSI
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name wsi --wsi

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --name wsi --wsi

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name wsi --wsi



# WNI
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --nobn --name wni --wni

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --name wni --wni

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name wni --wni



# WSI_RAT
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name wsi_rat --wsi --rat

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --name wsi_rat --wsi --rat

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name wsi_rat --wsi --rat



# WSI_RAT_SS1000
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name wsi_rat_ss1000 --wsi --rat --ss 1000

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --wsi --rat --name wsi_rat_ss1000 --ss 1000

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name wsi_rat_ss1000 --wsi --rat --ss 1000



# WSI_RAT_SS3000
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name wsi_rat_ss3000 --wsi --rat --ss 3000

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --wsi --rat --name wsi_rat_ss3000 --ss 3000

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name wsi_rat_ss3000 --wsi --rat --ss 3000



# WSI_RAT_SS5000
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name wsi_rat_ss5000 --wsi --rat --ss 5000

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --wsi --rat --name wsi_rat_ss5000 --ss 5000

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name wsi_rat_ss5000 --wsi --rat --ss 5000



# WSI_RAT_SS10000
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name wsi_rat_ss10000 --wsi --rat --ss 10000

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --wsi --rat --name wsi_rat_ss10000 --ss 10000

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name wsi_rat_ss10000 --wsi --rat --ss 10000



# RAT_SS1000
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name rat_ss1000 --rat --ss 1000

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --rat --name rat_ss1000 --ss 1000

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name rat_ss1000 --rat --ss 1000



# RAT_SS3000
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name rat_ss3000 --rat --ss 3000

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --rat --name rat_ss3000 --ss 3000

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name rat_ss3000 --rat --ss 3000



# RAT_SS5000
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name rat_ss5000 --rat --ss 5000

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --rat --name rat_ss5000 --ss 5000

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --name rat_ss5000 --rat --ss 5000



# RAT_SS10000
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 1 --name rat_ss10000 --rat --ss 10000

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 150 --rat --name rat_ss10000 --ss 10000

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --ecd poch 500 --batch 256 --n_layer 1 --name rat_ss10000 --rat --ss 10000



# L1
## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 150 --name L1



# L2
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 2 --name L2

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --name L2



# L3
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --epoch 500 --batch 256 --n_layer 3 --name L3

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 3 --gcn_dim 150 --name L3

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 3 --name L3



# no_LTR
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --init_dim 200 --gcn_dim 200 --epoch 500 --batch 256 --n_layer 1 --name noLTR --noltr

## DistMult
python run.py --score_func distmult --opn mult --gpu 7 --epoch 500 --batch 256 --n_layer 2 --init_dim 150 --embed_dim 150 --gcn_dim 150 --name noLTR --noltr

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --init_dim 200 --gcn_dim 200 --n_layer 1 --name noLTR --noltr
