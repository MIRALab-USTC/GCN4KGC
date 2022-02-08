# REPRO
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 150 --init_dim 150 --epoch 500 --batch 256 --num_base 5 --n_layer 1 --encoder rgcn --name repro

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 150 --init_dim 150 --encoder rgcn --num_base 5 --name repro

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 128 --init_dim 128 --embed_dim 128 --k_w 16 --k_h 8 --num_base 8 --encoder rgcn --name repro



# RAT
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 150 --init_dim 150 --epoch 500 --batch 256 --num_base 5 --n_layer 1 --encoder rgcn --name rat --rat

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch. 500 --batch 256 --n_layer 1 --gcn_dim 150 --init_dim 150 --encoder rgcn --num_base 5 --name rat --rat

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 128 --init_dim 128 --embed_dim 128 --k_w 16 --k_h 8 --num_base 8 --encoder rgcn --name rat --rat



# WSI
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 150 --init_dim 150 --epoch 500 --batch 256 --num_base 5 --n_layer 1 --encoder rgcn --name wsi --wsi

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 150 --init_dim 150 --encoder rgcn --num_base 5 --name wsi --wsi

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 128 --init_dim 128 --embed_dim 128 --k_w 16 --k_h 8 --num_base 8 --encoder rgcn --name wsi --wsi



# WNI
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 150 --init_dim 150 --epoch 500 --batch 256 --num_base 5 --n_layer 1 --encoder rgcn --name wni --wni

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 150 --init_dim 150 --encoder rgcn --num_base 5 --name wni --wni

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 128 --init_dim 128 --embed_dim 128 --k_w 16 --k_h 8 --num_base 8 --encoder rgcn --name wni --wni



# WSI_RAT
## TransE
python run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 150 --init_dim 150 --epoch 500 --batch 256 --num_base 5 --n_layer 1 --encoder rgcn --name wsi_rat --wsi --rat

## DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 150 --init_dim 150 --encoder rgcn --num_base 5 --name wsi_rat --wsi --rat

## ConvE
python run.py --score_func conve --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 1 --gcn_dim 128 --init_dim 128 --embed_dim 128 --k_w 16 --k_h 8 --num_base 8 --encoder rgcn --name wsi_rat --wsi --rat
