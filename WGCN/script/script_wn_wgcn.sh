# REPRO
## TransE
python run.py --gpu 0 --name wn_repro --decoder transe --n-layer 1 --init_emb_size 200 -d wn18rr -fa

## DistMult
python run.py --gpu 0 --name wn_repro --decoder distmult -d wn18rr -fb -fa -fd

## ConvE
python run.py --gpu 0 --name wn_repro --decoder conve -d wn18rr --n-layer 1 --init_emb_size 200 -fb -fd



# RAT
## TransE
python run.py --gpu 0 --name wn_rat --decoder transe --n-layer 1 --init_emb_size 200 -d wn18rr -fa --rat

## DistMult
python run.py --gpu 0 --name wn_rat --decoder distmult -d wn18rr -fa -fd --rat

## ConvE
python run.py --gpu 0 --name wn_rat --decoder conve -d wn18rr --n-layer 1 --init_emb_size 200 -fd --rat



# WNI
## TransE
python run.py --gpu 0 --name wn_wni --decoder transe --n-layer 1 --init_emb_size 200 -d wn18rr -fa --wni

## DistMult
python run.py --gpu 0 --name wn_wni --decoder distmult -d wn18rr-fa -fd --wni

## ConvE
python run.py --gpu 0 --name wn_wni --decoder conve -d wn18rr --n-layer 1 --init_emb_size 200 -fd --wni



# WSI
## TransE
python run.py --gpu 0 --name wn_wsi --decoder transe --n-layer 1 --init_emb_size 200 -d wn18rr -fa --wsi

## DistMult
python run.py --gpu 0 --name wn_wsi --decoder distmult -d wn18rr -fb -fa -fd --wsi

## ConvE
python run.py --gpu 0 --name wn_wsi --decoder conve -d wn18rr --n-layer 1 --init_emb_size 200 -fb -fd --wsi



# WSI_RAT
## TransE
python run.py --gpu 0 --name wn_wsi_rat --decoder transe --n-layer 1 --init_emb_size 200 -d wn18rr -fa --wsi --rat

## DistMult
python run.py --gpu 0 --name wn_wsi_rat --decoder distmult -d wn18rr-fa -fd --wsi --rat

## ConvE
python run.py --gpu 0 --name wn_wsi_rat --decoder conve -d wn18rr --n-layer 1 --init_emb_size 200 -fd --wsi  --rat
