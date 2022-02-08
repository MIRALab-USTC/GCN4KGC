# REPRO
## TransE
python run.py --gpu 0 --name repro --decoder transe --n-layer 1 --init_emb_size 200 -fa

## DistMult
python run.py --gpu 0 --name repro --decoder distmult -fb -fa -fd

## ConvE
python run.py --gpu 0 --name repro --decoder conve -fb -fa -fd



# RAT
## TransE
python run.py --gpu 0 --name rat --decoder transe --n-layer 1 --init_emb_size 200 -fa --rat

## DistMult
python run.py --gpu 0 --name rat --decoder distmult -fb -fa -fd --rat

## ConvE
python run.py --gpu 0 --name rat --decoder conve -fb -fa -fd --rat



# WSI
## TransE
python run.py --gpu 0 --name wsi --decoder transe --n-layer 1 --init_emb_size 200 -fa --wsi

## DistMult
python run.py --gpu 0 --name wsi --decoder distmult -fb -fa -fd --wsi

## ConvE
python run.py --gpu 0 --name wsi --decoder conve -fb -fa -fd --wsi



# WNI
## TransE
python run.py --gpu 0 --name wni --decoder transe --n-layer 1 --init_emb_size 200 -fa --wni

## DistMult
python run.py --gpu 0 --name wni --decoder distmult -fa -fd --wni

## ConvE
python run.py --gpu 0 --name wni --decoder conve -fa -fd --wni



# WSI_RAT
## TransE
python run.py --gpu 0 --name wsi_rat --decoder transe --n-layer 1 --init_emb_size 200 -fa --wsi --rat

## DistMult
python run.py --gpu 0 --name wsi_rat --decoder distmult -fb -fa -fd --wsi --rat

## ConvE
python run.py --gpu 0 --name wsi_rat --decoder conve -fb -fa -fd --wsi --rat



# WSI_RAT_SS1000
## TransE
python run.py --gpu 0 --name wsi_rat_ss1000 --decoder transe --n-layer 1 --init_emb_size 200 -fa --wsi --rat --ss 1000

## DistMult
python run.py --gpu 0 --name wsi_rat_ss1000 --decoder distmult -fb -fa -fd --wsi --rat --ss 1000

## ConvE
python run.py --gpu 0 --name wsi_rat_ss1000 --decoder conve -fb -fa -fd --wsi --rat --ss 1000



# WSI_RAT_SS3000
## TransE
python run.py --gpu 0 --name wsi_rat_ss3000 --decoder transe --n-layer 1 --init_emb_size 200 -fa --wsi --rat --ss 3000

## DistMult
python run.py --gpu 0 --name wsi_rat_ss3000 --decoder distmult -fb -fa -fd --wsi --rat --ss 3000

## ConvE
python run.py --gpu 0 --name wsi_rat_ss3000 --decoder conve -fb -fa -fd --wsi --rat --ss 3000



# WSI_RAT_SS5000
## TransE
python run.py --gpu 0 --name wsi_rat_ss5000 --decoder transe --n-layer 1 --init_emb_size 200 -fa --wsi --rat --ss 5000

## DistMult
python run.py --gpu 0 --name wsi_rat_ss5000 --decoder distmult -fb -fa -fd --wsi --rat --ss 5000

## ConvE
python run.py --gpu 0 --name wsi_rat_ss5000 --decoder conve -fb -fa -fd --wsi --rat --ss 5000



# WSI_RAT_SS10000
## TransE
python run.py --gpu 0 --name wsi_rat_ss10000 --decoder transe --n-layer 1 --init_emb_size 200 -fa --wsi --rat --ss 10000

## DistMult
python run.py --gpu 0 --name wsi_rat_ss10000 --decoder distmult -fb -fa -fd --wsi --rat --ss 10000

## ConvE
python run.py --gpu 0 --name wsi_rat_ss10000 --decoder conve -fb -fa -fd --wsi --rat --ss 10000



# RAT_SS1000
## TransE
python run.py --gpu 0 --name rat_ss1000 --decoder transe --n-layer 1 --init_emb_size 200 -fa --rat --ss 1000

## DistMult
python run.py --gpu 0 --name rat_ss1000 --decoder distmult -fb -fa -fd --rat --ss 1000

## ConvE
python run.py --gpu 0 --name rat_ss1000 --decoder conve -fb -fa -fd --rat --ss 1000



# RAT_SS3000
## TransE
python run.py --gpu 0 --name rat_ss3000 --decoder transe --n-layer 1 --init_emb_size 200 -fa --rat --ss 3000

## DistMult
python run.py --gpu 0 --name rat_ss3000 --decoder distmult -fb -fa -fd --rat --ss 3000

## ConvE
python run.py --gpu 0 --name rat_ss3000 --decoder conve -fb -fa -fd --rat --ss 3000



# RAT_SS5000
## TransE
python run.py --gpu 0 --name rat_ss5000 --decoder transe --n-layer 1 --init_emb_size 200 -fa --rat --ss 5000

## DistMult
python run.py --gpu 0 --name rat_ss5000 --decoder distmult -fb -fa -fd --rat --ss 5000

## ConvE
python run.py --gpu 0 --name rat_ss5000 --decoder conve -fb -fa -fd --rat --ss 5000



# RAT_SS10000
## TransE
python run.py --gpu 0 --name rat_ss10000 --decoder transe --n-layer 1 --init_emb_size 200 -fa --rat --ss 10000

## DistMult
python run.py --gpu 0 --name rat_ss10000 --decoder distmult -fb -fa -fd --rat --ss 10000

## ConvE
python run.py --gpu 0 --name rat_ss10000 --decoder conve -fb -fa -fd --rat --ss 10000


# L1
# DistMult
python run.py --gpu 0 --name L1 --decoder distmult -fa -fd -fb --n-layer 1

# ConvE
python run.py --gpu 0 --name L1 --decoder conve -fb -fa -fd --n-layer 1


# L2
# TransE
python run.py --gpu 0 --name L2 --decoder transe --n-layer 1 --init_emb_size 200 -fa --n-layer 2


# L3
# TransE
python run.py --gpu 0 --name L3 --decoder transe --n-layer 1 --init_emb_size 200 -fa --n-layer 3

# DistMult
python run.py --gpu 0 --name L3 --decoder distmult -fa -fd -fb --n-layer 3

# ConvE
python run.py --gpu 0 --name L3 --decoder conve -fb -fa -fd --n-layer 3
