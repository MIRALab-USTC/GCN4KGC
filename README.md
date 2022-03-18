# Rethinking Graph Convolutional Networks in Knowledge Graph Completion

This is the code of paper 
**Rethinking Graph Convolutional Networks in Knowledge Graph Completion**. 
Zhanqiu Zhang, Jie Wang, Jieping Ye, Feng Wu. WWW 2022. [[arXiv](https://arxiv.org/abs/2202.05679)]

## Requirements
- python 3.7
- torch 1.8
- dgl 0.7


## Reproduce the Results
Pleaes run the commands in `RGCN+CompGCN+LTE/script` or `WGCN/script` to reproduce the results.

Meaning of different options.
- rat: random adjacency tensors.
- wsi: without self-loop information.
- wni: without neighbor information.
- ss: sample set sizes for random sampled neighbors.


## Citation
If you find this code useful, please consider citing the following paper.
```
@inproceedings{WWW22_GCN4KGC,
 author = {Zhanqiu Zhang and Jie Wang and Jieping Ye and Feng Wu},
 booktitle = {The Web Conference 2022},
 title = {Rethinking Graph Convolutional Networks in Knowledge Graph Completion},
 year = {2022}
}
```

## Acknowledgement
We refer to the code of [CompGCN](https://github.com/malllabiisc/CompGCN), [WGCN](https://github.com/maqy1995/sacn_dgl), and [DGL](https://github.com/dmlc/dgl). Thanks for their contributions.
