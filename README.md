# GeoAlign
Source code for AKBC 2021 paper [Manifold Alignment across Geometric Spaces for Knowledge Base Representation Learning](https://www.akbc.ws/2021/assets/pdfs/TPymTKJR-Pi.pdf).

### Requirements
* python: 3.6
* torch: 1.0.0
* scikit-learn
* pandas
* cython
* tqdm
* numpy

The hyperbolic embeddings in this repository (the poincare-embeddings folder) is inherited from the [poincare-embeddings repository](https://github.com/facebookresearch/poincare-embeddings).

## Data
We construct two taxonomies (YAGOwordnet and wikiObjects) and one knowledge graph (YAGOfacts) from [YAGO3](https://yago-knowledge.org/downloads/yago-3). Please refer to our paper for the data construction details.

#### Taxonomy data
* taxonomy.csv: This file contains the edges (i.e., the hypernymy relations) of the taxonomy.
* full_taxonomy.csv: This file contains the edges of the full transitive closure of the taxonomy, which is used in our experiments.
* full_transitive.txt / basic_edges.txt: The edges in the full transitive closure of the taxonomy / the transitive reduction of the taxonomy.
* types.txt / entities.txt / taxonomy_nodes.txt: The types / entities / union of types and entities in the taxonomy.
* taxonomy: The folder contains the training set and test set under different training rates. It also provides the corresponding data for the baseline models, including [AttH](https://github.com/HazyResearch/KGEmb), [HAKE](https://github.com/MIRALab-USTC/KGE-HAKE), [JOIE](https://github.com/JunhengH/joie-kdd19), [MurP](https://github.com/ibalazevic/multirelational-poincare), [TransC](https://github.com/davidlvxin/TransC), and [OpenKE](https://github.com/thunlp/OpenKE).

#### Knowledge graph data
* TransE_KG: The folder contains the pretrained TransE embeddings of YAGOfacts and the pairwise distance matrix of the pretrained embeddings.

## Usage
To run the whole framework, run:
```
zsh run_all.sh
```
or
```
echo $Training_Rate"\n"$Data"\n"$Dimension"\n"$Hyperbolic_Model"\n" | xargs -L 4 -P $PARALLEL_R bash hyper_label_rate.sh
```
where $Training_Rate = {1, 2, 3, 4, 5}; $Data = {YAGOwordnet, wikiObjects}; $Dimension is the embedding dimension of the hyperbolic space; $Hyperbolic_Model = {lorentz, poincare}; $PARALLEL_R is the number of parallel programs.
More parameters can be edited in constants.sh.

### Citation
If you find this repository useful for your research, please kindly cite our paper:
```angular2
@inproceedings{
xiao2021manifold,
title={Manifold Alignment across Geometric Spaces for Knowledge Base Representation Learning},
author={Huiru Xiao and Yangqiu Song},
booktitle={3rd Conference on Automated Knowledge Base Construction},
year={2021},
url={https://openreview.net/forum?id=TPymTKJR-Pi}
}
```