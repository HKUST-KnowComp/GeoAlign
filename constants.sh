PARALLEL_R=3
PARALLEL_ARGS=2
set -x -e

GPU=-1

# Experment Args List
NUM_NEIGHBOR_S=(6)
T_HEAT_S=(100000000000)
WEI_SIMI_S=('0.5')
EIG_S=(11)
LEAF_NEI_S=(5)
WEI_LEAF_S=('0.05')
EIGEN_EPS=-1


EPOCHS=300
Embed_Args=(-epochs $EPOCHS \
       -negs 50 \
       -burnin 20 \
       -dampening 0.75 \
       -ndproc 4 \
       -batchsize 20 \
       -eval_each 50 \
       -fresh \
       -sparse \
       -burnin_multiplier 0.01 \
       -neg_multiplier 0.1 \
       -lr_type constant \
       -dampening 1.0 \
       -gpu $GPU)

