source ./lock_utils.sh
source ./constants.sh

echo $Embed_Args

set -e -x



yield_args(){
  checkpoint_dir=$1
  taxonmy_dir=$2
  for nn in "${NUM_NEIGHBOR_S[@]}"; do
    for th in "${T_HEAT_S[@]}"; do
      for ws in "${WEI_SIMI_S[@]}"; do
        for eig in "${EIG_S[@]}"; do
          gep_file_name=$checkpoint_dir/GEP_"$nn"_"$th"_"$ws"_"$eig".npz
          if [[ ! -f $gep_file_name ]];
          then
            python ManifoldAlignment.py -n $nn \
              -t $th -w $ws -e $eig \
              -dist $checkpoint_dir \
              -pt $taxonmy_dir \
              -out $gep_file_name > /dev/null
          fi
          for leafn in "${LEAF_NEI_S[@]}"; do
            for wl in "${WEI_LEAF_S[@]}"; do
                echo $nn $th $ws $eig $leafn $wl $gep_file_name
	          done
          done
        done
      done
    done
  done
}

execute_r(){
    set -x
    r=$1
    data=$2
    DIMS=$3
    MODEL=$4
    [[ -z $r ]] && exit 1
    HOMEDATA_DIR="./data/$data"
    TAXONOMY_DIR_BASE="$HOMEDATA_DIR/taxonomy"
    taxonmy_dir="$TAXONOMY_DIR_BASE/0.$r"
    echo $taxonmy_dir
    LEAVES="$HOMEDATA_DIR/entities.txt"
    CHECKPOINT_BASE="$HOMEDATA_DIR/Experiments/poincare-embeddings/checkpoints"
    pushd poincare-embeddings
    DATE_TIME=$(date +"%Y%m%d_%H%M%S")
    checkpoint_dir=$CHECKPOINT_BASE/"$MODE"_"0.$r"_"$DIMS"_"$DATE_TIME"
    checkpoint_dir=$(make_sure_dir $checkpoint_dir)
    checkpoint="$checkpoint_dir/checkpoint.bin"
    WORKING_DATA_DIR="$taxonmy_dir/$DATE_TIME"_"pretrain"

    WORKING_DATA_DIR=$(make_sure_dir $WORKING_DATA_DIR)
    WORKING_DATA_DIR=$(realpath $WORKING_DATA_DIR)
    log_file="$WORKING_DATA_DIR/embed.log"
    touch "$log_file"

    echo "\$Parameters: $EPOCHS $DIMS $MODE $MODEL" >> "$log_file"

    case "$MODEL" in
      "lorentz" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
      "poincare" ) EXTRA_ARGS=("-lr" "1.0");;
      "euclidean" ) EXTRA_ARGS=("-lr" "0.3");;
      * ) echo "$USAGE"; exit 1;;
    esac
    local_embed_args=(-dim $DIMS -trainset $taxonmy_dir/train_taxonomy.csv \
        -testhier $taxonmy_dir/train_taxonomy.csv \
        -checkpoint $checkpoint \
        -manifold "$MODEL" -eval -filter \
        -train_threads 12 -leaves $LEAVES \
        ${Embed_Args[@]} \
        ${EXTRA_ARGS[@]})

    python3 embed.py ${local_embed_args[@]} | tee -a "$log_file"
    popd

  yield_args $checkpoint_dir  $taxonmy_dir \
    | xargs   -L 1 -P $PARALLEL_ARGS bash hyper_align.sh \
      $taxonmy_dir $checkpoint_dir $DIMS $LEAVES $CHECKPOINT_BASE $MODEL
}

execute_r $@
