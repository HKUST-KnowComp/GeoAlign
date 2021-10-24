
source ./lock_utils.sh
source ./constants.sh

# echo $Embed_Args

set -e -x

USAGE='hyper_align.sh  taxonmy_dir checkpoint_dir nn th ws eig leafn wl gep_file_path'

execute_list_args(){
    echo "$@"
    [[ "$#" -ne 13 ]] && { echo $USAGE; exit 1; }
    taxonmy_dir=$1; shift;
    checkpoint_dir=$1; shift;
    DIMS=$1;         shift;
    LEAVES=$1;       shift;
    CHECKPOINT_BASE=$1; shift;
    MODEL=$1; shift;
    nn=$1;          shift;
    th=$1;          shift;
    ws=$1;          shift;
    eig=$1;         shift;
    leafn=$1;       shift;
    wl=$1;          shift;
    gep_file_name=$1;          shift;

    DATETIME=$(date +%Y%m%d_%H%M%S)
    dir_base="$nn"_"$th"_"$ws"_"$eig"_"$leafn"_"$wl"
    WORKING_DATA_DIR=$taxonmy_dir/"$DATETIME"_"$dir_base"

    WORKING_DATA_DIR=$(make_sure_dir $WORKING_DATA_DIR)
    WORKING_DATA_DIR=$(realpath $WORKING_DATA_DIR)
    log_file="$WORKING_DATA_DIR/embed.log"
    touch "$log_file"


    python AddLeafEdges.py $leafn $wl $EIGEN_EPS $gep_file_name $taxonmy_dir $WORKING_DATA_DIR | tee -a "$log_file"


    pushd poincare-embeddings
    DATE_TIME=$(date +"%Y%m%d_%H%M%S")

    checkpoint_dir_temp="$CHECKPOINT_BASE/$MODE_$DATE_TIME"
    checkpoint_dir_temp=$(make_sure_dir $checkpoint_dir_temp)
    checkpoint_temp="$checkpoint_dir_temp/checkpoint.bin"

    case "$MODEL" in
      "lorentz" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
      "poincare" ) EXTRA_ARGS=("-lr" "1.0");;
      "euclidean" ) EXTRA_ARGS=("-lr" "0.3");;
      * ) echo "$USAGE"; exit 1;;
    esac
    exp_embed_args=(-dim $DIMS -alignset $WORKING_DATA_DIR/aligned_taxonomy.csv \
        -trainset $taxonmy_dir/train_taxonomy.csv \
        -testset $taxonmy_dir/test_leaf_links.csv \
        -testhier $taxonmy_dir/train_taxonomy.csv \
        -checkpoint $checkpoint_temp -leaves $LEAVES \
        -manifold "$MODEL" \
        -filter -align -eval \
        -train_threads 12 \
        ${Embed_Args[@]} \
        ${EXTRA_ARGS[@]})
    if [[ $leafn = 1 ]]
    then
        exp_embed_args=(-naive_eval \
            ${exp_embed_args[@]}
        )
    fi

    echo "\$Parameters: $nn  $th $ws $eig $leafn $wl $EIGEN_EPS $EPOCHS $DIMS $MODE $MODEL" >> "$log_file"
    python3 embed.py ${exp_embed_args[@]} | tee -a "$log_file"
    popd
}

execute_list_args $@
