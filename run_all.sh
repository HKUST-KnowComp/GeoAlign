
source ./constants.sh

echo 5"\n"YAGOwordnet"\n"50"\n"lorentz"\n" | xargs -L 4 -P $PARALLEL_R bash hyper_label_rate.sh