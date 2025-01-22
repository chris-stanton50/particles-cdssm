#!/bin/zsh

# Define possible values for arguments
arg1_values=("ts_mv_ou" "mv_ou" "iou" "ibm")
arg2_values=("-f" "-s" "-os")
arg3_values=("-test")
arg4_values=("10" "20" "30")


# Loop through all possible combinations of arguments
for arg1 in "${arg1_values[@]}"; do
    for arg2 in "${arg2_values[@]}"; do
        for arg3 in "${arg3_values[@]}"; do
            for arg4 in "${arg4_values[@]}"; do
                echo "Running with arguments: $arg1 $arg2 $arg3"
                python3 parallel_filt_smth.py -local "$arg1" "$arg2" "$arg3" "$arg4" 
                echo ""
            done
        done
    done
done