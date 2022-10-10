
HW='ICX'
INP_DATA='/home/rpanchum/upenn/data/sidd-tcga/val_data_sidd-tcga_4mod_244/data_processed_1.csv'

run_mode_list=( 0 1 2 3 4 5)
fmwrk_mode_list=( 'PT-FP32' 'OV-FP32' 'OV-POT-INT8' 'OV-NNCF-INT8' 'OV-NNCF-PRUN06' 'OV-NNCF-QAT-Prun06-KD' )

# run_mode_list=( 5 )
# fmwrk_mode_list=( 'OV-NNCF-QAT-Prun06-KD' )

for i in ${!run_mode_list[@]}; do
    fmwrk_mode=${fmwrk_mode_list[i]}_${HW}
    cmd="mprof run python bench_mem_pt_ov.py -d $INP_DATA -r ${run_mode_list[i]} \
    -p ${fmwrk_mode} \
    2>&1 | tee infer_logs/bench_mem_${fmwrk_mode}.log"
    echo $cmd

    eval $cmd

    mprof plot -o bench_mem_${fmwrk_mode}.png
done

# Using /usr/bin/time -v for profiling
# for i in ${!run_mode_list[@]}; do
#     fmwrk_mode=${fmwrk_mode_list[i]}_${HW}
#     cmd="/usr/bin/time -v python bench_mem_pt_ov.py -d $INP_DATA -r ${run_mode_list[i]} \
#     -p ${fmwrk_mode} \
#     2>&1 | tee infer_logs/bench_mem_${fmwrk_mode}.log"
#     echo $cmd

#     eval $cmd
# done