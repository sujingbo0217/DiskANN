#!/bin/bash

export data_dir="/localdata/jsu068"

### Random filter label sets
zipf_label_set=(1 13 26 36 39 40 50)

## bigann
# for l in "${zipf_label_set[@]}"; do
    echo "Running compute_groundtruth_for_filters with filter_label: ${l}"
    build/apps/utils/compute_groundtruth_for_filters --data_type uint8 --dist_fn l2 \
        --base_file ${data_dir}/bigann/bigann.128D.10M.euclidean.base.u8bin \
        --query_file ${data_dir}/bigann/bigann.128D.10K.euclidean.query.u8bin \
        --gt_file ${data_dir}/bigann/bigann.gt.${l}.bin \
        --label_file ${data_dir}/bigann/bigann.10M.L50.zipf0.75.base.txt \
        --K 10 --filter_label ${l} --universal_label 0
done

build/apps/build_memory_index --data_type uint8 --dist_fn l2 \
    --data_path ${data_dir}/bigann/bigann.128D.10M.euclidean.base.u8bin \
    --index_path_prefix ${data_dir}/bigann/bigann_R32_L90_filtered_index \
    -R 32 --FilteredLbuild 90 --alpha 1.2 \
    --label_file ${data_dir}/bigann/bigann.10M.L50.zipf0.75.base.txt --universal_label 0

build/apps/build_stitched_index --data_type uint8 \
    --data_path ${data_dir}/bigann/bigann.128D.10M.euclidean.base.u8bin \
    --index_path_prefix ${data_dir}/bigann/bigann_R32_L100_SR64_stitched_index \
    -R 32 -L 100 --stitched_R 64 --alpha 1.2 \
    --label_file ${data_dir}/bigann/bigann.10M.L50.zipf0.75.base.txt --universal_label 0

for l in "${zipf_label_set[@]}"; do
    build/apps/search_memory_index --data_type uint8 --dist_fn l2 \
        --index_path_prefix ${data_dir}/bigann/bigann_R32_L90_filtered_index \
        --query_file ${data_dir}/bigann/bigann.128D.10K.euclidean.query.u8bin \
        --gt_file ${data_dir}/bigann/bigann.gt.${l}.bin --filter_label ${l} \
        -K 10 -L 10 15 20 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 370 390 410 430 450 470 490 510 530 550 570 590 610 630 650 \
        --result_path ${data_dir}/bigann/filtered_search_results_${l}

    build/apps/search_memory_index --data_type uint8 --dist_fn l2 \
        --index_path_prefix ${data_dir}/bigann/bigann_R32_L100_SR64_stitched_index \
        --query_file ${data_dir}/bigann/bigann.128D.10K.euclidean.query.u8bin \
        --gt_file ${data_dir}/bigann/bigann.gt.${l}.bin --filter_label ${l} \
        -K 10 -L 10 15 20 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 370 390 410 430 450 470 490 510 530 550 570 590 610 630 650 \
        --result_path ${data_dir}/bigann/stitched_search_results_${l}
done

## deep
for l in "${zipf_label_set[@]}"; do
    echo "Running compute_groundtruth_for_filters with filter_label: ${l}"
    build/apps/utils/compute_groundtruth_for_filters --data_type float --dist_fn mips \
        --base_file ${data_dir}/deep/deep.96D.10M.angular.base.fbin \
        --query_file ${data_dir}/deep/deep.96D.10K.angular.query.fbin \
        --gt_file ${data_dir}/deep/deep.gt.${l}.bin \
        --label_file ${data_dir}/deep/deep.10M.L50.zipf0.75.base.txt \
        --K 10 --filter_label ${l} --universal_label 0
done

build/apps/build_memory_index --data_type float --dist_fn mips \
    --data_path ${data_dir}/deep/deep.96D.10M.angular.base.fbin \
    --index_path_prefix ${data_dir}/deep/deep_R32_L90_filtered_index \
    -R 32 --FilteredLbuild 90 --alpha 1.2 \
    --label_file ${data_dir}/deep/deep.10M.L50.zipf0.75.base.txt --universal_label 0

build/apps/build_stitched_index --data_type float \
    --data_path ${data_dir}/deep/deep.96D.10M.angular.base.fbin \
    --index_path_prefix ${data_dir}/deep/deep_R32_L100_SR64_stitched_index \
    -R 32 -L 100 --stitched_R 64 --alpha 1.2 \
    --label_file ${data_dir}/deep/deep.10M.L50.zipf0.75.base.txt --universal_label 0

for l in "${zipf_label_set[@]}"; do
    build/apps/search_memory_index --data_type float --dist_fn mips \
        --index_path_prefix ${data_dir}/deep/deep_R32_L90_filtered_index \
        --query_file ${data_dir}/deep/deep.96D.10K.angular.query.fbin \
        --gt_file ${data_dir}/deep/deep.gt.${l}.bin --filter_label ${l} \
        -K 10 -L 10 15 20 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 370 390 410 430 450 470 490 510 530 550 570 590 610 630 650 \
        --result_path ${data_dir}/deep/filtered_search_results_${l}

    build/apps/search_memory_index --data_type float --dist_fn mips \
        --index_path_prefix ${data_dir}/deep/deep_R32_L100_SR64_stitched_index \
        --query_file ${data_dir}/deep/deep.96D.10K.angular.query.fbin \
        --gt_file ${data_dir}/deep/deep.gt.${l}.bin --filter_label ${l} \
        -K 10 -L 10 15 20 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 370 390 410 430 450 470 490 510 530 550 570 590 610 630 650 \
        --result_path ${data_dir}/deep/stitched_search_results_${l}
done

### Natural filter label sets
marco_label_set=(50)
yfcc_label_set=(23 29 89 29 1589 5893)

## marco
for l in "${marco_label_set[@]}"; do
    build/apps/utils/compute_groundtruth_for_filters --data_type float --dist_fn l2 \
        --base_file ${data_dir}/marco/embedding/marco.768D.10M.euclidean.fbin \
        --query_file ${data_dir}/marco/query/marco.768D.10K.euclidean.fbin \
        --gt_file ${data_dir}/marco/marco.gt.${l}.bin \
        --label_file ${data_dir}/marco/embedding/marco.filter.base.10M.txt \
        --K 10 --filter_label ${l} --universal_label 0
done

build/apps/build_memory_index --data_type float --dist_fn l2 \
    --data_path ${data_dir}/marco/embedding/marco.768D.10M.euclidean.fbin \
    --index_path_prefix ${data_dir}/marco/marco_R32_L90_filtered_index \
    -R 32 --FilteredLbuild 90 --alpha 1.2 \
    --label_file ${data_dir}/marco/embedding/marco.filter.base.10M.txt --universal_label 0

build/apps/build_stitched_index --data_type float \
    --data_path ${data_dir}/marco/embedding/marco.768D.10M.euclidean.fbin \
    --index_path_prefix ${data_dir}/marco/marco_R32_L100_SR64_stitched_index \
    -R 32 -L 100 --stitched_R 64 --alpha 1.2 \
    --label_file ${data_dir}/marco/embedding/marco.filter.base.10M.txt --universal_label 0

for l in "${marco_label_set[@]}"; do
    build/apps/search_memory_index --data_type float --dist_fn l2 \
        --index_path_prefix ${data_dir}/marco/marco_R32_L90_filtered_index \
        --query_file ${data_dir}/marco/query/marco.768D.10K.euclidean.fbin \
        --gt_file ${data_dir}/marco/marco.gt.${l}.bin --filter_label ${l} \
        -K 10 -L 10 15 20 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 370 390 410 430 450 470 490 510 530 550 570 590 610 630 650 \
        --result_path ${data_dir}/marco/filtered_search_results_${l}

    build/apps/search_memory_index --data_type float --dist_fn l2 \
        --index_path_prefix ${data_dir}/marco/marco_R32_L100_SR64_stitched_index \
        --query_file ${data_dir}/marco/query/marco.768D.10K.euclidean.fbin \
        --gt_file ${data_dir}/marco/marco.gt.${l}.bin --filter_label ${l} \
        -K 10 -L 10 15 20 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 370 390 410 430 450 470 490 510 530 550 570 590 610 630 650 \
        --result_path ${data_dir}/marco/stitched_search_results_${l}
done

## yfcc (unlimited runtime, TLE)
for l in "${yfcc_label_set[@]}"; do
    echo "Running compute_groundtruth_for_filters with filter_label: ${l}"
    build/apps/utils/compute_groundtruth_for_filters --data_type uint8 --dist_fn l2 \
        --base_file ${data_dir}/yfcc/yfcc.192D.10M.euclidean.base.u8bin \
        --query_file ${data_dir}/yfcc/yfcc.192D.100K.euclidean.query.u8bin \
        --gt_file ${data_dir}/yfcc/yfcc.gt.${l}.bin \
        --label_file ${data_dir}/yfcc/yfcc.filter.base.txt \
        --K 10 --filter_label ${l} --universal_label 0
done

build/apps/build_memory_index --data_type uint8 --dist_fn l2 \
    --data_path ${data_dir}/yfcc/yfcc.192D.10M.euclidean.base.u8bin \
    --index_path_prefix ${data_dir}/yfcc/yfcc_R32_L90_filtered_index \
    -R 32 --FilteredLbuild 90 --alpha 1.2 \
    --label_file ${data_dir}/yfcc/yfcc.filter.base.txt --universal_label 0

build/apps/build_stitched_index --data_type uint8 \
    --data_path ${data_dir}/yfcc/yfcc.192D.10M.euclidean.base.u8bin \
    --index_path_prefix ${data_dir}/yfcc/yfcc_R32_L100_SR64_stitched_index \
    -R 32 -L 100 --stitched_R 64 --alpha 1.2 \
    --label_file ${data_dir}/yfcc/yfcc.filter.base.txt --universal_label 0

for l in "${yfcc_label_set[@]}"; do
    build/apps/search_memory_index --data_type uint8 --dist_fn l2 \
        --index_path_prefix ${data_dir}/yfcc/yfcc_R32_L90_filtered_index \
        --query_file ${data_dir}/yfcc/yfcc.192D.100K.euclidean.query.u8bin \
        --gt_file ${data_dir}/yfcc/yfcc.gt.${l}.bin --filter_label ${l} \
        -K 10 -L 10 15 20 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 370 390 410 430 450 470 490 510 530 550 570 590 610 630 650 \
        --result_path ${data_dir}/yfcc/filtered_search_results_${l}

    build/apps/search_memory_index --data_type uint8 --dist_fn l2 \
        --index_path_prefix ${data_dir}/yfcc/yfcc_R32_L100_SR64_stitched_index \
        --query_file ${data_dir}/yfcc/yfcc.192D.100K.euclidean.query.u8bin \
        --gt_file ${data_dir}/yfcc/yfcc.gt.${l}.bin --filter_label ${l} \
        -K 10 -L 10 15 20 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 370 390 410 430 450 470 490 510 530 550 570 590 610 630 650 \
        --result_path ${data_dir}/yfcc/stitched_search_results_${l}
done
