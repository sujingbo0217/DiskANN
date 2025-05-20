#!/bin/bash

export R=96         # filtered R
export FilteredLbuild=100
export r=32         # stitched small R
export SR=64        # stitched large R
export L=100
export alpha=1.2
export K=10

export data_dir="/data/jsu068"
# 10 15 20 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 370 390 410 430 450 470 490 510 530 550 570 590 610 630 650

### Synthetic filter label sets
synthetic_labels=(3 10 15 20 37 50)

## bigann
for l in "${synthetic_labels[@]}"; do
    echo "Running compute_groundtruth_for_filters with filter_label: ${l}"
    build/apps/utils/compute_groundtruth_for_filters --data_type uint8 --dist_fn l2 \
        --base_file ${data_dir}/bigann/bigann.128D.10M.euclidean.base.u8bin \
        --query_file ${data_dir}/bigann/bigann.128D.10K.euclidean.query.u8bin \
        --gt_file ${data_dir}/bigann/diskann/filter/gt.diskann.L${l}.bin \
        --label_file ${data_dir}/bigann/bigann.10M.L50.zipf0.75.base.txt \
        --K ${K} --filter_label ${l} --universal_label -1
done

build/apps/build_memory_index --data_type uint8 --dist_fn l2 \
   --data_path ${data_dir}/bigann/bigann.128D.10M.euclidean.base.u8bin \
   --index_path_prefix ${data_dir}/bigann/diskann/filter/bigann_R${R}_L${FilteredLbuild}_filtered_index \
   -R ${R} --FilteredLbuild ${FilteredLbuild} --alpha ${alpha} \
   --label_file ${data_dir}/bigann/bigann.10M.L50.zipf0.75.base.txt --universal_label -1

build/apps/build_stitched_index --data_type uint8 \
    --data_path ${data_dir}/bigann/bigann.128D.10M.euclidean.base.u8bin \
    --index_path_prefix ${data_dir}/bigann/diskann/filter/bigann_R${r}_L${L}_SR${SR}_stitched_index \
    -R ${r} -L ${L} --stitched_R ${SR} --alpha ${alpha} \
    --label_file ${data_dir}/bigann/bigann.10M.L50.zipf0.75.base.txt --universal_label -1

for l in "${synthetic_labels[@]}"; do
    build/apps/search_memory_index --data_type uint8 --dist_fn l2 \
        --index_path_prefix ${data_dir}/bigann/diskann/filter/bigann_R${R}_L${FilteredLbuild}_filtered_index \
        --query_file ${data_dir}/bigann/bigann.128D.10K.euclidean.query.u8bin \
        --gt_file ${data_dir}/bigann/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/bigann/diskann/filter/filtered_search_results_${l}

    build/apps/search_memory_index --data_type uint8 --dist_fn l2 \
        --index_path_prefix ${data_dir}/bigann/diskann/filter/bigann_R${r}_L${L}_SR${SR}_stitched_index \
        --query_file ${data_dir}/bigann/bigann.128D.10K.euclidean.query.u8bin \
        --gt_file ${data_dir}/bigann/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/bigann/diskann/filter/stitched_search_results_${l}
done

## deep
for l in "${synthetic_labels[@]}"; do
    echo "Running compute_groundtruth_for_filters with filter_label: ${l}"
    build/apps/utils/compute_groundtruth_for_filters --data_type float --dist_fn mips \
        --base_file ${data_dir}/deep/deep.96D.10M.angular.base.fbin \
        --query_file ${data_dir}/deep/deep.96D.10K.angular.query.fbin \
        --gt_file ${data_dir}/deep/diskann/filter/gt.diskann.L${l}.bin \
        --label_file ${data_dir}/deep/bigann.10M.L50.zipf0.75.base.txt \
        --K ${K} --filter_label ${l} --universal_label -1
done

build/apps/build_memory_index --data_type float --dist_fn mips \
    --data_path ${data_dir}/deep/deep.96D.10M.angular.base.fbin \
    --index_path_prefix ${data_dir}/deep/diskann/filter/deep_R${R}_L${FilteredLbuild}_filtered_index \
    -R ${R} --FilteredLbuild ${FilteredLbuild} --alpha ${alpha} \
    --label_file ${data_dir}/deep/bigann.10M.L50.zipf0.75.base.txt --universal_label -1

build/apps/build_stitched_index --data_type float \
    --data_path ${data_dir}/deep/deep.96D.10M.angular.base.fbin \
    --index_path_prefix ${data_dir}/deep/diskann/filter/deep_R${r}_L${L}_SR${SR}_stitched_index \
    -R ${r} -L ${L} --stitched_R ${SR} --alpha ${alpha} \
    --label_file ${data_dir}/deep/bigann.10M.L50.zipf0.75.base.txt --universal_label -1

for l in "${synthetic_labels[@]}"; do
    build/apps/search_memory_index --data_type float --dist_fn mips \
        --index_path_prefix ${data_dir}/deep/diskann/filter/deep_R${R}_L${FilteredLbuild}_filtered_index \
        --query_file ${data_dir}/deep/deep.96D.10K.angular.query.fbin \
        --gt_file ${data_dir}/deep/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/deep/diskann/filter/filtered_search_results_${l}

    build/apps/search_memory_index --data_type float --dist_fn mips \
        --index_path_prefix ${data_dir}/deep/diskann/filter/deep_R${r}_L${L}_SR${SR}_stitched_index \
        --query_file ${data_dir}/deep/deep.96D.10K.angular.query.fbin \
        --gt_file ${data_dir}/deep/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/deep/diskann/filter/stitched_search_results_${l}
done

## cohere
for l in "${synthetic_labels[@]}"; do
    echo "Running compute_groundtruth_for_filters with filter_label: ${l}"
    build/apps/utils/compute_groundtruth_for_filters --data_type float --dist_fn mips \
        --base_file ${data_dir}/cohere/cohere.768D.10M.angular.base.fbin \
        --query_file ${data_dir}/cohere/cohere.768D.10M.angular.query.fbin \
        --gt_file ${data_dir}/cohere/diskann/filter/gt.diskann.L${l}.bin \
        --label_file ${data_dir}/cohere/cohere.10M.L50.zipf0.75.txt \
        --K ${K} --filter_label ${l} --universal_label -1
done

build/apps/build_memory_index --data_type float --dist_fn mips \
    --data_path ${data_dir}/cohere/cohere.768D.10M.angular.base.fbin \
    --index_path_prefix ${data_dir}/cohere/diskann/filter/cohere_R${R}_L${FilteredLbuild}_filtered_index \
    -R ${R} --FilteredLbuild ${FilteredLbuild} --alpha ${alpha} \
    --label_file ${data_dir}/cohere/cohere.10M.L50.zipf0.75.txt --universal_label -1

build/apps/build_stitched_index --data_type float \
    --data_path ${data_dir}/cohere/cohere.768D.10M.angular.base.fbin \
    --index_path_prefix ${data_dir}/cohere/diskann/filter/cohere_R${r}_L${L}_SR${SR}_stitched_index \
    -R ${r} -L ${L} --stitched_R ${SR} --alpha ${alpha} \
    --label_file ${data_dir}/cohere/cohere.10M.L50.zipf0.75.txt --universal_label -1

for l in "${synthetic_labels[@]}"; do
    build/apps/search_memory_index --data_type float --dist_fn mips \
        --index_path_prefix ${data_dir}/cohere/diskann/filter/cohere_R${R}_L${FilteredLbuild}_filtered_index \
        --query_file ${data_dir}/cohere/cohere.768D.10M.angular.query.fbin \
        --gt_file ${data_dir}/cohere/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/cohere/diskann/filter/filtered_search_results_${l}

    build/apps/search_memory_index --data_type float --dist_fn mips \
        --index_path_prefix ${data_dir}/cohere/diskann/filter/cohere_R${r}_L${L}_SR${SR}_stitched_index \
        --query_file ${data_dir}/cohere/cohere.768D.10M.angular.query.fbin \
        --gt_file ${data_dir}/cohere/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/cohere/diskann/filter/stitched_search_results_${l}
done

## openai
for l in "${synthetic_labels[@]}"; do
    echo "Running compute_groundtruth_for_filters with filter_label: ${l}"
    build/apps/utils/compute_groundtruth_for_filters --data_type float --dist_fn mips \
        --base_file ${data_dir}/openai/openai.1536D.5M.angular.base.fbin \
        --query_file ${data_dir}/openai/openai.1536D.5M.angular.query.fbin \
        --gt_file ${data_dir}/openai/diskann/filter/gt.diskann.L${l}.bin \
        --label_file ${data_dir}/openai/openai.5M.L50.zipf0.75.txt \
        --K ${K} --filter_label ${l} --universal_label -1
done

build/apps/build_memory_index --data_type float --dist_fn mips \
    --data_path ${data_dir}/openai/openai.1536D.5M.angular.base.fbin \
    --index_path_prefix ${data_dir}/openai/diskann/filter/openai_R${R}_L${FilteredLbuild}_filtered_index \
    -R ${R} --FilteredLbuild ${FilteredLbuild} --alpha ${alpha} \
    --label_file ${data_dir}/openai/openai.5M.L50.zipf0.75.txt --universal_label -1

build/apps/build_stitched_index --data_type float \
    --data_path ${data_dir}/openai/openai.1536D.5M.angular.base.fbin \
    --index_path_prefix ${data_dir}/openai/diskann/filter/openai_R${r}_L${L}_SR${SR}_stitched_index \
    -R ${r} -L ${l} --stitched_R ${SR} --alpha ${alpha} \
    --label_file ${data_dir}/openai/openai.5M.L50.zipf0.75.txt --universal_label -1

for l in "${synthetic_labels[@]}"; do
    build/apps/search_memory_index --data_type float --dist_fn mips \
        --index_path_prefix ${data_dir}/openai/diskann/filter/openai_R${R}_L${FilteredLbuild}_filtered_index \
        --query_file ${data_dir}/openai/openai.1536D.5M.angular.query.fbin \
        --gt_file ${data_dir}/openai/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/openai/diskann/filter/filtered_search_results_${l}

    build/apps/search_memory_index --data_type float --dist_fn mips \
        --index_path_prefix ${data_dir}/openai/diskann/filter/openai_R${r}_L${L}_SR${SR}_stitched_index \
        --query_file ${data_dir}/openai/openai.1536D.5M.angular.query.fbin \
        --gt_file ${data_dir}/openai/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/openai/diskann/filter/stitched_search_results_${l}
done

### Natural filter label sets
marco_label_set=(48 49 50 51 52)
yfcc_label_set=(23 29 89 20 1589)

## marco
for l in "${marco_label_set[@]}"; do
    build/apps/utils/compute_groundtruth_for_filters --data_type float --dist_fn l2 \
        --base_file ${data_dir}/marco/embedding/marco.768D.10M.euclidean.fbin \
        --query_file ${data_dir}/marco/query/marco.768D.10K.euclidean.fbin \
        --gt_file ${data_dir}/marco/diskann/filter/gt.diskann.L${l}.bin \
        --label_file ${data_dir}/marco/embedding/marco.filter.base.10M.new.txt \
        --K ${K} --filter_label ${l} --universal_label 0
done

build/apps/build_memory_index --data_type float --dist_fn l2 \
    --data_path ${data_dir}/marco/embedding/marco.768D.10M.euclidean.fbin \
    --index_path_prefix ${data_dir}/marco/diskann/filter/marco_R${R}_L${FilteredLbuild}_filtered_index \
    -R ${R} --FilteredLbuild ${FilteredLbuild} --alpha ${alpha} \
    --label_file ${data_dir}/marco/embedding/marco.filter.base.10M.new.txt --universal_label 0

build/apps/build_stitched_index --data_type float \
    --data_path ${data_dir}/marco/embedding/marco.768D.10M.euclidean.fbin \
    --index_path_prefix ${data_dir}/marco/diskann/filter/marco_R${r}_L${L}_SR${SR}_stitched_index \
    -R ${r} -L ${L} --stitched_R ${SR} --alpha ${alpha} \
    --label_file ${data_dir}/marco/embedding/marco.filter.base.10M.new.txt --universal_label 0

for l in "${marco_label_set[@]}"; do
    build/apps/search_memory_index --data_type float --dist_fn l2 \
        --index_path_prefix ${data_dir}/marco/diskann/filter/marco_R${R}_L${FilteredLbuild}_filtered_index \
        --query_file ${data_dir}/marco/query/marco.768D.10K.euclidean.fbin \
        --gt_file ${data_dir}/marco/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/marco/diskann/filter/filtered_search_results_${l}

    build/apps/search_memory_index --data_type float --dist_fn l2 \
        --index_path_prefix ${data_dir}/marco/diskann/filter/marco_R${r}_L${L}_SR${SR}_stitched_index \
        --query_file ${data_dir}/marco/query/marco.768D.10K.euclidean.fbin \
        --gt_file ${data_dir}/marco/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/marco/diskann/filter/stitched_search_results_${l}
done

## yfcc (unlimited runtime, TLE)
for l in "${yfcc_label_set[@]}"; do
    echo "Running compute_groundtruth_for_filters with filter_label: ${l}"
    build/apps/utils/compute_groundtruth_for_filters --data_type uint8 --dist_fn l2 \
        --base_file ${data_dir}/yfcc/yfcc.192D.10M.euclidean.base.u8bin \
        --query_file ${data_dir}/yfcc/yfcc.192D.100K.euclidean.query.u8bin \
        --gt_file ${data_dir}/yfcc/diskann/filter/gt.diskann.L${l}.bin \
        --label_file ${data_dir}/yfcc/yfcc.filter.base.txt \
        --K ${K} --filter_label ${l}
done

build/apps/build_memory_index --data_type uint8 --dist_fn l2 \
    --data_path ${data_dir}/yfcc/yfcc.192D.10M.euclidean.base.u8bin \
    --index_path_prefix ${data_dir}/yfcc/diskann/filter/yfcc_R${R}_L${FilteredLbuild}_filtered_index \
    -R ${R} --FilteredLbuild ${FilteredLbuild} --alpha ${alpha} \
    --label_file ${data_dir}/yfcc/yfcc.filter.base.txt

build/apps/build_stitched_index --data_type uint8 \
    --data_path ${data_dir}/yfcc/yfcc.192D.10M.euclidean.base.u8bin \
    --index_path_prefix ${data_dir}/yfcc/diskann/filter/yfcc_R${r}_L${L}_SR${SR}_stitched_index \
    -R ${r} -L ${L} --stitched_R ${SR} --alpha ${alpha} \
    --label_file ${data_dir}/yfcc/yfcc.filter.base.txt

for l in "${yfcc_label_set[@]}"; do
    build/apps/search_memory_index --data_type uint8 --dist_fn l2 \
        --index_path_prefix ${data_dir}/yfcc/yfcc_R${R}_L${FilteredLbuild}_filtered_index \
        --query_file ${data_dir}/yfcc/yfcc.192D.100K.euclidean.query.u8bin \
        --gt_file ${data_dir}/yfcc/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/diskann/filter/yfcc/filtered_search_results_${l}

    build/apps/search_memory_index --data_type uint8 --dist_fn l2 \
        --index_path_prefix ${data_dir}/yfcc/yfcc_R${r}_L${L}_SR${SR}_stitched_index \
        --query_file ${data_dir}/yfcc/yfcc.192D.100K.euclidean.query.u8bin \
        --gt_file ${data_dir}/yfcc/diskann/filter/gt.diskann.L${l}.bin --filter_label ${l} \
        -K ${K} -L 10 15 20 50 110 210 410 610 \
        --result_path ${data_dir}/diskann/filter/yfcc/stitched_search_results_${l}
done
