#!/bin/bash

export R=32
export L=75
export alpha=1.2
export K=10

export op_dir="/data/jsu068"
export data_dir="/data/zshen055/ANN"

export Large=100000000
export Middle=10000000
export Small=5000000

del_pctg=(10 20 30 40 50 60 70 80 90)

# # bigann-10M test
# build/apps/test_insert_deletes_consolidate --data_type float --dist_fn l2 \
#     --data_path /data/zshen055/ANN/BIGANN/base.10M.fbin \
#     --index_path_prefix /data/jsu068/bigann/diskann/dynamic/index/bigann_R32_L75_dynamic_index_test \
#     -R 32 -L 75 --alpha 1.3 \
#     --points_to_skip 0 --max_points_to_insert 10000000 --beginning_index_size 0 \
#     --points_per_checkpoint 10000000 --checkpoints_per_snapshot 0 \
#     --points_to_delete_from_beginning 2000000 --start_deletes_after 10000000 \
#     --start_point_norm 508 --num_start_points 0 --do_concurrent false

# echo ">>> Generating ground truth files..."
# build/apps/utils/compute_groundtruth --data_type float --dist_fn l2 \
#     --base_file /data/jsu068/bigann/base.10M.fbin --query_file /data/jsu068/bigann/query.10K.fbin \
#     --K 100 --gt_file /data/jsu068/bigann/gt-dynamic-10M-K100.after-delete-1000000 \
#     --tags_file /data/jsu068/bigann/diskann/dynamic/index/bigann_R32_L75_dynamic_index_test.after-delete-1000000-10000000.tags

# echo ">>> Searching on consolidated index..."
# build/apps/search_memory_index --data_type float --dist_fn l2 \
#     --index_path_prefix /data/jsu068/bigann/diskann/dynamic/index/bigann_R32_L75_dynamic_index_test.after-delete-2000000-10000000 \
#     --result_path /data/jsu068/bigann/diskann/dynamic/results/bigann_R32_L75_dynamic_index_test.delete-2000000 \
#     --query_file /data/zshen055/ANN/BIGANN/query.10K.fbin --gt_file /data/jsu068/bigann/gt-dynamic-10M-last_8000000.ibin \
#     --dynamic true --tags 1 -K 10 -L 10 20 50 100 200 300 400 500

# echo ">>> Searching on rebuilt index..."
# build/apps/search_memory_index --data_type float --dist_fn l2 \
#     --index_path_prefix /data/jsu068/bigann/diskann/dynamic/index/rebuild_bigann_R32_L75_dynamic_index_test.last-8000000 \
#     --result_path /data/jsu068/bigann/diskann/dynamic/results/rebuild_bigann_R32_L75_dynamic_index_test.last-8000000 \
#     --query_file /data/zshen055/ANN/BIGANN/query.10K.fbin --gt_file /data/jsu068/bigann/gt-dynamic-10M-last_8000000.ibin \
#     --dynamic true --tags 1 -K 10 -L 10 20 50 100 200 300 400 500

# # bigann-10M delete-test
# build/apps/delete-test --data_type float --dist_fn l2 \
#     --data_path /data/zshen055/ANN/BIGANN/base.10M.fbin \
#     --index_path_prefix /data/jsu068/bigann/diskann/dynamic/index/bigann_R32_L75_dynamic_index_test \
#     -R 32 --Lbuild 75 --alpha 1.2 \
#     --points_to_skip 0 --max_points_to_insert 10000000 --beginning_index_size 0 \
#     --points_per_checkpoint 10000000 --checkpoints_per_snapshot 0 \
#     --points_to_delete_from_beginning 8000000 --start_deletes_after 10000000 \
#     --start_point_norm 508 --num_start_points 0 --do_concurrent false \
#     --query_file /data/zshen055/ANN/BIGANN/query.10K.fbin --gt_file /data/zshen055/ANN/BIGANN/gt/last_8000000.ibin \
#     --dynamic true --tags 1 -K 10 --search_list 10 20 50 100 200 300 400 500

# # openai-5M delete-test
# build/apps/delete-test --data_type float --dist_fn l2 \
#     --data_path /data/zshen055/ANN/openai/openai_large_5m/base.fbin \
#     --index_path_prefix /data/jsu068/openai/diskann/dynamic/index/openai_R32_L75_dynamic_index_test \
#     -R 32 --Lbuild 75 --alpha 1.2 \
#     --points_to_skip 0 --max_points_to_insert 5000000 --beginning_index_size 0 \
#     --points_per_checkpoint 5000000 --checkpoints_per_snapshot 0 \
#     --points_to_delete_from_beginning 2000000 --start_deletes_after 5000000 \
#     --start_point_norm 1 --num_start_points 0 --do_concurrent false \
#     --query_file /data/zshen055/ANN/openai/openai_large_5m/query.fbin \
#     --gt_file /data/zshen055/ANN/openai/openai_large_5m/gt/last_2000000.ibin \
#     --dynamic true --tags 1 -K 10 --search_list 10 20 50 100 200 300 400 500


### bigann-100M ----------------------------------------------------------------
# avg norm (l2): 508.6453
for d in "${del_pctg[@]}"; do
    del_cnt=$((Large / 100 * d))
    last_npts=$((Large - del_cnt))

    # echo ">>> Marking, consolidating, and rebuilding..."
    build/apps/delete-test --data_type float --dist_fn l2 \
        --data_path ${data_dir}/BIGANN/base.100M.fbin \
        --index_path_prefix ${op_dir}/bigann/diskann/dynamic/index/bigann_R${R}_L${L}_dynamic_index \
        -R ${R} --Lbuild ${L} --alpha ${alpha} \
        --points_to_skip 0 --max_points_to_insert ${Large} --beginning_index_size 0 \
        --points_per_checkpoint ${Large} --checkpoints_per_snapshot 0 \
        --points_to_delete_from_beginning ${del_cnt} --start_deletes_after ${Large} \
        --start_point_norm 508 --num_start_points 0 --do_concurrent false \
        --query_file ${data_dir}/BIGANN/query.10K.fbin \
        --gt_file ${data_dir}/BIGANN/gt/100M/last_${del_cnt}.ibin \
        --dynamic true --tags 1 -K 10 --search_list 10 20 50 100 200 300 400 500

    # echo ">>> Searching on consolidated index..."
    # build/apps/search_memory_index --data_type float --dist_fn l2 \
    #     --index_path_prefix ${op_dir}/bigann/diskann/dynamic/index/bigann_R${R}_L${L}_dynamic_index.after-delete-${del_cnt}-${Large} \
    #     --result_path ${op_dir}/bigann/diskann/dynamic/results/bigann_R${R}_L${L}_dynamic_index.delete-${del_cnt} \
    #     --query_file ${data_dir}/BIGANN/query.10K.fbin --gt_file ${op_dir}/bigann/gt-dynamic-100M-last_${last_npts}.ibin \
    #     --dynamic true --tags 1 -K 10 -L 10 20 50 100 200 300 400 500

    # echo ">>> Searching on rebuilt index..."
    # build/apps/search_memory_index --data_type float --dist_fn l2 \
    #     --index_path_prefix ${op_dir}/bigann/diskann/dynamic/index/rebuild_bigann_R${R}_L${L}_dynamic_index.last-${last_npts} \
    #     --result_path ${op_dir}/bigann/diskann/dynamic/results/rebuild_bigann_R${R}_L${L}_dynamic_index.delete-${del_cnt} \
    #     --query_file ${data_dir}/BIGANN/query.10K.fbin --gt_file ${op_dir}/bigann/gt-dynamic-100M-last_${last_npts}.ibin \
    #     --dynamic true --tags 1 -K 10 -L 10 20 50 100 200 300 400 500
done
#   ----------------------------------------------------------------------------


### deep-100M   ----------------------------------------------------------------
# avg norm (angular) = 1
# avg norm (l2) = 1
for d in "${del_pctg[@]}"; do
    del_cnt=$((Large / 100 * d))
    last_npts=$((Large - del_cnt))

    # echo ">>> Marking, consolidating, and rebuilding..."
    build/apps/delete-test --data_type float --dist_fn cosine \
        --data_path ${data_dir}/Yandex-DEEP/base.1B.fbin \
        --index_path_prefix ${op_dir}/deep/diskann/dynamic/index/deep_R${R}_L${L}_dynamic_index \
        -R ${R} --Lbuild ${L} --alpha ${alpha} \
        --points_to_skip 0 --max_points_to_insert ${Large} --beginning_index_size 0 \
        --points_per_checkpoint ${Large} --checkpoints_per_snapshot 0 \
        --points_to_delete_from_beginning ${del_cnt} --start_deletes_after ${Large} \
        --start_point_norm 1 --num_start_points 0 --do_concurrent false \
        --query_file ${data_dir}/Yandex-DEEP/query.public.10K.fbin \
        --gt_file ${data_dir}/Yandex-DEEP/gt/100M/last_${del_cnt}.ibin \
        --dynamic true --tags 1 -K 10 --search_list 10 20 50 100 200 300 400 500

    # echo ">>> Searching on consolidated index..."
    # build/apps/search_memory_index --data_type float --dist_fn cosine \
    #     --index_path_prefix ${op_dir}/deep/diskann/dynamic/index/deep_R${R}_L${L}_dynamic_index.after-delete-${del_cnt}-${Large} \
    #     --result_path ${op_dir}/deep/diskann/dynamic/results/deep_R${R}_L${L}_dynamic_index.delete-${del_cnt} \
    #     --query_file ${data_dir}/Yandex-DEEP/query.public.10K.fbin --gt_file ${op_dir}/deep/gt-dynamic-100M-last_${last_npts}.ibin \
    #     --dynamic true --tags 1 -K 10 -L 10 20 50 100 200 300 400 500

    # echo ">>> Searching on rebuilt index..."
    # build/apps/search_memory_index --data_type float --dist_fn cosine \
    #     --index_path_prefix ${op_dir}/deep/diskann/dynamic/index/rebuild_deep_R${R}_L${L}_dynamic_index.last-${last_npts} \
    #     --result_path ${op_dir}/deep/diskann/dynamic/results/rebuild_deep_R${R}_L${L}_dynamic_index.delete-${del_cnt} \
    #     --query_file ${data_dir}/Yandex-DEEP/query.public.10K.fbin --gt_file ${op_dir}/deep/gt-dynamic-100M-last_${last_npts}.ibin \
    #     --dynamic true --tags 1 -K 10 -L 10 20 50 100 200 300 400 500
done
#   ----------------------------------------------------------------------------


# cohere-10M    ----------------------------------------------------------------
# avg norm (angular) = 1
# avg norm (l2) = 13.7555
for d in "${del_pctg[@]}"; do
    del_cnt=$((Middle / 100 * d))
    last_npts=$((Middle - del_cnt))

    # echo ">>> Marking, consolidating, and rebuilding..."
    build/apps/delete-test --data_type float --dist_fn cosine \
        --data_path ${data_dir}/cohere/cohere_large_10m/base.fbin \
        --index_path_prefix ${op_dir}/cohere/diskann/dynamic/index/cohere_R${R}_L${L}_dynamic_index \
        -R ${R} --Lbuild ${L} --alpha ${alpha} \
        --points_to_skip 0 --max_points_to_insert ${Middle} --beginning_index_size 0 \
        --points_per_checkpoint ${Middle} --checkpoints_per_snapshot 0 \
        --points_to_delete_from_beginning ${del_cnt} --start_deletes_after ${Middle} \
        --start_point_norm 13 --num_start_points 0 --do_concurrent false \
        --query_file ${data_dir}/cohere/cohere_large_10m/query.fbin \
        --gt_file ${data_dir}/cohere/cohere_large_10m/gt/last_${del_cnt}.ibin \
        --dynamic true --tags 1 -K 10 --search_list 10 20 50 100 200 300 400 500

    # echo ">>> Searching on consolidated index..."
    # build/apps/search_memory_index --data_type float --dist_fn cosine \
    #     --index_path_prefix ${op_dir}/cohere/diskann/dynamic/index/cohere_R${R}_L${L}_dynamic_index.after-delete-${del_cnt}-${Middle} \
    #     --result_path ${op_dir}/cohere/diskann/dynamic/results/cohere_R${R}_L${L}_dynamic_index.delete-${del_cnt} \
    #     --query_file ${data_dir}/cohere/cohere_large_10m/query.fbin --gt_file ${op_dir}/cohere/gt-dynamic-10M-last_${last_npts}.ibin \
    #     --dynamic true --tags 1 -K 10 -L 10 20 50 100 200 300 400 500

    # echo ">>> Searching on rebuilt index..."
    # build/apps/search_memory_index --data_type float --dist_fn cosine \
    #     --index_path_prefix ${op_dir}/cohere/diskann/dynamic/index/rebuild_cohere_R${R}_L${L}_dynamic_index.last-${last_npts} \
    #     --result_path ${op_dir}/cohere/diskann/dynamic/results/rebuild_cohere_R${R}_L${L}_dynamic_index.delete-${del_cnt} \
    #     --query_file ${data_dir}/cohere/cohere_large_10m/query.fbin --gt_file ${op_dir}/cohere/gt-dynamic-10M-last_${last_npts}.ibin \
    #     --dynamic true --tags 1 -K 10 -L 10 20 50 100 200 300 400 500
done
#   ------------------------------------------------------------------------


# openai-5M ----------------------------------------------------------------
# avg norm (angular) = 1
# avg norm (l2) = 1
for d in "${del_pctg[@]}"; do
    del_cnt=$((Small / 100 * d))
    last_npts=$((Small - del_cnt))

    # echo ">>> Marking, consolidating, and rebuilding..."
    build/apps/delete-test --data_type float --dist_fn cosine \
        --data_path ${data_dir}/openai/openai_large_5m/base.fbin \
        --index_path_prefix ${op_dir}/openai/diskann/dynamic/index/openai_R${R}_L${L}_dynamic_index \
        -R ${R} --Lbuild ${L} --alpha ${alpha} \
        --points_to_skip 0 --max_points_to_insert ${Small} --beginning_index_size 0 \
        --points_per_checkpoint ${Small} --checkpoints_per_snapshot 0 \
        --points_to_delete_from_beginning ${del_cnt} --start_deletes_after ${Small} \
        --start_point_norm 1 --num_start_points 0 --do_concurrent false \
        --query_file ${data_dir}/openai/openai_large_5m/query.fbin \
        --gt_file ${data_dir}/openai/openai_large_5m/gt/last_${del_cnt}.ibin \
        --dynamic true --tags 1 -K 10 --search_list 10 20 50 100 200 300 400 500

    # echo ">>> Searching on consolidated index..."
    # build/apps/search_memory_index --data_type float --dist_fn cosine \
    #     --index_path_prefix ${op_dir}/openai/diskann/dynamic/index/openai_R${R}_L${L}_dynamic_index.after-delete-${del_cnt}-${Small} \
    #     --result_path ${op_dir}/openai/diskann/dynamic/results/openai_R${R}_L${L}_dynamic_index.delete-${del_cnt} \
    #     --query_file ${data_dir}/openai/openai_large_5m/query.fbin --gt_file ${data_dir}/openai/openai_large_5m/gt/last_${last_npts}.ibin \
    #     --dynamic true --tags 1 -K 10 -L 10 20 50 100 200 300 400 500

    # echo ">>> Searching on rebuilt index..."
    # build/apps/search_memory_index --data_type float --dist_fn cosine \
    #     --index_path_prefix ${op_dir}/openai/diskann/dynamic/index/rebuild_openai_R${R}_L${L}_dynamic_index.last-${last_npts} \
    #     --result_path ${op_dir}/openai/diskann/dynamic/results/rebuild_openai_R${R}_L${L}_dynamic_index.delete-${del_cnt} \
    #     --query_file ${data_dir}/openai/openai_large_5m/query.fbin --gt_file ${op_dir}/openai/gt-dynamic-5M-last_${last_npts}.ibin \
    #     --dynamic true --tags 1 -K 10 -L 10 20 50 100 200 300 400 500
done
#   ------------------------------------------------------------------------
