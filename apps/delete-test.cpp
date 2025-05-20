// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <set>

#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <future>

#include "utils.h"
#include "filter_utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

namespace po = boost::program_options;


template <typename T, typename LabelT = uint32_t, class Idx>
int search_memory_index(Idx &index, diskann::Metric metric, /*const std::string &result_path_prefix,*/
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const std::vector<std::string> &query_filters, const float fail_if_recall_below)
{
    using TagT = uint32_t;
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool calc_recall_flag = false;
    if (truthset_file != std::string("null") && file_exists(truthset_file))
    {
        diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }
    else
    {
        diskann::cout << " Truthset file " << truthset_file << " not found. Not computing recall." << std::endl;
    }

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    // const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);
    const size_t num_frozen_pts = 0;
    if (metric == diskann::FAST_L2)
        index.optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    if (tags)
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(20) << "Mean Latency (mus)"
                  << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 20 + 15;
    }
    else
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 18 + 20 + 15;
    }
    uint32_t recalls_to_print = 0;
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;
    if (calc_recall_flag)
    {
        for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
        {
            std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
        }
        recalls_to_print = recall_at + 1 - first_recall;
        table_width += recalls_to_print * 12;
    }
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats;
    if (not tags || filtered_search)
    {
        cmp_stats = std::vector<uint32_t>(query_num, 0);
    }

    std::vector<TagT> query_result_tags;
    if (tags)
    {
        query_result_tags.resize(recall_at * query_num);
    }

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            auto qs = std::chrono::high_resolution_clock::now();
            if (filtered_search && !tags)
            {
                std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                auto retval = index.search_with_filters(query + i * query_aligned_dim, raw_filter, recall_at, L,
                                                         query_result_ids[test_id].data() + i * recall_at,
                                                         query_result_dists[test_id].data() + i * recall_at);
                cmp_stats[i] = retval.second;
            }
            else if (metric == diskann::FAST_L2)
            {
                index.search_with_optimized_layout(query + i * query_aligned_dim, recall_at, L,
                                                    query_result_ids[test_id].data() + i * recall_at);
            }
            else if (tags)
            {
                if (!filtered_search)
                {
                    index.search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res);
                }
                else
                {
                    std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                    index.search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res, true, raw_filter);
                }

                for (int64_t r = 0; r < (int64_t)recall_at; r++)
                {
                    query_result_ids[test_id][recall_at * i + r] = query_result_tags[recall_at * i + r];
                }
            }
            else
            {
                cmp_stats[i] = index.search(query + i * query_aligned_dim, recall_at, L,
                                            query_result_ids[test_id].data() + i * recall_at).second;
            }
            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        double displayed_qps = query_num / diff.count();

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::vector<double> recalls;
        if (calc_recall_flag)
        {
            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
            }
        }

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float)query_num;

        if (tags && !filtered_search)
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float)mean_latency
                      << std::setw(15) << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        else
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
                      << std::setw(20) << (float)mean_latency << std::setw(15)
                      << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        for (double recall : recalls)
        {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(recall, best_recall);
        }
        std::cout << std::endl;
    }
/*
    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }
        std::string cur_result_path_prefix = result_path_prefix + "_" + std::to_string(L);

        std::string cur_result_path = cur_result_path_prefix + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = cur_result_path_prefix + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id].data(), query_num, recall_at);

        test_id++;
    }
*/
    diskann::aligned_free(query);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

// load_aligned_bin modified to read pieces of the file, but using ifstream
// instead of cached_ifstream.
template <typename T>
inline void load_aligned_bin_part(const std::string &bin_file, T *data, size_t offset_points, size_t points_to_read)
{
    diskann::Timer timer;
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(bin_file, std::ios::binary | std::ios::ate);
    size_t actual_file_size = reader.tellg();
    reader.seekg(0, std::ios::beg);

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    size_t npts = (uint32_t)npts_i32;
    size_t dim = (uint32_t)dim_i32;

    size_t expected_actual_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size)
    {
        std::stringstream stream;
        stream << "Error. File size mismatch. Actual size is " << actual_file_size << " while expected size is  "
               << expected_actual_file_size << " npts = " << npts << " dim = " << dim << " size of <T>= " << sizeof(T)
               << std::endl;
        std::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (offset_points + points_to_read > npts)
    {
        std::stringstream stream;
        stream << "Error. Not enough points in file. Requested " << offset_points << "  offset and " << points_to_read
               << " points, but have only " << npts << " points" << std::endl;
        std::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    reader.seekg(2 * sizeof(uint32_t) + offset_points * dim * sizeof(T));

    const size_t rounded_dim = ROUND_UP(dim, 8);

    for (size_t i = 0; i < points_to_read; i++)
    {
        reader.read((char *)(data + i * rounded_dim), dim * sizeof(T));
        memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    reader.close();

    const double elapsedSeconds = timer.elapsed() / 1000000.0;
    std::cout << "Read " << points_to_read << " points using non-cached reads in " << elapsedSeconds << std::endl;
}

std::string get_save_filename(const std::string &save_path, size_t points_to_skip, size_t points_deleted,
                              size_t last_point_threshold)
{
    std::string final_path = save_path;
    if (points_to_skip > 0)
    {
        final_path += "skip" + std::to_string(points_to_skip) + "-";
    }

    final_path += std::to_string(points_deleted) + "-";
    final_path += std::to_string(last_point_threshold);
    return final_path;
}

template <typename T, typename TagT, typename LabelT>
void insert_till_next_checkpoint(diskann::AbstractIndex &index, size_t start, size_t end, int32_t thread_count, T *data,
                                 size_t aligned_dim, std::vector<std::vector<LabelT>> &location_to_labels)
{
    diskann::Timer insert_timer;
#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
    for (int64_t j = start; j < (int64_t)end; j++)
    {
        if (!location_to_labels.empty())
        {
            index.insert_point(&data[(j - start) * aligned_dim], 1 + static_cast<TagT>(j),
                               location_to_labels[j - start]);
        }
        else
        {
            // index.insert_point(&data[(j - start) * aligned_dim], 1 + static_cast<TagT>(j));
            index.insert_point(&data[(j - start) * aligned_dim], static_cast<TagT>(j));
        }
    }
    const double elapsedSeconds = insert_timer.elapsed() / 1000000.0;
    diskann::cout << "Insertion time " << elapsedSeconds << " seconds (" << (end - start) / elapsedSeconds
              << " points/second overall, " << (end - start) / elapsedSeconds / thread_count << " per thread)\n ";
}

bool index_exists(const std::string& file_path) {
    std::ifstream f(file_path);
    return f.good();
}

template <typename T, typename TagT, typename L, class Idx>
void delete_from_beginning(Idx &index, diskann::IndexWriteParameters &delete_params,
                           size_t points_to_skip, size_t points_to_delete_from_beginning, L run_search)
{
    try
    {
        diskann::cout << std::endl
                  << "Lazy deleting points " << points_to_skip << " to "
                  << points_to_skip + points_to_delete_from_beginning << "... " << std::endl;
        diskann::Timer timer;
        for (size_t i = 0; i < points_to_delete_from_beginning; ++i) {
            index.lazy_delete(static_cast<TagT>(i)); // Since tags are data location + 1
        }
        const double t_mark = timer.elapsed() / 1000000.0;
        diskann::cout << "### Marking time: " << t_mark << " seconds." << std::endl << std::endl;

        // print graph information
        auto* concrete_index = dynamic_cast<diskann::Index<T, TagT, uint32_t>*>(&index);
        if (concrete_index) concrete_index->print_status();
        
        // search on index after lazy deletion
        run_search(index);

        diskann::cout << std::endl
                  << "Consolidating after lazy deletion from " << points_to_skip << " to "
                  << points_to_skip + points_to_delete_from_beginning << "... " << std::endl;

        auto report = index.consolidate_deletes(delete_params);
        diskann::cout << "### Consolidation time: " << report._time << " seconds." << std::endl << std::endl;
        diskann::cout << "#active points: " << report._active_points << std::endl
                  << "max points: " << report._max_points << std::endl
                  << "empty slots: " << report._empty_slots << std::endl
                  << "deletes processed: " << report._slots_released << std::endl
                  << "latest delete size: " << report._delete_set_size << std::endl
                  << "rate: (" << points_to_delete_from_beginning / report._time << " points/second overall, "
                  << points_to_delete_from_beginning / report._time / delete_params.num_threads << " per thread)"
                  << std::endl << std::endl;

        // print graph information
        concrete_index = dynamic_cast<diskann::Index<T, TagT, uint32_t>*>(&index);
        if (concrete_index) concrete_index->print_status();
        
        // search on index after consolidation
        run_search(index);
    }
    catch (std::system_error &e)
    {
        diskann::cout << "Exception caught in deletion thread: " << e.what() << std::endl;
    }
}

template <typename T, typename L>
void build_incremental_index(const std::string &data_path, diskann::IndexWriteParameters &params, size_t points_to_skip,
                             size_t max_points_to_insert, size_t beginning_index_size, float start_point_norm,
                             uint32_t num_start_pts, size_t points_per_checkpoint, size_t checkpoints_per_snapshot,
                             const std::string &save_path, size_t points_to_delete_from_beginning,
                             size_t start_deletes_after, bool concurrent, const std::string &label_file,
                             const std::string &universal_label, const diskann::Metric &metric, L run_search)
{
    size_t dim, aligned_dim;
    size_t num_points;
    diskann::get_bin_metadata(data_path, num_points, dim);
    aligned_dim = ROUND_UP(dim, 8);
    bool has_labels = label_file != "";
    using TagT = uint32_t;
    using LabelT = uint32_t;

    size_t current_point_offset = points_to_skip;
    const size_t last_point_threshold = points_to_skip + max_points_to_insert;

    bool enable_tags = true;
    using TagT = uint32_t;
    // build index for deletion
    auto index_search_params = diskann::IndexSearchParams(params.search_list_size, params.num_threads);
    diskann::IndexConfig index_config = diskann::IndexConfigBuilder()
                                            .with_metric(metric)
                                            .with_dimension(dim)
                                            .with_max_points(max_points_to_insert)
                                            .is_dynamic_index(true)
                                            .with_index_write_params(params)
                                            .with_index_search_params(index_search_params)
                                            .with_data_type(diskann_type_to_name<T>())
                                            .with_tag_type(diskann_type_to_name<TagT>())
                                            .with_label_type(diskann_type_to_name<LabelT>())
                                            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                                            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                                            .is_enable_tags(enable_tags)
                                            .is_filtered(has_labels)
                                            .with_num_frozen_pts(num_start_pts)
                                            .is_concurrent_consolidate(concurrent)
                                            .build();

    diskann::IndexFactory index_factory = diskann::IndexFactory(index_config);
    auto index = index_factory.create_instance();

    if (universal_label != "")
    {
        LabelT u_label = 0;
        index->set_universal_label(u_label);
    }

    if (points_to_skip > num_points)
    {
        throw diskann::ANNException("Asked to skip more points than in data file", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (max_points_to_insert == 0)
    {
        max_points_to_insert = num_points;
    }

    if (points_to_skip + max_points_to_insert > num_points)
    {
        max_points_to_insert = num_points - points_to_skip;
        std::cerr << "WARNING: Reducing max_points_to_insert to " << max_points_to_insert
                  << " points since the data file has only that many" << std::endl;
    }

    if (beginning_index_size > max_points_to_insert)
    {
        beginning_index_size = max_points_to_insert;
        std::cerr << "WARNING: Reducing beginning index size to " << beginning_index_size
                  << " points since the data file has only that many" << std::endl;
    }
    if (checkpoints_per_snapshot > 0 && beginning_index_size > points_per_checkpoint)
    {
        beginning_index_size = points_per_checkpoint;
        std::cerr << "WARNING: Reducing beginning index size to " << beginning_index_size << std::endl;
    }

    T *data = nullptr;
    diskann::alloc_aligned(
        (void **)&data, std::max(points_per_checkpoint, beginning_index_size) * aligned_dim * sizeof(T), 8 * sizeof(T));

    std::vector<TagT> tags(max_points_to_insert);
    std::iota(tags.begin(), tags.end(), 0);

    load_aligned_bin_part(data_path, data, 0, max_points_to_insert); // 0 to data_size
    std::cout << "load aligned bin succeeded" << std::endl;
    diskann::Timer timer;

    if (beginning_index_size > 0)
    {
        index->build(data, beginning_index_size, tags);
    }
    else
    {
        if (!index_exists(save_path)) {
            index->set_start_points_at_random(static_cast<T>(start_point_norm));
        } else {
            index->load(save_path.c_str(), params.num_threads, 500);
        }
    }

    const double elapsedSeconds = timer.elapsed() / 1000000.0;
    diskann::cout << "Initial non-incremental index build time for " << max_points_to_insert << " points took "
              << elapsedSeconds << " seconds (" << max_points_to_insert / elapsedSeconds << " points/second)\n ";

    current_point_offset += beginning_index_size;

    if (points_to_delete_from_beginning > max_points_to_insert)
    {
        points_to_delete_from_beginning = static_cast<uint32_t>(max_points_to_insert);
        std::cerr << "WARNING: Reducing points to delete from beginning to " << points_to_delete_from_beginning
                  << " points since the data file has only that many" << std::endl;
    }

    std::vector<std::vector<LabelT>> location_to_labels;
    for (size_t start = current_point_offset; !index_exists(save_path) && start < last_point_threshold;
            start += points_per_checkpoint, current_point_offset += points_per_checkpoint)
    {
        const size_t end = std::min(start + points_per_checkpoint, last_point_threshold);
        std::cout << std::endl << "Inserting from " << start << " to " << end << std::endl;

        load_aligned_bin_part(data_path, data, start, end - start);
        insert_till_next_checkpoint<T, TagT, LabelT>(*index, start, end, (int32_t)params.num_threads, data,
                                                        aligned_dim, location_to_labels);
        std::cout << "Number of points in the index post insertion " << end << std::endl;
    }      
    // save original index at the first time
    if (!index_exists(save_path)) {
        diskann::cout << "Saving original dynamic index..." << std::endl;
        index->save(save_path.c_str(), true);
    }

    if (points_to_delete_from_beginning > 0)
    {
        // auto concrete_index = dynamic_cast<diskann::Index<T, TagT, LabelT>*>(index.get());
        // if (concrete_index == nullptr) {
        //     throw diskann::ANNException("Failed to cast AbstractIndex to concrete Index type", -1, __FUNCSIG__, __FILE__, __LINE__);
        // }
        delete_from_beginning<T, TagT, L>(*index, params, points_to_skip, points_to_delete_from_beginning, run_search);
    }

    // index->save(save_path_inc.c_str(), true);
    // return index;

    // rebuild index by remaining nodes
    diskann::cout << std::endl << "Index rebuilding..." << std::endl;

    size_t remaining_size = max_points_to_insert - points_to_delete_from_beginning;
    try {
        auto rebuild_index_config = diskann::IndexConfigBuilder()
                                        .with_metric(metric)
                                        .with_dimension(dim)
                                        .with_max_points(remaining_size)
                                        .is_dynamic_index(true)
                                        .with_index_write_params(params)
                                        .with_index_search_params(index_search_params)
                                        .with_data_type(diskann_type_to_name<T>())
                                        .with_tag_type(diskann_type_to_name<TagT>())
                                        .with_label_type(diskann_type_to_name<LabelT>())
                                        .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                                        .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                                        .is_enable_tags(enable_tags)
                                        .is_filtered(has_labels)
                                        .with_num_frozen_pts(num_start_pts)
                                        .is_concurrent_consolidate(concurrent)
                                        .build();

        auto rebuild_index_factory = diskann::IndexFactory(rebuild_index_config);
        auto rebuild_index = rebuild_index_factory.create_instance();

        // T *remaining_data = nullptr;
        // diskann::alloc_aligned((void **)&remaining_data, remaining_size * aligned_dim * sizeof(T), 8 * sizeof(T));
        std::vector<TagT> remaining_tags(remaining_size);
        std::iota(remaining_tags.begin(), remaining_tags.end(), points_to_delete_from_beginning);
        load_aligned_bin_part(data_path, data, points_to_delete_from_beginning, remaining_size);

        diskann::Timer rebuild_timer;
        
        rebuild_index->build(data, remaining_size, remaining_tags);

        const double elapsedSecondsRebuild = rebuild_timer.elapsed() / 1000000.0;
        diskann::cout << "### Index rebuild time for " << remaining_size << " points took "
                << elapsedSecondsRebuild << " seconds (" << remaining_size/ elapsedSecondsRebuild << " points/second)\n ";

        // auto pos = save_path.find_last_of('/');
        // assert(pos != std::string::npos);
        // std::string dir = save_path.substr(0, pos + 1);     // includes last slash
        // std::string fname = save_path.substr(pos + 1);
        // const std::string save_rebuild_path_inc = dir + "rebuild_" + fname + ".last-" + std::to_string(remaining_size);
        // diskann::cout << "rebuild path: " << save_rebuild_path_inc << std::endl;
        // rebuild_index->save(save_rebuild_path_inc.c_str(), true);

        // search on rebuilt index
        run_search(*rebuild_index);
    } catch (std::system_error &e) {
        diskann::cout << "Error occurred: " << e.what() << std::endl;
        throw diskann::ANNException("Rebuild failed", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::aligned_free(data);
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, index_path_prefix;
    uint32_t num_threads, R, L, num_start_pts;
    float alpha, start_point_norm;
    size_t points_to_skip, max_points_to_insert, beginning_index_size, points_per_checkpoint, checkpoints_per_snapshot,
        points_to_delete_from_beginning, start_deletes_after;
    bool concurrent;

    std::string /*result_path,*/ query_file, gt_file, filter_label, label_type, query_filters_file;
    uint32_t K;
    std::vector<uint32_t> Lvec;
    bool print_all_recalls, dynamic, tags, show_qps_per_thread;
    float fail_if_recall_below = 0.0f;

    // label options
    std::string label_file, universal_label;
    std::uint32_t Lf, unique_labels_supported;

    po::options_description desc{program_options_utils::make_program_description("test_insert_deletes_consolidate",
                                                                                 "Test insert deletes & consolidate")};
    // po::options_description desc{
        // program_options_utils::make_program_description("search_memory_index", "Searches in-memory DiskANN indexes")};

    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);
        required_configs.add_options()("points_to_skip", po::value<uint64_t>(&points_to_skip)->required(),
                                       "Skip these first set of points from file");
        required_configs.add_options()("beginning_index_size", po::value<uint64_t>(&beginning_index_size)->required(),
                                       "Batch build will be called on these set of points");
        required_configs.add_options()("points_per_checkpoint", po::value<uint64_t>(&points_per_checkpoint)->required(),
                                       "Insertions are done in batches of points_per_checkpoint");
        required_configs.add_options()("checkpoints_per_snapshot",
                                       po::value<uint64_t>(&checkpoints_per_snapshot)->required(),
                                       "Save the index to disk every few checkpoints");
        required_configs.add_options()("points_to_delete_from_beginning",
                                       po::value<uint64_t>(&points_to_delete_from_beginning)->required(), "");
        // required_configs.add_options()("result_path", po::value<std::string>(&result_path)->required(),
        //                                 program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                        program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                        program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list",
                                        po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                        program_options_utils::SEARCH_LIST_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA);
        optional_configs.add_options()("max_points_to_insert",
                                       po::value<uint64_t>(&max_points_to_insert)->default_value(0),
                                       "These number of points from the file are inserted after "
                                       "points_to_skip");
        optional_configs.add_options()("do_concurrent", po::value<bool>(&concurrent)->default_value(false), "");
        optional_configs.add_options()("start_deletes_after",
                                       po::value<uint64_t>(&start_deletes_after)->default_value(0), "");
        optional_configs.add_options()("start_point_norm", po::value<float>(&start_point_norm)->default_value(0),
                                       "Set the start point to a random point on a sphere of this radius");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()(
            "dynamic", po::value<bool>(&dynamic)->default_value(false),
            "Whether the index is dynamic. Dynamic indices must have associated tags.  Default false.");
        optional_configs.add_options()("tags", po::value<bool>(&tags)->default_value(false),
                                       "Whether to search with external identifiers (tags). Default false.");
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);

        // optional params for filters
        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       "Input label file in txt format for Filtered Index search. "
                                       "The file should contain comma separated filters for each node "
                                       "with each line corresponding to a graph node");
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       "Universal label, if using it, only in conjunction with labels_file");
        optional_configs.add_options()("FilteredLbuild,Lf", po::value<uint32_t>(&Lf)->default_value(0),
                                       "Build complexity for filtered points, higher value "
                                       "results in better graphs");
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       "Storage type of Labels <uint/ushort>, default value is uint which "
                                       "will consume memory 4 bytes per filter");
        optional_configs.add_options()("unique_labels_supported",
                                       po::value<uint32_t>(&unique_labels_supported)->default_value(0),
                                       "Number of unique labels supported by the dynamic index.");

        optional_configs.add_options()(
            "num_start_points",
            po::value<uint32_t>(&num_start_pts)->default_value(diskann::defaults::NUM_FROZEN_POINTS_DYNAMIC),
            "Set the number of random start (frozen) points to use when "
            "inserting and searching");

        // Output controls
        po::options_description output_controls("Output controls");
        output_controls.add_options()("print_all_recalls", po::bool_switch(&print_all_recalls),
                                      "Print recalls at all positions, from 1 up to specified "
                                      "recall_at value");
        output_controls.add_options()("print_qps_per_thread", po::bool_switch(&show_qps_per_thread),
                                      "Print overall QPS divided by the number of threads in "
                                      "the output table");       

        // Merge required and optional parameters
        // desc.add(required_configs).add(optional_configs);
        desc.add(required_configs).add(optional_configs).add(output_controls);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (beginning_index_size == 0)
            if (start_point_norm == 0)
            {
                std::cout << "When beginning_index_size is 0, use a start "
                             "point with  "
                             "appropriate norm"
                          << std::endl;
                return -1;
            }
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    bool has_labels = false;
    if (!label_file.empty() || label_file != "")
    {
        has_labels = true;
    }

    if (num_start_pts < unique_labels_supported)
    {
        num_start_pts = unique_labels_supported;
    }

    auto run_search = [&](auto &index) {
        diskann::Metric metric;
        if ((dist_fn == std::string("mips")) && (data_type == std::string("float")))
        {
            metric = diskann::Metric::INNER_PRODUCT;
        }
        else if (dist_fn == std::string("l2"))
        {
            metric = diskann::Metric::L2;
        }
        else if (dist_fn == std::string("cosine"))
        {
            metric = diskann::Metric::COSINE;
        }
        else if ((dist_fn == std::string("fast_l2")) && (data_type == std::string("float")))
        {
            metric = diskann::Metric::FAST_L2;
        }
        else
        {
            std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                        "supported in general, and mips/fast_l2 only for floating "
                        "point data."
                    << std::endl;
            return -1;
        }

        if (dynamic && not tags)
        {
            std::cerr << "Tags must be enabled while searching dynamically built indices" << std::endl;
            return -1;
        }

        if (fail_if_recall_below < 0.0 || fail_if_recall_below >= 100.0)
        {
            std::cerr << "fail_if_recall_below parameter must be between 0 and 100%" << std::endl;
            return -1;
        }

        std::vector<std::string> query_filters;

        try
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t>(index, metric, query_file, gt_file,
                                                num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                show_qps_per_thread, query_filters, fail_if_recall_below);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t>(index, metric, query_file, gt_file,
                                                    num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                    show_qps_per_thread, query_filters, fail_if_recall_below);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float>(index, metric, query_file, gt_file,
                                                num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                show_qps_per_thread, query_filters, fail_if_recall_below);
            }
            else
            {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        }
        catch (std::exception &e)
        {
            std::cout << std::string(e.what()) << std::endl;
            diskann::cerr << "Index search failed." << std::endl;
            return -1;
        }
    };

    try
    {
        diskann::IndexWriteParameters params = diskann::IndexWriteParametersBuilder(L, R)
                                                   .with_max_occlusion_size(500)
                                                   .with_alpha(alpha)
                                                   .with_num_threads(num_threads)
                                                   .with_filter_list_size(Lf)
                                                   .build();

        diskann::Metric metric;
        if (dist_fn == std::string("mips"))
        {
            metric = diskann::Metric::INNER_PRODUCT;
        }
        else if (dist_fn == std::string("l2"))
        {
            metric = diskann::Metric::L2;
        }
        else if (dist_fn == std::string("cosine"))
        {
            metric = diskann::Metric::COSINE;
        }
        else
        {
            std::cout << "Unsupported distance function. Currently only L2/ Inner "
                        "Product/Cosine are supported."
                        << std::endl;
            return -1;
        }

        if (data_type == std::string("int8"))
        {
            // auto index = build_incremental_index<int8_t>(
            build_incremental_index<int8_t>(
                data_path, params, points_to_skip, max_points_to_insert, beginning_index_size, start_point_norm,
                num_start_pts, points_per_checkpoint, checkpoints_per_snapshot, index_path_prefix,
                points_to_delete_from_beginning, start_deletes_after, concurrent, label_file, universal_label, metric, run_search);
            // return run_search(index);
        }
        else if (data_type == std::string("uint8"))
        {
            // auto index = build_incremental_index<uint8_t>(
            build_incremental_index<uint8_t>(
                data_path, params, points_to_skip, max_points_to_insert, beginning_index_size, start_point_norm,
                num_start_pts, points_per_checkpoint, checkpoints_per_snapshot, index_path_prefix,
                points_to_delete_from_beginning, start_deletes_after, concurrent, label_file, universal_label, metric, run_search);
            // return run_search(index);
        }
        else if (data_type == std::string("float"))
        {
            // auto index = build_incremental_index<float>(
            build_incremental_index<float>(
                data_path, params, points_to_skip, max_points_to_insert,
                beginning_index_size, start_point_norm, num_start_pts, points_per_checkpoint,
                checkpoints_per_snapshot, index_path_prefix, points_to_delete_from_beginning,
                start_deletes_after, concurrent, label_file, universal_label, metric, run_search);
            // return run_search(index);
        }
        else
            std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        exit(-1);
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception" << std::endl;
        exit(-1);
    }
}