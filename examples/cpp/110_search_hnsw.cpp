
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vsag/vsag.h>
#include "index/hnsw.h"
#include "utils/util_functions.h"
#include "utils/crouting_timer.h"
#include "logger.h"

#include <iostream>
#include <fstream>

#include <fmt/format.h>

int
main(int argc, char** argv) {
#ifdef CROUTING_COLLECT_INFO
    vsag::logger::info("Open information collection");
#endif
    /******************* Prepare Base Dataset *****************/
    // Parameters for building
    int max_degree = 64;
    int ef_construction = 300;

    // Parameters for searching
    int ef_search_list[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 900};
    int ef_search = 0;
    int64_t topk = 10;
    int groundtruth_num = 100;

    int64_t dim = 128;
    std::string filename = std::string("/data/gua/datafile/sift/sift_base.fvecs");
    std::string query_filename = std::string("/data/gua/datafile/sift/sift_query.fvecs");
    std::string groundtruth_filename = std::string("/data/gua/datafile/sift/sift_groundtruth.ivecs");
    std::string save_file = std::string("/data/gua/newgraph/sift/sift_base_"
        + std::to_string(max_degree) + "_"
        + std::to_string(ef_construction)
        + ".vsag");

    /******************* Create HNSW Index *****************/
    // hnsw_build_parameters is the configuration for building an HNSW index.
    // The "dtype" specifies the data type, which supports float32 and int8.
    // The "metric_type" indicates the distance metric type (e.g., cosine, inner product, and L2).
    // The "dim" represents the dimensionality of the vectors, indicating the number of features for each data point.
    // The "hnsw" section contains parameters specific to HNSW:
    // - "max_degree": The maximum number of connections for each node in the graph.
    // - "ef_construction": The size used for nearest neighbor search during graph construction, which affects both speed and the quality of the graph.
    std::string hnsw_build_parameters = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": )" + std::to_string(dim) + R"(,
        "hnsw": {
            "max_degree": )" + std::to_string(max_degree) + R"(,
            "ef_construction": )" + std::to_string(ef_construction) + R"(
        }
    })";
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_parameters).value();

    // Deserialize hnsw
    std::ifstream in_file_stream = std::ifstream(save_file);
    if (!in_file_stream.is_open()) {
        vsag::logger::error("Error: open file! {}", save_file);
        exit(-1);
    } else {
        vsag::logger::info("Data loading from {}", save_file);
        index->Deserialize(in_file_stream);
        vsag::logger::info("Data loading completed! Current HNSW holds {} points",
                           index->GetNumElements());
    }

    /******************* Test HNSW Query *****************/
    // Read in dataset for query-test
    int query_num_vectors = 0;
    int query_dim = 0;
    float *query_vectors = NULL;
    vsag::load_data(query_filename, query_vectors, query_num_vectors, query_dim);
    assert(query_dim == dim);
    vsag::logger::info("Query data loaded: {} vectors with dimension {}",
                       query_num_vectors, query_dim);

    // Read in groundtruth for query
    unsigned int *groundtruth;
    vsag::load_data_groundtruth(groundtruth_filename, groundtruth,
                        groundtruth_num, query_num_vectors);

    // Perform query on the index
    // The "hnsw_search_parameters" is the configuration for searching in an HNSW index.
    // The "hnsw" section contains parameters specific to the search operation:
    // - "ef_search": The size of the dynamic list used for nearest neighbor search,
    // which influences both recall and search speed.
    auto hnsw_search_parameters_base = std::string(
        R"(
        {{
            "hnsw": {{
                "ef_search": {}
            }}
        }}
        )");
#ifdef USE_DOUBLE_CHECK_HNSW
        index->SetUseDoubleCheck(true);
#endif

    for(size_t i_ef_search = 0; i_ef_search < sizeof(ef_search_list) / sizeof(int); ++i_ef_search) {
#ifdef CROUTING_COLLECT_INFO
        // Reset counter and timer
        vsag::counter_hops_search_1 = 0;
        vsag::counter_hops_search_2 = 0;
        vsag::counter_pass_during_search_1 = 0;
        vsag::counter_pass_during_search_2 = 0;
        double query_timer = 0;
#endif

        ef_search = ef_search_list[i_ef_search];
        auto hnsw_search_parameters = fmt::format(hnsw_search_parameters_base, ef_search);

        uint64_t correct = 0;
        float *query_single_vector = new float[query_dim];
        auto query_base = vsag::Dataset::Make();
        vsag::logger::info("Start query test with topK = {}, ef_search = {}", topk, ef_search);
        for(int i_search = 0; i_search < query_num_vectors; ++i_search) {
            memcpy(query_single_vector, query_vectors + i_search * query_dim, query_dim * sizeof(float));
            // Get query vector
            query_base->NumElements(1)->Dim(query_dim)->Float32Vectors(query_single_vector)->Owner(true);
#ifdef CROUTING_COLLECT_INFO
            uint64_t timer = NowNanos();
#endif
            auto knn_result = index->KnnSearch(query_base, topk, hnsw_search_parameters);
#ifdef CROUTING_COLLECT_INFO
            query_timer += static_cast<double>(ElapsedNanos(timer));
#endif
            if(knn_result.has_value()) {
                auto result = knn_result.value();
                auto res_ids = result->GetIds();
                for(int i_topk = 0; i_topk < topk; ++i_topk) {
                    if(std::find(groundtruth + i_search * groundtruth_num,
                                groundtruth + i_search * groundtruth_num + topk,
                                res_ids[i_topk]) 
                        != groundtruth + i_search * groundtruth_num + topk
                    ) {
                        ++correct;
                    }
                }
            } else {
                vsag::logger::error("Search failed for query {}: {}", i_search, knn_result.error().message);
                exit(1);
            }
        }

#ifdef CROUTING_COLLECT_INFO
        vsag::logger::info("counter_hops_search_1: {}", vsag::counter_hops_search_1);
        vsag::logger::info("counter_hops_search_2: {}", vsag::counter_hops_search_2);
        vsag::logger::info("counter_pass_during_search_1: {}", vsag::counter_pass_during_search_1);
        vsag::logger::info("counter_pass_during_search_2: {}", vsag::counter_pass_during_search_2);
        vsag::logger::info("total latency: {} ns", query_timer);
        vsag::logger::info("average query latency: {} ns", query_timer / query_num_vectors);
#endif

        vsag::logger::info("ef_search {} with recall: {} (#correct = {})", ef_search, static_cast<float>(correct) / (query_num_vectors * topk), correct);
    }
    return 0;
}
