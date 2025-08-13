
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

#include <iostream>
#include <fstream>

#include <fmt/format.h>


void load_data(const std::string &filename, float *&data, int &num,
               int &dim)
{ // load data with sift10K/sift1M/gist1M pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "Error: open file! " << filename << std::endl;
        exit(-1);
    }
    else
    {
        std::cout << "Data loading from " << filename << std::endl;
    }
    in.read((char *)&dim, 4);
    std::cout << "Data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    std::cout << "Data quantity: " << num << std::endl;
    
    data = new float[dim * num];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
    std::cout << "Data loading completed!" << std::endl;
}

void load_data_groundtruth(const std::string &filename, unsigned int *&data,
                            int groundtruth_num, int query_num)
{ // load data with sift10K/sift1M/gist1M pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "Error: open file! " << filename << std::endl;
        exit(-1);
    }
    else
    {
        std::cout << "Data loading from " << filename << std::endl;
    }
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;

    if ((unsigned)(fsize / (groundtruth_num + 1) / 4) != query_num)
    {
        std::cout << "fsize = " << (fsize / (groundtruth_num + 1) / 4) << " query_num = " << query_num << std::endl;
        std::cout << "Error: file size!" << std::endl;
        exit(-1);
    }
    else
    {
        std::cout << "Data quantity: " << query_num << std::endl;
    };

    data = (unsigned int *)new char[query_num * groundtruth_num * sizeof(unsigned int)];

    in.seekg(0, std::ios::beg);
    unsigned int temp;
    for (size_t i = 0; i < query_num; i++)
    {
        in.read((char *)&temp, 4);
        if (temp != groundtruth_num)
        {
            std::cout << "Error: temp value!" << std::endl;
            exit(-1);
        }
        in.read((char *)(data + i * groundtruth_num), temp * 4);
        // if (i == 0)
        //     for (int j = 0; j < temp; j++)
        //         std::cout << data + i * groundtruth_num << std::endl;
    }
    in.close();
    std::cout << "Data loading completed!" << std::endl;
}

int
main(int argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    // Parameters for building
    int max_degree = 32;
    int ef_construction = 256;

    // Parameters for searching
    int ef_search = 100;
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
        std::cerr << "Error: open file! " << save_file << std::endl;
        exit(-1);
    } else {
        std::cout << "Data loading from " << save_file << std::endl;
        index->Deserialize(in_file_stream);
        std::cout << "Data loading completed! Current HNSW holds "
            << index->GetNumElements() << " points" << std::endl;
    }

    /******************* Test HNSW Query *****************/
    // Read in dataset for query-test
    int query_num_vectors = 0;
    int query_dim = 0;
    float *query_vectors = NULL;
    load_data(query_filename, query_vectors, query_num_vectors, query_dim);
    assert(query_dim == dim);
    std::cout << "Query data loaded: " << query_num_vectors
              << " vectors with dimension " << query_dim << std::endl;

    // Read in groundtruth for query
    unsigned int *groundtruth;
    load_data_groundtruth(groundtruth_filename, groundtruth,
                        groundtruth_num, query_num_vectors);

    // Perform query on the index
    // The "hnsw_search_parameters" is the configuration for searching in an HNSW index.
    // The "hnsw" section contains parameters specific to the search operation:
    // - "ef_search": The size of the dynamic list used for nearest neighbor search,
    // which influences both recall and search speed.
    auto hnsw_search_parameters = std::string(
        R"(
        {{
            "hnsw": {{
                "ef_search": {}
            }}
        }}
        )");
    hnsw_search_parameters = fmt::format(hnsw_search_parameters, ef_search);

    uint64_t correct = 0;
    float *query_single_vector = new float[query_dim];
    auto query_base = vsag::Dataset::Make();
    std::cout << "Test query with topK = " << topk << ", candidate_size = " << ef_search << std::endl;
    for(int i_search = 0; i_search < query_num_vectors; ++i_search) {
        memcpy(query_single_vector, query_vectors + i_search * query_dim, query_dim * sizeof(float));
        // Get query vector
        query_base->NumElements(1)->Dim(query_dim)->Float32Vectors(query_single_vector)->Owner(true);

        auto knn_result = index->KnnSearch(query_base, topk, hnsw_search_parameters);
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
            std::cerr << "Search Error: " << knn_result.error().message << std::endl;
            exit(1);
        }
    }

    std::cout << "Query done. Recall = " << correct * 1.0 / topk / query_num_vectors << std::endl;

    return 0;
}
