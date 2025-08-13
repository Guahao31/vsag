
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

int
main(int argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    // Parameters for building
    int max_degree = 32;
    int ef_construction = 256;

    int64_t num_vectors = 1000;
    int64_t dim = 128;
    // auto vectors = new float[dim * num_vectors];
    float *vectors = NULL;
    std::string filename = std::string("/data/gua/datafile/sift/sift_base.fvecs");
    std::string save_file = std::string("/data/gua/newgraph/sift/sift_base_"
        + std::to_string(max_degree) + "_"
        + std::to_string(ef_construction)
        + ".vsag");

    // Readin dataset
    int t_num_vectors = 0;
    int t_dim = 0;
    load_data(filename, vectors, t_num_vectors, t_dim);
    dim = static_cast<int64_t>(t_dim);
    num_vectors = static_cast<int64_t>(t_num_vectors);
    std::cout << "Data loaded: " << num_vectors << " vectors with dimension " << dim << std::endl;
    auto ids = new int64_t[num_vectors];
    for(int64_t i = 0; i < num_vectors; ++i) ids[i] = i;

    auto base = vsag::Dataset::Make();
    // Transfer the ownership of the data (ids, vectors) to the base.
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);

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

    /******************* Build HNSW Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HNSW contains: " << index->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    std::cout << "Saving graph into file: " << save_file << std::endl;
    std::ofstream out_file_stream = std::ofstream(save_file);
    if(auto save_result = index->Serialize(out_file_stream); save_result.has_value()) {
        std::cout << "Saved graph successfully" << std::endl;
    } else {
        std::cerr << "Failed to save index: " << save_result.error().message << std::endl;
        exit(-1);
    }


    /******************* KnnSearch For HNSW Index *****************/
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    auto query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    // hnsw_search_parameters is the configuration for searching in an HNSW index.
    // The "hnsw" section contains parameters specific to the search operation:
    // - "ef_search": The size of the dynamic list used for nearest neighbor search, which influences both recall and search speed.
    auto hnsw_search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);
    auto knn_result = index->KnnSearch(query, topk, hnsw_search_parameters);

    /******************* Print Search Result *****************/
    if (knn_result.has_value()) {
        auto result = knn_result.value();
        std::cout << "results: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result.error().message << std::endl;
    }

    return 0;
}
