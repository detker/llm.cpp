#ifndef LLM_CPP_TRANSFORMERWEIGHTS_HPP
#define LLM_CPP_TRANSFORMERWEIGHTS_HPP

#include <iostream>
#include <concepts>
#include <memory>
#include <cuda_runtime.h>
#include <errorUtils.hpp>

typedef uint16_t float16_t;

template<typename T>
concept FP1632 = std::same_as<T, float> || std::same_as<T, float16_t>;

template<FP1632 T>
class WeightsVector {
public:
    WeightsVector() = delete;
    WeightsVector(const WeightsVector&) = delete;
    WeightsVector& operator=(const WeightsVector&) = delete;

    WeightsVector(const T *data, int m, int n, DType dtype, BackendType btype)
        : m(m), n(n), btype(btype), owns_memory(btype == BackendType::GPU) {
        if (btype == BackendType::CPU) {
            this->data = const_cast<T*>(data);
        }
        else if (btype == BackendType::GPU) {
            CUDA_CHECK(cudaMalloc((void **)&this->data, this->getSize()));
            CUDA_CHECK(cudaMemcpy((void *)this->data, data, this->getSize(), cudaMemcpyHostToDevice));
        }
    }

    WeightsVector(int m, int n, BackendType btype)
        : m(m), n(n), btype(btype), owns_memory(true) {
        if (btype == BackendType::CPU) {
            this->data = new T[m * n]();
        }
        else if (btype == BackendType::GPU) {
            CUDA_CHECK(cudaMalloc((void **)&this->data, this->getSize()));
        }
    }

    ~WeightsVector() {
        if (owns_memory) {
            if (btype == BackendType::GPU) {
                CUDA_CHECK(cudaFree((void *)this->data));
            } else {
                delete[] this->data;
            }
        }
    }

    [[nodiscard]] int getM() const {
        return m;
    }

    [[nodiscard]] int getN() const {
        return n;
    }

    [[nodiscard]] size_t getSize() const {
        return m * n * sizeof(T);
    }

    [[nodiscard]] const T* getData() const {
        return data;
    }

    T* getMutableData() {
        return data;
    }

    operator T*() { return data; }
    operator const T*() const { return data; }

    [[nodiscard]] DType getDType() const {
        if constexpr (std::is_same_v<T, float16_t>) {
            return DType::FP16;
        }
        else if constexpr (std::is_same_v<T, float>) {
            return DType::FP32;
        }
        else {
            ERR("Unsupported data type");
        }
    }

private:
    T *data;
    int m;
    int n;
    BackendType btype;
    bool owns_memory;
};


template<FP1632 T>
struct LayerWeightsAuto {
    std::unique_ptr<WeightsVector<float>> norm_att_weight;
    std::unique_ptr<WeightsVector<T>> q_proj_weight;
    std::unique_ptr<WeightsVector<T>> k_proj_weight;
    std::unique_ptr<WeightsVector<T>> v_proj_weight;
    std::unique_ptr<WeightsVector<T>> o_proj_weight;

    std::unique_ptr<WeightsVector<float>> mlp_norm_weight;
    std::unique_ptr<WeightsVector<T>> mlp_w1_weight;
    std::unique_ptr<WeightsVector<T>> mlp_w2_weight;
    std::unique_ptr<WeightsVector<T>> mlp_w3_weight;
};

template<FP1632 T>
struct TransformerWeightsAuto {
    std::unique_ptr<WeightsVector<T>> token_embd_table;

    std::unique_ptr<LayerWeightsAuto<T>[]> layer_weights;

    std::unique_ptr<WeightsVector<float>> final_norm_weight;
    std::unique_ptr<WeightsVector<T>> output_proj_weight;
};

#endif //LLM_CPP_TRANSFORMERWEIGHTS_HPP
