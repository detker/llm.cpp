#ifndef LLM_CPP_ERRORUTILS_HPP
#define LLM_CPP_ERRORUTILS_HPP

#include <cstdio>
#include <cstdlib>

#define ERR(source) (perror(source), fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), exit(EXIT_FAILURE))

#define CUDA_CHECK(call) do {                                                                 \
    cudaError_t e = (call);                                                                   \
    if (e != cudaSuccess) {                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                                              \
    }                                                                                         \
} while(0)

#define NCCL_CHECK(call) do {                                                                 \
    ncclResult_t r = (call);                                                                  \
    if (r != ncclSuccess) {                                                                   \
        fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(1);                                                                              \
    }                                                                                         \
} while(0)

inline void usage(const char* prog_name) {
    fprintf(stderr, "Usage: %s <data_path> <dim> <k> <niter>\n", prog_name);
    exit(EXIT_FAILURE);
}

#endif //LLM_CPP_ERRORUTILS_HPP