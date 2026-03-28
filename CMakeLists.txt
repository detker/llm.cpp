cmake_minimum_required(VERSION 3.20)

project(llm_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Debug/Release/etc)" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")

add_executable(llm_cpp src/main.cpp
        include/Config.hpp
        include/TransformerWeights.hpp
        include/RunState.hpp
        include/utils.hpp
        external/nlohmann/json.hpp
        include/errorUtils.hpp
        src/utils.cpp
        src/RunState.cpp
        include/Inference.hpp
        src/Inference.cpp
        include/Sampler.hpp
        src/Sampler.cpp
        include/Tokenizer.hpp
        src/Tokenizer.cpp
        include/Timer.hpp
)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET llm_cpp PROPERTY CUDA_ARCHITECTURES native)
endif()

set_target_properties(llm_cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(llm_cpp PRIVATE CUDA::cudart)

target_include_directories(llm_cpp PRIVATE
        include
        external
)

