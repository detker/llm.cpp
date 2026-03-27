cmake_minimum_required(VERSION 4.1)
project(llm_cpp)

set(CMAKE_CXX_STANDARD 20)

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
        src/Tokenizer.cpp)

target_include_directories(llm_cpp PRIVATE
        include
        external
)

