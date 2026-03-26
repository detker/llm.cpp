cmake_minimum_required(VERSION 4.1)
project(llm_cpp)

set(CMAKE_CXX_STANDARD 20)

add_executable(llm_cpp src/main.cpp
        include/Config.hpp
        include/TransformerWeights.hpp
        include/RunState.hpp
        include/utils.hpp)

target_include_directories(llm_cpp PRIVATE include)

