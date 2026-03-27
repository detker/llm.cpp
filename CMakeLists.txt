cmake_minimum_required(VERSION 4.1)
project(llm_cpp)

set(CMAKE_CXX_STANDARD 20)

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
        src/Inference.cpp)

target_include_directories(llm_cpp PRIVATE
        include
        external
)

