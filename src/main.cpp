#include <iostream>
#include <vector>

#include "Config.hpp"
#include "utils.hpp"
#include "TransformerWeights.hpp"
#include "RunState.hpp"

int main(int argc, char **argv) {
    DataUtils dataUtils("../model.bin");
    Config config = dataUtils.getConfig();
    TransformerWeights weights = dataUtils.mapModelWeights();
    RunState runState = RunState(config);


    return 0;
}