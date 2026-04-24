#include <atomic>
#include <iostream>
#include <memory>
#include <variant>
#include <vector>

#include "Config.hpp"
#include "Dispatcher.hpp"
#include "Inference.hpp"
#include "InferenceLoop.hpp"
#include "RunState.hpp"
#include "Sampler.hpp"
#include "Timer.hpp"
#include "Tokenizer.hpp"
#include "TransformerWeights.hpp"
#include "utils.hpp"

std::atomic<bool> InferenceLoop::interrupt_requested{false};

int main(int argc, char **argv) {
  std::mt19937 rng(std::random_device{}());

  MiscUtils::ParseResult args = MiscUtils::parseArgs(argc, argv);
  DataUtils dataUtils(args);
  Config config = dataUtils.getConfig();
  if (args.max_seq_len > 0)
    config.max_seq_len = args.max_seq_len;
  std::unique_ptr<Tokenizer> tokenizer = dataUtils.getTokenizer();
  RunState runState(&config);
  Sampler sampler(rng);

  auto [elapsed_seconds, tokens_generated] =
      RunInference<1>(std::move(args.txt), &config, &runState, &dataUtils,
                      std::move(tokenizer), &sampler);

  if (tokens_generated == 0) {
    ERR("No tokens generated");
  }

  MiscUtils::Metrics metrics{.tokens_generated = tokens_generated,
                             .elapsed_seconds = elapsed_seconds,
                             .weights_size_bytes = MiscUtils::calcWeightSize(
                                 args.model_path.c_str(),
                                 dataUtils.getHeaderSize(),
                                 config.vocab_size * sizeof(char))};

  MiscUtils::printMetrics(metrics);

  return EXIT_SUCCESS;
}
