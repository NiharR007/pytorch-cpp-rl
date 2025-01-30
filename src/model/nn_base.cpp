#include "cpprl/doctest_disable.h"
#include "third_party/doctest.h"
#include <torch/torch.h>

#include "cpprl/model/nn_base.h"
#include "cpprl/model/model_utils.h"

using namespace torch;

namespace cpprl
{
NNBase::NNBase(bool recurrent,
               unsigned int recurrent_input_size,
               unsigned int hidden_size)
    : gru(nullptr),
      hidden_size(hidden_size),
      recurrent(recurrent)
{
    // Init GRU
    if (recurrent)
    {
        gru = nn::GRU(nn::GRUOptions(recurrent_input_size, hidden_size));
        register_module("gru", gru);
        // Init weights
        init_weights(gru->named_parameters(), 1, 0);
    }
}

// Do not use.
//
// Instantiate a subclass and use theirs instead
std::vector<torch::Tensor> NNBase::forward(torch::Tensor /*inputs*/,
                                           torch::Tensor /*hxs*/,
                                           torch::Tensor /*masks*/)
{
    return std::vector<torch::Tensor>();
}

unsigned int NNBase::get_hidden_size() const
{
    if (recurrent)
    {
        return hidden_size;
    }
    return 1;
}

std::vector<torch::Tensor> NNBase::forward_gru(
    torch::Tensor x, torch::Tensor hxs, torch::Tensor masks)
{
    // GRU now returns a tuple of (output, hidden_state)
    auto gru_output = gru->forward(x.unsqueeze(0),
                                 (hxs * masks).unsqueeze(0));
    
    // Use std::get to access tuple elements
    auto output = std::get<0>(gru_output);
    auto state = std::get<1>(gru_output);
    
    return {output.squeeze(0), state.squeeze(0)};
}

std::vector<torch::Tensor> NNBase::forward_gru_sequence(
    torch::Tensor x, torch::Tensor hxs, torch::Tensor masks)
{
    std::vector<torch::Tensor> outputs;
    auto gru_output = gru->forward(x,
                                 (hxs * masks).unsqueeze(0));
    
    // Use std::get to access the output
    outputs.push_back(std::get<0>(gru_output));
    return outputs;
}

TEST_CASE("NNBase")
{
    auto base = std::make_shared<NNBase>(true, 5, 10);

    SUBCASE("forward_gru() outputs correct shapes when given samples from one"
            " agent")
    {
        auto inputs = torch::rand({4, 5});
        auto rnn_hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = base->forward_gru(inputs, rnn_hxs, masks);

        REQUIRE(outputs.size() == 2);

        // x
        CHECK(outputs[0].size(0) == 4);
        CHECK(outputs[0].size(1) == 10);

        // rnn_hxs
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 10);
    }

    SUBCASE("forward_gru() outputs correct shapes when given samples from "
            "multiple agents")
    {
        auto inputs = torch::rand({12, 5});
        auto rnn_hxs = torch::rand({4, 10});
        auto masks = torch::zeros({12, 1});
        auto outputs = base->forward_gru(inputs, rnn_hxs, masks);

        REQUIRE(outputs.size() == 2);

        // x
        CHECK(outputs[0].size(0) == 12);
        CHECK(outputs[0].size(1) == 10);

        // rnn_hxs
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 10);
    }
}
}