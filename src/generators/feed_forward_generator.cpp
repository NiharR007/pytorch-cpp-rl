#include <algorithm>
#include <vector>

#include <torch/torch.h>

#include "cpprl/generators/feed_forward_generator.h"
#include "cpprl/generators/generator.h"
#include "third_party/doctest.h"

namespace cpprl
{
FeedForwardGenerator::FeedForwardGenerator(int mini_batch_size,
                                           torch::Tensor observations,
                                           torch::Tensor hidden_states,
                                           torch::Tensor actions,
                                           torch::Tensor value_predictions,
                                           torch::Tensor returns,
                                           torch::Tensor masks,
                                           torch::Tensor action_log_probs,
                                           torch::Tensor advantages)
    : observations(observations),
      hidden_states(hidden_states),
      actions(actions),
      value_predictions(value_predictions),
      returns(returns),
      masks(masks),
      action_log_probs(action_log_probs),
      advantages(advantages),
      index(0)
{
    int batch_size = advantages.numel();
    indices = torch::randperm(batch_size,
                              torch::TensorOptions(torch::kLong))
                  .view({-1, mini_batch_size});
}

bool FeedForwardGenerator::done() const
{
    return index >= indices.size(0);
}

MiniBatch FeedForwardGenerator::next()
{
    if (index >= indices.size(0))
    {
        throw std::runtime_error("No minibatches left in generator.");
    }

    MiniBatch mini_batch;

    try {
        int timesteps = observations.size(0) - 1;
        if (timesteps <= 0) {
            throw std::runtime_error("Invalid timesteps value: " + std::to_string(timesteps));
        }

        // Debug print for indices
        const auto& current_indices = indices[index];
        std::cout << "\nDebug Info:" << std::endl;
        std::cout << "Current index: " << index << std::endl;
        std::cout << "Indices tensor shape: [";
        for (int64_t i = 0; i < current_indices.dim(); ++i) {
            std::cout << current_indices.size(i) << " ";
        }
        std::cout << "]" << std::endl;
        
        // Print max and min indices
        auto max_index = current_indices.max().item<int64_t>();
        auto min_index = current_indices.min().item<int64_t>();
        std::cout << "Min index: " << min_index << ", Max index: " << max_index << std::endl;

        auto observations_shape = observations.sizes().vec();
        observations_shape.erase(observations_shape.begin());
        observations_shape[0] = -1;

        // Debug print for tensor shapes before operations
        std::cout << "\nTensor shapes before operations:" << std::endl;
        
        // Observations
        auto obs_view = observations.narrow(0, 0, timesteps).view(observations_shape);
        std::cout << "observations view shape: [";
        for (int64_t i = 0; i < obs_view.dim(); ++i) {
            std::cout << obs_view.size(i) << " ";
        }
        std::cout << "]" << std::endl;

        // Check if indices are valid for each tensor
        if (max_index >= obs_view.size(0)) {
            throw std::runtime_error("Index out of range for observations: max_index = " + 
                std::to_string(max_index) + ", dim size = " + 
                std::to_string(obs_view.size(0)));
        }

        mini_batch.observations = obs_view.index_select(0, current_indices);

        // Hidden states
        auto hidden_view = hidden_states.narrow(0, 0, timesteps)
                                      .view({-1, hidden_states.size(-1)});
        std::cout << "hidden_states view shape: [";
        for (int64_t i = 0; i < hidden_view.dim(); ++i) {
            std::cout << hidden_view.size(i) << " ";
        }
        std::cout << "]" << std::endl;

        if (max_index >= hidden_view.size(0)) {
            throw std::runtime_error("Index out of range for hidden_states: max_index = " + 
                std::to_string(max_index) + ", dim size = " + 
                std::to_string(hidden_view.size(0)));
        }

        mini_batch.hidden_states = hidden_view.index_select(0, current_indices);

        // Actions
        auto actions_view = actions.view({-1, actions.size(-1)});
        std::cout << "actions view shape: [";
        for (int64_t i = 0; i < actions_view.dim(); ++i) {
            std::cout << actions_view.size(i) << " ";
        }
        std::cout << "]" << std::endl;

        if (max_index >= actions_view.size(0)) {
            throw std::runtime_error("Index out of range for actions: max_index = " + 
                std::to_string(max_index) + ", dim size = " + 
                std::to_string(actions_view.size(0)));
        }

        mini_batch.actions = actions_view.index_select(0, current_indices);

        // Continue with similar debug checks for other tensors...
        // Value predictions
        // Value predictions
        auto value_pred_view = value_predictions.narrow(0, 0, timesteps);
        std::cout << "value_predictions view shape: [";
        for (int64_t i = 0; i < value_pred_view.dim(); ++i) {
            std::cout << value_pred_view.size(i) << " ";
        }
        std::cout << "]" << std::endl;

        // Adjust indices for value_predictions
        int num_envs = 8; // Replace with your actual num_envs
        auto value_indices = (current_indices / num_envs).to(torch::kLong);

        // Validate adjusted indices
        auto max_value_index = value_indices.max().item<int64_t>();
        auto min_value_index = value_indices.min().item<int64_t>();
        std::cout << "Adjusted Min index for value_pred: " << min_value_index 
                << ", Adjusted Max index: " << max_value_index << std::endl;

        if (max_value_index >= value_pred_view.size(0)) {
            throw std::runtime_error("Adjusted index out of range for value_predictions: max_index = " + 
                std::to_string(max_value_index) + ", dim size = " + 
                std::to_string(value_pred_view.size(0)));
        }

        mini_batch.value_predictions = value_pred_view.index_select(0, value_indices);

        // Returns
        auto returns_view = returns.narrow(0, 0, timesteps).view({-1, 1});
        std::cout << "returns view shape: [";
        for (int64_t i = 0; i < returns_view.dim(); ++i) {
            std::cout << returns_view.size(i) << " ";
        }
        std::cout << "]" << std::endl;

        mini_batch.returns = returns_view.index_select(0, current_indices);

        // Masks
        auto masks_view = masks.narrow(0, 0, timesteps).view({-1, 1});
        mini_batch.masks = masks_view.index_select(0, current_indices);

        // Action log probs
        auto log_probs_view = action_log_probs.view({-1, 1});
        mini_batch.action_log_probs = log_probs_view.index_select(0, current_indices);

        // Advantages
        auto advantages_view = advantages.view({-1, 1});
        mini_batch.advantages = advantages_view.index_select(0, current_indices);

        index++;
        return mini_batch;
    }
    catch (const c10::Error& e) {
        throw std::runtime_error(std::string("PyTorch error in next(): ") + e.what());
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error in next(): ") + e.what());
    }
}

TEST_CASE("FeedForwardGenerator")
{
    FeedForwardGenerator generator(5, torch::rand({6, 3, 4}), torch::rand({6, 3, 3}),
                                   torch::rand({5, 3, 1}), torch::rand({6, 3, 1}),
                                   torch::rand({6, 3, 1}), torch::ones({6, 3, 1}),
                                   torch::rand({5, 3, 1}), torch::rand({5, 3, 1}));

    SUBCASE("Minibatch tensors are correct sizes")
    {
        auto minibatch = generator.next();

        CHECK(minibatch.observations.sizes().vec() == std::vector<int64_t>{5, 4});
        CHECK(minibatch.hidden_states.sizes().vec() == std::vector<int64_t>{5, 3});
        CHECK(minibatch.actions.sizes().vec() == std::vector<int64_t>{5, 1});
        CHECK(minibatch.value_predictions.sizes().vec() == std::vector<int64_t>{5, 1});
        CHECK(minibatch.returns.sizes().vec() == std::vector<int64_t>{5, 1});
        CHECK(minibatch.masks.sizes().vec() == std::vector<int64_t>{5, 1});
        CHECK(minibatch.action_log_probs.sizes().vec() == std::vector<int64_t>{5, 1});
        CHECK(minibatch.advantages.sizes().vec() == std::vector<int64_t>{5, 1});
    }

    SUBCASE("done() indicates whether the generator has finished")
    {
        CHECK(!generator.done());
        generator.next();
        CHECK(!generator.done());
        generator.next();
        CHECK(!generator.done());
        generator.next();
        CHECK(generator.done());
    }

    SUBCASE("Calling a generator after it has finished throws an exception")
    {
        generator.next();
        generator.next();
        generator.next();
        CHECK_THROWS(generator.next());
    }
}
}