#pragma once

#include "optim/optimizer.hpp"

namespace helix {

    class SGD : public Optimizer {
    public:
        SGD(std::vector<Tensor> params, float lr);

        void step() override;

        float learning_rate() const { return learning_rate_; }
        void set_learning_rate(const float lr) { learning_rate_ = lr; }

    private:
        float learning_rate_;
    };

}  // namespace helix
