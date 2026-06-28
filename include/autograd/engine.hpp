#pragma once

#include <vector>

#include "core/autograd_meta.hpp"  // For AutogradProvider
#include "core/tensor.hpp"

namespace helix {

    class BackwardEngine {
    public:
        void run(Tensor& target, const std::vector<Tensor>& grad_outputs = {});
    };

    class AutogradEngineProvider : public AutogradProvider {
    public:
        AutogradMeta* create_meta() override;
        void backward(Tensor& tensor, const std::vector<Tensor>& grad_outputs) override;
        Tensor& get_grad(const Tensor& tensor) override;
        const Tensor& get_grad(const Tensor& tensor) const override;
        bool has_grad(const Tensor& tensor) const override;

    private:
        BackwardEngine engine_;
    };

    // To initialize Autograd properly at startup or link time
    void init_autograd();

}  // namespace helix
