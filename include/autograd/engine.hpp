#pragma once

#include <vector>

#include "core/autograd_meta.hpp"  // For AutogradProvider
#include "core/tensor.hpp"

namespace helix {

    class BackwardEngine {
    public:
        void run(Tensor& target, const std::vector<Tensor>& grad_outputs = {}, bool retain_graph = false);
    };

    class AutogradEngineProvider : public AutogradProvider {
    public:
        std::shared_ptr<AutogradMeta> create_meta(DType dtype) override;
        void backward(Tensor& tensor, const std::vector<Tensor>& grad_outputs, bool retain_graph) override;
        Tensor& get_grad(const Tensor& tensor) override;
        const Tensor& get_grad(const Tensor& tensor) const override;
        bool has_grad(const Tensor& tensor) const override;

    private:
        BackwardEngine engine_;
    };

    // To initialize Autograd properly at startup or link time
    void init_autograd();

}  // namespace helix
