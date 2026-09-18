#pragma once

#include <optional>
#include <vector>

#include "core/autograd_meta.hpp"  // For AutogradProvider
#include "core/tensor.hpp"

namespace helix {

    class GraphBuilderInterface;

    /**
     * @class no_grad
     * @brief Context-manager that disables gradient calculation.
     *
     * Disabling gradient calculation is useful for inference, when you are sure
     * that you will not call Tensor::backward(). It reduces memory consumption
     * and computational overhead.
     */
    class no_grad {
    public:
        no_grad();
        ~no_grad() noexcept;

        no_grad(const no_grad&) = delete;
        auto operator=(const no_grad&) -> no_grad& = delete;
        no_grad(no_grad&&) = delete;
        auto operator=(no_grad&&) -> no_grad& = delete;

    private:
        std::optional<GraphBuilderInterface*> prev_builder_override_;
    };

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
        bool is_leaf(const Tensor& tensor) const override;

    private:
        BackwardEngine engine_;
    };

    // To initialize Autograd properly at startup or link time
    void init_autograd();

}  // namespace helix
