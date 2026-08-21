#pragma once

#include <memory>
#include <vector>

#include "core/dtype.hpp"

namespace helix {

    // Forward declaration of AutogradMeta
    // This allows TensorImpl to hold a pointer to AutogradMeta
    // without introducing a compile-time dependency on the autograd module.
    class AutogradMeta;

    class Tensor;

    // Interface to delegate Autograd-specific operations from Core to the Autograd module.
    class AutogradProvider {
    public:
        virtual ~AutogradProvider() = default;
        virtual std::shared_ptr<AutogradMeta> create_meta(DType dtype) = 0;
        virtual void backward(Tensor& tensor, const std::vector<Tensor>& grad_outputs, bool retain_graph) = 0;
        virtual Tensor& get_grad(const Tensor& tensor) = 0;
        virtual const Tensor& get_grad(const Tensor& tensor) const = 0;
        virtual bool has_grad(const Tensor& tensor) const = 0;
    };

    void register_autograd_provider(AutogradProvider* provider);
    AutogradProvider* get_autograd_provider();

}  // namespace helix
