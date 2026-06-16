#pragma once

namespace helix {

    // Forward declaration of AutogradMeta
    // This allows TensorImpl to hold a pointer to AutogradMeta
    // without introducing a compile-time dependency on the autograd module.
    class AutogradMeta;

    // Custom deleter for AutogradMeta.
    // The implementation must be provided by the autograd module to ensure Core
    // has no compile-time dependency on Autograd's internals.
    struct AutogradMetaDeleter {
        void operator()(AutogradMeta* meta) const;
    };

    class Tensor;

    // Interface to delegate Autograd-specific operations from Core to the Autograd module.
    class AutogradProvider {
    public:
        virtual ~AutogradProvider() = default;
        virtual AutogradMeta* create_meta() = 0;
        virtual void backward(Tensor& tensor, const std::vector<Tensor>& grad_outputs) = 0;
        virtual Tensor& get_grad(const Tensor& tensor) = 0;
        virtual const Tensor& get_grad(const Tensor& tensor) const = 0;
    };

    void register_autograd_provider(AutogradProvider* provider);
    AutogradProvider* get_autograd_provider();

}  // namespace helix
