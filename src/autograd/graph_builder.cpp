#include "autograd/graph_builder.hpp"
#include "autograd/function.hpp"
#include "autograd/autograd_meta.hpp"
#include "core/tensor_impl.hpp"

namespace helix {

    void AutogradGraphBuilder::build(const OperationContext& ctx) {
        // Step 1: Check if any input requires gradient
        bool requires_grad = false;
        for (const auto& input_ref : ctx.inputs) {
            if (input_ref.get().requires_grad()) {
                requires_grad = true;
                break;
            }
        }

        // Lazy Allocation: If no inputs require grad, do not create node
        if (!requires_grad) {
            return;
        }

        // Step 2: Create the corresponding backward node
        std::shared_ptr<Node> node;

        switch (ctx.type) {
            case OpType::Add:
                node = std::make_shared<AddBackward>();
                break;
            case OpType::Sub:
                node = std::make_shared<SubBackward>();
                break;
            case OpType::Mul:
                node = std::make_shared<MulBackward>(ctx.inputs[0].get(), ctx.inputs[1].get());
                break;
            case OpType::Div:
                node = std::make_shared<DivBackward>(ctx.inputs[0].get(), ctx.inputs[1].get());
                break;
            case OpType::MatMul:
                node = std::make_shared<MatMulBackward>(ctx.inputs[0].get(), ctx.inputs[1].get());
                break;
            case OpType::Neg:
                node = std::make_shared<NegBackward>();
                break;
            case OpType::Sum:
                node = std::make_shared<SumBackward>(ctx.inputs[0].get().shape());
                break;
            case OpType::Mean:
                node = std::make_shared<MeanBackward>(ctx.inputs[0].get().shape());
                break;
            default:
                // View operations or unsupported ops will be ignored.
                // In a complete framework, these would also require backward nodes.
                return;
        }

        // Step 3: Link edges to parents' grad_fns
        for (const auto& input_ref : ctx.inputs) {
            const Tensor& t = input_ref.get();
            if (t.requires_grad()) {
                auto meta = static_cast<AutogradMeta*>(t.impl()->autograd_meta());
                if (meta) {
                    if (meta->grad_fn()) {
                        node->add_next_edge(meta->grad_fn());
                    } else {
                        node->add_next_edge(meta->grad_accumulator());
                    }
                } else {
                    node->add_next_edge(nullptr);
                }
            } else {
                node->add_next_edge(nullptr);
            }
        }

        // Step 4: Mark output tensor as requires_grad and set its grad_fn
        ctx.out.set_requires_grad(true);
        auto out_meta = static_cast<AutogradMeta*>(ctx.out.impl()->autograd_meta());
        if (out_meta) {
            out_meta->set_grad_fn(node);
        }
    }

}  // namespace helix
