#include "autograd/graph_builder.hpp"

#include "autograd/autograd_meta.hpp"
#include "autograd/function.hpp"

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
                node = std::make_shared<AddBackward>(ctx.inputs[0].get().shape(), ctx.inputs[1].get().shape());
                break;
            case OpType::Sub:
                node = std::make_shared<SubBackward>(ctx.inputs[0].get().shape(), ctx.inputs[1].get().shape());
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
            case OpType::Exp:
                node = std::make_shared<ExpBackward>(ctx.out);
                break;
            case OpType::Tanh:
                node = std::make_shared<TanhBackward>(ctx.out);
                break;
            case OpType::Log:
                node = std::make_shared<LogBackward>(ctx.inputs[0].get());
                break;
            case OpType::Sqrt:
                node = std::make_shared<SqrtBackward>(ctx.out);
                break;
            case OpType::Pow:
                node = std::make_shared<PowBackward>(
                    ctx.inputs[0].get(), std::any_cast<float>(ctx.attributes.at("exponent"))
                );
                break;
            case OpType::ReLU:
                node = std::make_shared<ReLUBackward>(ctx.inputs[0].get());
                break;
            case OpType::Sum: {
                auto axis = std::any_cast<std::optional<size_t>>(ctx.attributes.at("axis"));
                auto keepdim = std::any_cast<bool>(ctx.attributes.at("keepdim"));
                node = std::make_shared<SumBackward>(ctx.inputs[0].get().shape(), axis, keepdim);
                break;
            }
            case OpType::Mean: {
                auto axis = std::any_cast<std::optional<size_t>>(ctx.attributes.at("axis"));
                auto keepdim = std::any_cast<bool>(ctx.attributes.at("keepdim"));
                node = std::make_shared<MeanBackward>(ctx.inputs[0].get().shape(), axis, keepdim);
                break;
            }
            case OpType::CrossEntropy: {
                auto log_softmax = std::any_cast<Tensor>(ctx.attributes.at("log_softmax"));
                node = std::make_shared<CrossEntropyLossBackward>(log_softmax, ctx.inputs[1].get());
                break;
            }

            case OpType::Clone:
                node = std::make_shared<CloneBackward>();
                break;
            case OpType::View: {
                auto original_shape = std::any_cast<Shape>(ctx.attributes.at("original_shape"));
                node = std::make_shared<ViewBackward>(original_shape);
                break;
            }
            case OpType::Slice: {
                auto dim = std::any_cast<size_t>(ctx.attributes.at("dim"));
                auto start = std::any_cast<size_t>(ctx.attributes.at("start"));
                auto end = std::any_cast<size_t>(ctx.attributes.at("end"));
                auto input_shape = std::any_cast<Shape>(ctx.attributes.at("input_shape"));
                node = std::make_shared<SliceBackward>(input_shape, dim, start, end);
                break;
            }
            case OpType::Transpose: {
                auto dim0 = std::any_cast<size_t>(ctx.attributes.at("dim0"));
                auto dim1 = std::any_cast<size_t>(ctx.attributes.at("dim1"));
                node = std::make_shared<TransposeBackward>(dim0, dim1);
                break;
            }
            case OpType::BroadcastTo: {
                auto input_shape = std::any_cast<Shape>(ctx.attributes.at("input_shape"));
                node = std::make_shared<BroadcastToBackward>(input_shape);
                break;
            }
            default:
                // Unsupported ops will be ignored.
                // In a complete framework, these would also require backward nodes.
                return;
        }

        // Step 3: Link edges to parents' grad_fns
        for (const auto& input_ref : ctx.inputs) {
            const Tensor& t = input_ref.get();
            if (t.requires_grad()) {
                const auto meta = t.impl()->autograd_meta();
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
        const auto out_meta = ctx.out.impl()->autograd_meta();
        if (out_meta) {
            out_meta->set_grad_fn(node);
        }
    }

}  // namespace helix
