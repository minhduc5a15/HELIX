#include "autograd/engine.hpp"

#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "autograd/autograd_meta.hpp"
#include "autograd/graph_builder.hpp"
#include "autograd/node.hpp"
#include "core/dispatcher.hpp"

namespace helix {

    // Definition of AccumulateGrad Node
    class AccumulateGrad : public Node {
    public:
        AccumulateGrad(AutogradMeta* meta) : meta_(meta) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override {
            if (!meta_->has_grad()) {
                meta_->set_grad(grad_outputs[0]);
            } else {
                meta_->set_grad(meta_->grad() + grad_outputs[0]);
            }
            return {};
        }

    private:
        AutogradMeta* meta_;
    };

    // AutogradEngineProvider Implementation
    AutogradMeta* AutogradEngineProvider::create_meta() {
        auto meta = new AutogradMeta(true);
        meta->set_grad_accumulator(std::make_shared<AccumulateGrad>(meta));
        return meta;
    }

    void AutogradEngineProvider::backward(Tensor& tensor, const std::vector<Tensor>& grad_outputs) {
        engine_.run(tensor, grad_outputs);
    }

    Tensor& AutogradEngineProvider::get_grad(const Tensor& tensor) {
        auto meta = static_cast<AutogradMeta*>(tensor.impl()->autograd_meta());
        return meta->grad();
    }

    const Tensor& AutogradEngineProvider::get_grad(const Tensor& tensor) const {
        auto meta = static_cast<AutogradMeta*>(tensor.impl()->autograd_meta());
        return meta->grad();
    }

    bool AutogradEngineProvider::has_grad(const Tensor& tensor) const {
        auto meta = static_cast<AutogradMeta*>(tensor.impl()->autograd_meta());
        return meta && meta->has_grad();
    }

    struct NoGradGuard {
        NoGradGuard() {
            prev_builder_ = Dispatcher::get_graph_builder();
            Dispatcher::register_graph_builder(nullptr);
        }
        ~NoGradGuard() { Dispatcher::register_graph_builder(prev_builder_); }

    private:
        GraphBuilderInterface* prev_builder_;
    };

    // BackwardEngine Implementation
    void BackwardEngine::run(Tensor& target, const std::vector<Tensor>& grad_outputs) {
        auto meta = static_cast<AutogradMeta*>(target.impl()->autograd_meta());
        if (!meta || !meta->requires_grad() || (!meta->grad_fn() && !meta->grad_accumulator())) {
            throw std::runtime_error("Cannot run backward on this tensor.");
        }

        NoGradGuard guard;

        auto root = meta->grad_fn() ? meta->grad_fn() : meta->grad_accumulator();

        // Step 1: Topological Sort (compute in-degrees for reverse traversal)
        std::unordered_map<Node*, int> in_degrees;
        std::queue<Node*> queue;
        std::unordered_set<Node*> visited;

        std::vector<Node*> nodes_to_process;
        nodes_to_process.push_back(root.get());
        visited.insert(root.get());

        size_t head = 0;
        while (head < nodes_to_process.size()) {
            Node* curr = nodes_to_process[head++];
            for (const auto& next_node_ptr : curr->next_edges()) {
                if (next_node_ptr) {
                    Node* next = next_node_ptr.get();
                    in_degrees[next]++;
                    if (!visited.contains(next)) {
                        visited.insert(next);
                        nodes_to_process.push_back(next);
                    }
                }
            }
        }

        // Step 2: Initialize gradients queue
        std::unordered_map<Node*, std::vector<Tensor>> node_gradients;

        if (grad_outputs.empty()) {
            if (target.numel() != 1) {
                throw std::runtime_error("grad can be implicitly created only for scalar outputs");
            }
            std::vector<float> data = {1.0f};
            node_gradients[root.get()] = {Tensor(data, target.shape())};
        } else {
            node_gradients[root.get()] = grad_outputs;
        }

        std::queue<Node*> ready_queue;
        ready_queue.push(root.get());

        while (!ready_queue.empty()) {
            Node* curr = ready_queue.front();
            ready_queue.pop();

            auto current_grads = node_gradients[curr];
            auto input_grads = curr->backward(current_grads);

            const auto& next_edges = curr->next_edges();
            if (input_grads.size() != next_edges.size()) {
                throw std::runtime_error("Mismatch between generated gradients and number of inputs.");
            }

            for (size_t i = 0; i < next_edges.size(); ++i) {
                if (next_edges[i]) {
                    Node* next = next_edges[i].get();

                    if (!node_gradients.contains(next)) {
                        node_gradients[next] = {input_grads[i]};
                    } else {
                        // Assuming single output per node
                        node_gradients[next][0] = node_gradients[next][0] + input_grads[i];
                    }

                    in_degrees[next]--;
                    if (in_degrees[next] == 0) {
                        ready_queue.push(next);
                    }
                }
            }
        }
    }

    // Initialization hook
    static AutogradEngineProvider* g_provider = nullptr;
    static AutogradGraphBuilder* g_builder = nullptr;

    void init_autograd() {
        if (!g_provider) {
            g_provider = new AutogradEngineProvider();
            register_autograd_provider(g_provider);
        }
        if (!g_builder) {
            g_builder = new AutogradGraphBuilder();
            Dispatcher::register_graph_builder(g_builder);
        }
    }

}  // namespace helix
