#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "helix.hpp"

using namespace helix;

// Helper function to swap endianness for 32-bit integers
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) | ((val << 8) & 0x00ff0000) | ((val >> 8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

// Read MNIST Images
Tensor read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    uint32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic = swap_endian(magic);
    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number.");
    }

    size_t image_size = rows * cols;
    std::vector<unsigned char> raw_data(num_images * image_size);
    file.read(reinterpret_cast<char*>(raw_data.data()), num_images * image_size);

    std::vector<float> data(num_images * image_size);
    for (size_t i = 0; i < raw_data.size(); ++i) {
        data[i] = static_cast<float>(raw_data[i]) / 255.0f;
    }

    return Tensor(data, {static_cast<size_t>(num_images), image_size});
}

// Read MNIST Labels
std::pair<Tensor, std::vector<uint8_t>> read_mnist_labels(const std::string& path, size_t num_classes = 10) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic = swap_endian(magic);
    num_labels = swap_endian(num_labels);

    if (magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number.");
    }

    std::vector<uint8_t> raw_labels(num_labels);
    file.read(reinterpret_cast<char*>(raw_labels.data()), num_labels);

    std::vector<float> data(num_labels * num_classes, 0.0f);
    for (size_t i = 0; i < num_labels; ++i) {
        data[i * num_classes + raw_labels[i]] = 1.0f;
    }

    return {Tensor(data, {static_cast<size_t>(num_labels), num_classes}), raw_labels};
}

int main() {
    init_autograd();
    std::cout << "Loading MNIST dataset..." << std::endl;

    std::string train_images_path = "data/mnist/train-images-idx3-ubyte";
    std::string train_labels_path = "data/mnist/train-labels-idx1-ubyte";
    std::string test_images_path = "data/mnist/t10k-images-idx3-ubyte";
    std::string test_labels_path = "data/mnist/t10k-labels-idx1-ubyte";

    Tensor X_train, Y_train_onehot;
    std::vector<uint8_t> Y_train_raw;
    Tensor X_test, Y_test_onehot;
    std::vector<uint8_t> Y_test_raw;

    try {
        X_train = read_mnist_images(train_images_path);
        auto train_labels = read_mnist_labels(train_labels_path);
        Y_train_onehot = train_labels.first;
        Y_train_raw = train_labels.second;

        X_test = read_mnist_images(test_images_path);
        auto test_labels = read_mnist_labels(test_labels_path);
        Y_test_onehot = test_labels.first;
        Y_test_raw = test_labels.second;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load data: " << e.what() << std::endl;
        std::cerr << "Please run scripts/download_mnist.sh first." << std::endl;
        return 1;
    }

    std::cout << "Train data: " << X_train.shape()[0] << " samples" << std::endl;
    std::cout << "Test data: " << X_test.shape()[0] << " samples" << std::endl;

    // Define model: 784 -> 128 -> ReLU -> 10
    Sequential model(Linear(784, 128), ReLU(), Linear(128, 10));

    // Hyperparameters
    size_t num_epochs = 10;
    size_t batch_size = 64;
    float learning_rate = 0.05f;

    SGD optimizer(model.parameters(), learning_rate);
    size_t num_train_samples = X_train.shape()[0];
    size_t num_batches = num_train_samples / batch_size;

    std::cout << "\nStarting training..." << std::endl;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;
        size_t correct_preds = 0;

        for (size_t i = 0; i < num_batches; ++i) {
            size_t start_idx = i * batch_size;
            size_t end_idx = start_idx + batch_size;

            Tensor X_batch = X_train.slice(0, start_idx, end_idx);
            Tensor Y_batch = Y_train_onehot.slice(0, start_idx, end_idx);

            optimizer.zero_grad();
            Tensor preds = model(X_batch);
            Tensor loss = cross_entropy_loss(preds, Y_batch);

            loss.backward();
            optimizer.step();

            total_loss += loss.item();

            // Calculate accuracy for current batch
            const float* pred_data = preds.data_ptr();
            for (size_t b = 0; b < batch_size; ++b) {
                size_t true_label = Y_train_raw[start_idx + b];

                float max_val = pred_data[b * 10];
                size_t max_idx = 0;
                for (size_t c = 1; c < 10; ++c) {
                    if (pred_data[b * 10 + c] > max_val) {
                        max_val = pred_data[b * 10 + c];
                        max_idx = c;
                    }
                }

                if (max_idx == true_label) correct_preds++;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        float avg_loss = total_loss / num_batches;
        float accuracy = static_cast<float>(correct_preds) / (num_batches * batch_size) * 100.0f;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] " << "Loss: " << std::fixed
                  << std::setprecision(4) << avg_loss << " - " << "Acc: " << std::fixed << std::setprecision(2)
                  << accuracy << "% - " << "Time: " << std::fixed << std::setprecision(2) << duration.count() << "s"
                  << std::endl;
    }

    std::cout << "\nEvaluating on test set..." << std::endl;

    // Evaluate in batches to prevent allocating too much memory if needed,
    // but 10,000 samples fit easily in RAM. Let's process the whole test set at once.
    Tensor test_preds = model(X_test);
    Tensor test_loss = cross_entropy_loss(test_preds, Y_test_onehot);

    size_t correct_test_preds = 0;
    size_t num_test_samples = X_test.shape()[0];
    const float* test_pred_data = test_preds.data_ptr();

    for (size_t b = 0; b < num_test_samples; ++b) {
        size_t true_label = Y_test_raw[b];

        float max_val = test_pred_data[b * 10];
        size_t max_idx = 0;
        for (size_t c = 1; c < 10; ++c) {
            if (test_pred_data[b * 10 + c] > max_val) {
                max_val = test_pred_data[b * 10 + c];
                max_idx = c;
            }
        }

        if (max_idx == true_label) correct_test_preds++;
    }

    float test_accuracy = static_cast<float>(correct_test_preds) / num_test_samples * 100.0f;
    std::cout << "Test Loss: " << std::fixed << std::setprecision(4) << test_loss.item() << std::endl;
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << test_accuracy << "%" << std::endl;

    return 0;
}
