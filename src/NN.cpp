#include <cstdlib>
#include <random>
#include <algorithm>
#include <cmath>
#include "NN.hpp"

// Вспомогательный оператор для вычитания векторов
std::vector<double> operator-(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs) {
    assert(lhs.size() == rhs.size(),
        "std::vector::operator-: Inconsistent size", 2);
    std::vector<double> result(lhs.size());
    for (unsigned int i = 0; i < lhs.size(); i++)
        result[i] = lhs[i] - rhs[i];
    return result;
}

NeuralNetwork::NeuralNetwork() {
    weights1 = weight_init(2.0, HIDDEN_SIZE, INPUT_SIZE + 1);
    weights2 = weight_init(2.0, OUTPUT_SIZE, HIDDEN_SIZE + 1);
}

void NeuralNetwork::train(
    const unsigned int iterations,
    const Matrix<unsigned char>& images,
    const Matrix<unsigned char>& labels) {
    const double alpha = 1.5;

    for (unsigned int i = 0; i < iterations; ++i) {
        Matrix<double> gradient_1(weights1.rows(), weights1.cols(), 0.0);
        Matrix<double> gradient_2(weights2.rows(), weights2.cols(), 0.0);
        double cost = 0.0;

        compute_gradients_and_cost(images, labels, gradient_1, gradient_2, cost);

        printf("Cost after %d iteration(s): %f\n", i + 1, cost);

        weights1 = weights1 - gradient_1 * alpha;
        weights2 = weights2 - gradient_2 * alpha;
    }
}

std::vector<double> vectorize_label(unsigned char label) {
    std::vector<double> result(10, 0.0);
    result[(unsigned int)label] = 1.0;
    return result;
}

std::vector<double> log(const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (unsigned int i = 0; i < result.size(); ++i) {
        double v = std::max(vec[i], 1e-12); // защита от log(0)
        result[i] = std::log(v);
    }
    return result;
}

void NeuralNetwork::compute_gradients_and_cost(
    const Matrix<unsigned char>& images,
    const Matrix<unsigned char>& labels,
    Matrix<double>& gradient_1,
    Matrix<double>& gradient_2,
    double& cost) {

    unsigned int m = images.rows();
    const double lambda = 1.0;

    for (unsigned int i = 0; i < m; ++i) {
        std::vector<double> first_layer(images[i].begin(), images[i].end());
        first_layer.insert(first_layer.begin(), 1.0);

        std::vector<double> hidden_layer(HIDDEN_SIZE), last_layer(OUTPUT_SIZE);

        hidden_layer = feed_forward(first_layer, weights1);
        hidden_layer.insert(hidden_layer.begin(), 1.0);

        last_layer = feed_forward(hidden_layer, weights2);

        const std::vector<double> vector_outcome = vectorize_label(labels[i][0]);
        const std::vector<double> ones(10, 1.0);

        const double first_part = ((Matrix<double>(vector_outcome) * (double)(-1)).transpose() * log(last_layer))[0];
        const double second_part = ((Matrix<double>(ones - vector_outcome)).transpose() * log(ones - last_layer))[0];

        cost += 1.0 / m * (first_part - second_part);

        const Matrix<double> d3(last_layer - vector_outcome);
        const std::vector<double> ones2(HIDDEN_SIZE + 1, 1);
        Matrix<double> d2((weights2.transpose() * d3)
            .hadamard(Matrix<double>(hidden_layer))
            .hadamard(Matrix<double>(ones2 - hidden_layer)));

        gradient_2 += d3 * Matrix<double>(hidden_layer).transpose();

        std::vector<double> d2_vec(HIDDEN_SIZE);
        for (unsigned int i = 0; i < HIDDEN_SIZE; ++i) d2_vec[i] = d2[i + 1][0];

        gradient_1 += Matrix<double>(d2_vec) * Matrix<double>(first_layer).transpose();
    }

    Matrix<double> temp_weights1(weights1);
    for (unsigned int i = 0; i < temp_weights1.rows(); ++i) {
        temp_weights1[i][0] = 0.0;
    }
    Matrix<double> temp_weights2(weights2);
    for (unsigned int i = 0; i < temp_weights2.rows(); ++i) {
        temp_weights2[i][0] = 0.0;
    }

    gradient_1 = gradient_1 / ((double)m) + temp_weights1 * (lambda / m);
    gradient_2 = gradient_2 / ((double)m) + temp_weights2 * (lambda / m);

    double regularizationCost = 0.0;
    for (unsigned int i = 0; i < weights1.rows(); ++i) {
        for (unsigned int j = 1; j < weights1.cols(); ++j) {
            regularizationCost += weights1[i][j] * weights1[i][j];
        }
    }
    for (unsigned int i = 0; i < weights2.rows(); ++i) {
        for (unsigned int j = 1; j < weights2.cols(); ++j) {
            regularizationCost += weights2[i][j] * weights2[i][j];
        }
    }

    cost += lambda / (2 * m) * regularizationCost;
}

inline std::vector<double> NeuralNetwork::feed_forward(
    const std::vector<double>& input,
    const Matrix<double>& weights) {
#ifdef LEAKY_RELU
    return leaky_relu(weights * input, 0.01);
#elif defined(PERS)
    return bent_identity(weights * input);
#else
    return sigmoid(weights * input);
#endif
}

Matrix<double> NeuralNetwork::weight_init(double maxWeight, unsigned int rows, unsigned int cols) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-maxWeight, maxWeight);

    Matrix<double> weights(rows, cols);
    for (int i = 0; i < weights.rows(); i++)
        for (int j = 0; j < weights.cols(); j++)
            weights[i][j] = dist(e2);

    return weights;
}

unsigned int NeuralNetwork::compute(const Example& e) {
    std::vector<double> first_layer(e.data, e.data + INPUT_SIZE);
    first_layer.insert(first_layer.begin(), 1.0);

    std::vector<double> hidden_layer(HIDDEN_SIZE), last_layer(OUTPUT_SIZE);

    hidden_layer = feed_forward(first_layer, weights1);
    hidden_layer.insert(hidden_layer.begin(), 1.0);
    last_layer = feed_forward(hidden_layer, weights2);

    unsigned int max_val_index = 0;
    for (int i = 1; i < 10; i++) {
        if (last_layer[i] > last_layer[max_val_index])
            max_val_index = i;
    }
    return max_val_index;
}

std::vector<double> NeuralNetwork::sigmoid(const std::vector<double>& x) {
    std::vector<double> result(x
