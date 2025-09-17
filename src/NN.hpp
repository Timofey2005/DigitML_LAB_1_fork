#ifndef NN_HPP
#define NN_HPP

#define INPUT_SIZE  (28 * 28)
#define HIDDEN_SIZE 15
#define OUTPUT_SIZE 10

#include "../lib/matrix.h"
#include "dataset.hpp"
#include <cmath>
#include <vector>
#include <functional>
#include <stdexcept>

class NeuralNetwork {
private:
    // Матрицы весов с учётом bias (+1 к количеству входов)
    Matrix<double> weights1 = Matrix<double>(HIDDEN_SIZE, INPUT_SIZE + 1);
    Matrix<double> weights2 = Matrix<double>(OUTPUT_SIZE, HIDDEN_SIZE + 1);

    Matrix<double> weight_init(double max_weight, unsigned int rows, unsigned int cols);

    // Функции активации и их производные (по умолчанию — Leaky ReLU)
    std::function<std::vector<double>(const std::vector<double>&)> activation;
    std::function<std::vector<double>(const std::vector<double>&)> activation_prime;

    std::vector<double> feed_forward(const std::vector<double>& input,
                                     const Matrix<double>& weights);

public:
    // Конструктор по умолчанию — Leaky ReLU
    NeuralNetwork();

    // Конструктор с выбором активации
    NeuralNetwork(std::function<std::vector<double>(const std::vector<double>&)> act,
                  std::function<std::vector<double>(const std::vector<double>&)> act_prime);

    NeuralNetwork(const NeuralNetwork& rhs) = default;
    virtual ~NeuralNetwork() = default;

    void train(const unsigned int iterations,
               const Matrix<unsigned char>& images,
               const Matrix<unsigned char>& labels);

    void compute_gradients_and_cost(const Matrix<unsigned char>& images,
                                    const Matrix<unsigned char>& labels,
                                    Matrix<double>& gradient_1,
                                    Matrix<double>& gradient_2,
                                    double& cost);

    unsigned int compute(const Example& e);

    // Доступные функции активации
    static std::vector<double> sigmoid(const std::vector<double>& x);
    static std::vector<double> bent_identity(const std::vector<double>& x);
    static std::vector<double> sigmoid_prime(const std::vector<double>& x);
    static std::vector<double> leaky_relu(const std::vector<double>& x);
    static std::vector<double> leaky_relu_prime(const std::vector<double>& x);
};

#endif // NN_HPP
