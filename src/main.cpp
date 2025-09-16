#include "dataset.hpp"
#include "NN.hpp"
#include "../lib/matrix.h"
#include <vector>
#include <iostream>

void debug(Example e) {
    static std::string shades = " .:-=+*#%@";
    for (unsigned int i = 0; i < 28 * 28; i++) {
        if (i % 28 == 0) printf("\n");
        printf("%c", shades[e.data[i] / 30]);
    }
    printf("\nLabel: %d\n", e.label);
}

std::vector<double> load_matrix(Example& e) {
    return std::vector<double>(e.data, e.data + 28 * 28);
}

const double calculate_accuracy(const Matrix<unsigned char>& images,
                                 const Matrix<unsigned char>& labels,
                                 const NeuralNetwork& n) {
    unsigned int correct = 0;
    for (unsigned int i = 0; i < images.rows(); ++i) {
        Example e;
        for (int j = 0; j < 28 * 28; ++j) {
            e.data[j] = images[i][j];
        }
        e.label = labels[i][0];
        unsigned int guess = n.compute(e);
        if (guess == static_cast<unsigned int>(e.label)) correct++;
    }
    return static_cast<double>(correct) / images.rows();
}

#ifdef TESTS
#include <gtest/gtest.h>

NeuralNetwork n;

TEST(FunctionTesting, test_bent_identity) {
    std::vector<double> t1 = {0};
    EXPECT_NEAR(n.bent_identity(t1)[0], 0.0, 1e-4);
}

TEST(FunctionTesting, test_sigmoid_incr) {
    std::vector<double> t1 = {-10, 0, 10};
    std::vector<double> expected = {n.sigmoid({-10})[0],
                                    n.sigmoid({0})[0],
                                    n.sigmoid({10})[0]};
    auto result = n.sigmoid(t1);
    for (size_t i = 0; i < t1.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-9);
    }
}

TEST(FunctionTesting, test_sigmoid_decr) {
    Matrix<unsigned char> images_test(0, 0);
    Matrix<unsigned char> labels_test(0, 0);
    load_dataset(images_test, labels_test,
                 "data/t10k-images-idx3-ubyte",
                 "data/t10k-labels-idx1-ubyte");
    EXPECT_GT(calculate_accuracy(images_test, labels_test, n), 0.01);
}

TEST(FunctionTesting, test_throw) {
    const unsigned int num_iterations = 5;
    Matrix<unsigned char> images_train(0, 0);
    Matrix<unsigned char> labels_train(0, 0);
    load_dataset(images_train, labels_train,
                 "data/train-images-idx3-ubyte",
                 "data/train-labels-idx1-ubyte");
    EXPECT_NO_THROW(n.train(num_iterations, images_train, labels_train));
}

TEST(FunctionTesting, test_sigmoid_comp) {
    std::vector<double> t1 = {-10};
    EXPECT_GT(n.sigmoid(t1)[0], n.bent_identity(t1)[0]);
}

TEST(FunctionTesting, test_leaky_relu_positive) {
    std::vector<double> t = {5.0};
    EXPECT_DOUBLE_EQ(n.leaky_relu(t)[0], 5.0);
}

TEST(FunctionTesting, test_leaky_relu_negative) {
    std::vector<double> t = {-3.0};
    EXPECT_DOUBLE_EQ(n.leaky_relu(t)[0], -0.03);
}

TEST(FunctionTesting, test_leaky_relu_prime_negative) {
    std::vector<double> t = {-3.0};
    EXPECT_DOUBLE_EQ(n.leaky_relu_prime(t)[0], 0.01);
}

TEST(FunctionTesting, test_leaky_relu_prime_zero) {
    std::vector<double> t = {0.0};
    EXPECT_DOUBLE_EQ(n.leaky_relu_prime(t)[0], 1.0);
}

TEST(FunctionTesting, test_leaky_relu_prime_positive) {
    std::vector<double> t = {3.0};
    EXPECT_DOUBLE_EQ(n.leaky_relu_prime(t)[0], 1.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#else // обычный режим

int main() {
    Matrix<unsigned char> images_train(0, 0);
    Matrix<unsigned char> labels_train(0, 0);
    load_dataset(images_train, labels_train,
                 "data/train-images-idx3-ubyte",
                 "data/train-labels-idx1-ubyte");

    Matrix<unsigned char> images_test(0, 0);
    Matrix<unsigned char> labels_test(0, 0);
    load_dataset(images_test, labels_test,
                 "data/t10k-images-idx3-ubyte",
                 "data/t10k-labels-idx1-ubyte");

    NeuralNetwork n;
    const unsigned int num_iterations = 5;
    n.train(num_iterations, images_train, labels_train);

    const double accuracy_train = calculate_accuracy(images_train, labels_train, n);
    const double accuracy_test = calculate_accuracy(images_test, labels_test, n);

    printf("Accuracy on training data: %f\n", accuracy_train);
    printf("Accuracy on test data: %f\n", accuracy_test);

    return 0;
}

#endif
