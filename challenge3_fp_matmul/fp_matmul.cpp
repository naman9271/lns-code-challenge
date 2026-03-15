// Challenge 3: Standalone FP32 matrix multiplication
// Uses the same matrix data as the ggml blog example
// Computes C = A * B^T using nested for loops

#include <cstdio>

// Matrix multiply: C = A * B^T
// A is (rows_A x cols_A), B is (rows_B x cols_B), cols_A == cols_B
// Result C is (rows_A x rows_B)
void matmul(const float *A, const float *B, float *C,
            int rows_A, int cols_A, int rows_B) {
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < rows_B; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                sum += A[i * cols_A + k] * B[j * cols_A + k];
            }
            C[i * rows_B + j] = sum;
        }
    }
}

int main(void) {
    // Same matrix data as ggml blog example
    const int rows_A = 4, cols_A = 2;
    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };

    const int rows_B = 3, cols_B = 2;
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    // Result C = A * B^T  => (4 x 3)
    float result[rows_A * rows_B];

    matmul(matrix_A, matrix_B, result, rows_A, cols_A, rows_B);

    // Print in transposed format to match ggml output
    printf("FP32 mul mat (%d x %d) (transposed result):\n[", rows_B, rows_A);
    for (int j = 0; j < rows_B; j++) {
        if (j > 0) printf("\n");
        for (int i = 0; i < rows_A; i++) {
            printf(" %.2f", result[i * rows_B + j]);
        }
    }
    printf(" ]\n");

    return 0;
}
