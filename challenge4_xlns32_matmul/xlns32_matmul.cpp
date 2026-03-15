// Challenge 4: Matrix multiplication using 32-bit LNS (xlns32) internally
// The function signature and main program use 32-bit FP;
// the conversion to/from LNS is handled by overloaded operators.
// The sum of products is computed entirely in LNS (not FP).

#include <cstdio>

// Define xlns32_ideal for ideal sb/db (no lookup table approximation)
#define xlns32_ideal
#include "../xlnscpp/xlns32.cpp"

// Matrix multiply: C = A * B^T
// The function signature uses float, but internally xlns32_float
// performs all arithmetic in LNS. The overloaded assignment operator
// automatically converts float -> LNS on input, and xlns32_2float()
// converts LNS -> float on output.
void matmul(const float *A, const float *B, float *C,
            int rows_A, int cols_A, int rows_B) {
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < rows_B; j++) {
            xlns32_float sum;
            sum = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                xlns32_float a_val, b_val;
                a_val = A[i * cols_A + k];  // float -> LNS
                b_val = B[j * cols_A + k];  // float -> LNS
                sum = sum + a_val * b_val;  // LNS * and +
            }
            C[i * rows_B + j] = xlns32_2float(sum);  // LNS -> float
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

    float result[rows_A * rows_B];
    matmul(matrix_A, matrix_B, result, rows_A, cols_A, rows_B);

    printf("xlns32 LNS mul mat (%d x %d) (transposed result):\n[", rows_B, rows_A);
    for (int j = 0; j < rows_B; j++) {
        if (j > 0) printf("\n");
        for (int i = 0; i < rows_A; i++) {
            printf(" %.2f", result[i * rows_B + j]);
        }
    }
    printf(" ]\n");

    return 0;
}
