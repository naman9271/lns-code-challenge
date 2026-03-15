// Challenge 6: Unified matrix multiplication demonstrating the #ifdef pattern
// used for ggml-cpu backend modification.
//
// This is a standalone prototype of how ggml-cpu's vec_dot / MUL_MAT would
// be conditionally compiled to use LNS arithmetic instead of FP SIMD.
//
// Compile variants:
//   g++ -o matmul_fp  matmul_unified.cpp -lm                          # FP32 baseline
//   g++ -o matmul_x32 matmul_unified.cpp -lm -DGGML_XLNS32           # 32-bit LNS
//   g++ -o matmul_x16 matmul_unified.cpp -lm -DGGML_XLNS16           # 16-bit LNS
//
// This mirrors the ggml build system pattern:
//   cmake -B build -DGGML_XLNS16=ON   # passes -DGGML_XLNS16 to compiler
//   cmake -B build -DGGML_XLNS32=ON   # passes -DGGML_XLNS32 to compiler

#include <cstdio>

#ifdef GGML_XLNS16
  #define xlns16_ideal
  #include "../xlnscpp/xlns16.cpp"
#elif defined(GGML_XLNS32)
  #define xlns32_ideal
  #include "../xlnscpp/xlns32.cpp"
#endif

// ---- Matrix multiply: C = A * B^T ----
// This function mirrors ggml_vec_dot_f32 pattern:
//   - float* interface (same as ggml tensors)
//   - #ifdef selects LNS or FP path
//   - LNS path uses overloaded operators for sum-of-products
//   - In ggml, the #else branch contains 120+ lines of AVX2/ARM NEON/SVE SIMD
void matmul(const float *A, const float *B, float *C,
            int rows_A, int cols_A, int rows_B) {
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < rows_B; j++) {

#ifdef GGML_XLNS16
            // --- 16-bit LNS arithmetic path ---
            // Mirrors: ggml_vec_dot_f32 with xlns16
            xlns16_float sum;
            sum = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                xlns16_float a_val, b_val;
                a_val = A[i * cols_A + k];  // float -> LNS (overloaded =)
                b_val = B[j * cols_A + k];  // float -> LNS (overloaded =)
                sum = sum + a_val * b_val;  // LNS mul (*) then LNS add (+)
            }
            C[i * rows_B + j] = xlns16_2float(sum);  // LNS -> float

#elif defined(GGML_XLNS32)
            // --- 32-bit LNS arithmetic path ---
            // Mirrors: ggml_vec_dot_f32 with xlns32
            xlns32_float sum;
            sum = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                xlns32_float a_val, b_val;
                a_val = A[i * cols_A + k];
                b_val = B[j * cols_A + k];
                sum = sum + a_val * b_val;
            }
            C[i * rows_B + j] = xlns32_2float(sum);

#else
            // --- Standard FP32 arithmetic path ---
            // In ggml, this contains AVX2/ARM NEON/SVE SIMD code.
            // Here we use simple scalar FP for clarity.
            float sum = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                sum += A[i * cols_A + k] * B[j * cols_A + k];
            }
            C[i * rows_B + j] = sum;
#endif

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

    // Identify which mode we're running
#ifdef GGML_XLNS16
    const char *mode = "GGML_XLNS16 (16-bit LNS)";
#elif defined(GGML_XLNS32)
    const char *mode = "GGML_XLNS32 (32-bit LNS)";
#else
    const char *mode = "FP32 (standard float)";
#endif

    float result[rows_A * rows_B];
    matmul(matrix_A, matrix_B, result, rows_A, cols_A, rows_B);

    printf("Mode: %s\n", mode);
    printf("mul mat (%d x %d) (transposed result):\n[", rows_B, rows_A);
    for (int j = 0; j < rows_B; j++) {
        if (j > 0) printf("\n");
        for (int i = 0; i < rows_A; i++) {
            printf(" %.2f", result[i * rows_B + j]);
        }
    }
    printf(" ]\n");

    return 0;
}
