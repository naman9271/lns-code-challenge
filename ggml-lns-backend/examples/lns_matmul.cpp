// demo_lns_matmul.cpp — Backend Comparison Demo (FP32 CPU vs LNS)
//
// Runs the same matrix multiplication through both the standard ggml CPU
// backend (FP32) and the LNS backend, then prints side-by-side results
// with error statistics.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-lns.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <ctime>

// Matrix data (same as ggml blog tutorial)
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

// Build a compute graph for matmul: result = A × B^T

struct matmul_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    std::vector<uint8_t> buf;
};

struct ggml_cgraph * build_matmul_graph(matmul_model & model,
                                        int m_rows_A, int m_cols_A, int m_rows_B) {
    size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    model.buf.resize(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ model.buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph  * gf  = ggml_new_graph(ctx);

    // In ggml, tensor dimensions are (columns, rows)
    model.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m_cols_A, m_rows_A);
    model.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m_cols_A, m_rows_B);

    struct ggml_tensor * result = ggml_mul_mat(ctx, model.a, model.b);
    ggml_build_forward_expand(gf, result);
    ggml_free(ctx);

    return gf;
}

// Run matmul on a specific backend and return results

std::vector<float> run_matmul(ggml_backend_t backend,
                               float * data_A, float * data_B,
                               int m_rows_A, int m_cols_A, int m_rows_B) {
    matmul_model model;
    struct ggml_cgraph * gf = build_matmul_graph(model, m_rows_A, m_cols_A, m_rows_B);

    // Allocate tensors
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_gallocr_t allocr = ggml_gallocr_new(buft);
    ggml_gallocr_alloc_graph(allocr, gf);

    // Load data
    ggml_backend_tensor_set(model.a, data_A, 0, m_rows_A * m_cols_A * sizeof(float));
    ggml_backend_tensor_set(model.b, data_B, 0, m_rows_B * m_cols_A * sizeof(float));

    // Compute
    ggml_backend_graph_compute(backend, gf);

    // Get results
    struct ggml_tensor * result = ggml_graph_node(gf, -1);
    int64_t ne = ggml_nelements(result);
    std::vector<float> out(ne);
    ggml_backend_tensor_get(result, out.data(), 0, ne * sizeof(float));

    ggml_gallocr_free(allocr);
    return out;
}

// Print a matrix and compute error statistics

void print_matrix(const char * label, const float * data, int rows, int cols) {
    printf("%s (%d x %d):\n[", label, cols, rows);
    for (int j = 0; j < rows; j++) {
        if (j > 0) printf("\n ");
        for (int i = 0; i < cols; i++) {
            printf(" %8.2f", data[j * cols + i]);
        }
    }
    printf(" ]\n\n");
}

void print_error_stats(const float * ref, const float * test, int n, const char * label) {
    float sum_err = 0.0f;
    float max_err = 0.0f;

    for (int i = 0; i < n; i++) {
        float rel_err = (ref[i] != 0.0f) ? fabsf(test[i] - ref[i]) / fabsf(ref[i]) : 0.0f;
        sum_err += rel_err;
        if (rel_err > max_err) max_err = rel_err;
    }

    printf("  %s error stats:\n", label);
    printf("    Mean relative error: %.4f%%\n", (sum_err / n) * 100.0f);
    printf("    Max  relative error: %.4f%%\n", max_err * 100.0f);
    printf("\n");
}

// Generate random matrix data

void fill_random(float * data, int n, float scale) {
    for (int i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}


int main(void) {
    ggml_time_init();
    srand(42);
    printf("  ggml-LNS Backend — Matrix Multiply Comparison Demo\n");

    // ---- Initialize backends ----
    ggml_backend_load_all();
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t lns_backend = ggml_backend_lns_init();

    printf("CPU backend: %s\n", ggml_backend_name(cpu_backend));
    printf("LNS backend: %s\n", ggml_backend_name(lns_backend));
    printf("Is LNS?: %s\n\n", ggml_backend_is_lns(lns_backend) ? "yes" : "no");

    printf("  Test 1: Small 4x2 × 3x2 matrix multiply\n");

    auto cpu_result1 = run_matmul(cpu_backend, matrix_A, matrix_B, rows_A, cols_A, rows_B);
    auto lns_result1 = run_matmul(lns_backend, matrix_A, matrix_B, rows_A, cols_A, rows_B);

    print_matrix("FP32 CPU result", cpu_result1.data(), rows_A, rows_B);
    print_matrix("LNS  result    ", lns_result1.data(), rows_A, rows_B);

    // Element-by-element comparison
    printf("  Element-by-element comparison:\n");
    printf("  %-10s %-10s %-10s %-12s\n", "FP32", "LNS", "AbsErr", "RelErr");
    for (int i = 0; i < rows_A * rows_B; i++) {
        float abs_err = fabsf(lns_result1[i] - cpu_result1[i]);
        float rel_err = (cpu_result1[i] != 0.0f)
                        ? fabsf(lns_result1[i] - cpu_result1[i]) / fabsf(cpu_result1[i]) * 100.0f
                        : 0.0f;
        printf("  %-10.2f %-10.2f %-10.4f  %.4f%%\n",
               cpu_result1[i], lns_result1[i], abs_err, rel_err);
    }
    printf("\n");
    print_error_stats(cpu_result1.data(), lns_result1.data(), rows_A * rows_B, "Small matmul");

    printf("  Test 2: Random 32x16 × 32x16 matrix multiply\n");

    const int big_rows_A = 32, big_cols_A = 16, big_rows_B = 32;
    std::vector<float> big_A(big_rows_A * big_cols_A);
    std::vector<float> big_B(big_rows_B * big_cols_A);

    fill_random(big_A.data(), big_rows_A * big_cols_A, 5.0f);
    fill_random(big_B.data(), big_rows_B * big_cols_A, 5.0f);

    auto cpu_result2 = run_matmul(cpu_backend, big_A.data(), big_B.data(),
                                   big_rows_A, big_cols_A, big_rows_B);
    auto lns_result2 = run_matmul(lns_backend, big_A.data(), big_B.data(),
                                   big_rows_A, big_cols_A, big_rows_B);

    printf("  (showing first 8 elements of %d total)\n\n", big_rows_A * big_rows_B);
    printf("  %-10s %-10s %-12s\n", "FP32", "LNS", "RelErr");
    for (int i = 0; i < 8; i++) {
        float rel_err = (cpu_result2[i] != 0.0f)
                        ? fabsf(lns_result2[i] - cpu_result2[i]) / fabsf(cpu_result2[i]) * 100.0f
                        : 0.0f;
        printf("  %-10.4f %-10.4f  %.4f%%\n",
               cpu_result2[i], lns_result2[i], rel_err);
    }
    printf("\n");
    print_error_stats(cpu_result2.data(), lns_result2.data(),
                      big_rows_A * big_rows_B, "Large matmul");

    ggml_backend_free(lns_backend);
    ggml_backend_free(cpu_backend);
    return 0;
}
