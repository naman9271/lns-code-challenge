// demo_mini_transformer.cpp   Mini Transformer Block Demo (FP32 vs LNS)
//
// Runs a COMPLETE transformer layer (attention + FFN) through both the
// standard ggml CPU backend and the LNS backend, comparing outputs.
//
// This exercises ALL 8 LNS operations in the exact pattern LLMs use:
//   Attention: RMS_NORM → MUL_MAT(Q,K,V) → SCALE → SOFT_MAX → MUL_MAT → ADD
//   FFN:       RMS_NORM → MUL_MAT → SILU → MUL → MUL_MAT → ADD

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-lns.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Transformer dimensions (small but realistic proportions)

static const int SEQ_LEN = 8;     // sequence length
static const int D_MODEL = 64;    // model dimension
static const int N_HEADS = 4;     // number of attention heads
static const int D_HEAD  = D_MODEL / N_HEADS; // 16 per head
static const int D_FF    = D_MODEL * 2;       // 128 FFN hidden dim

// Random weight initialization
static void fill_random(float * data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    }
}

static void fill_ones(float * data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 1.0f;
    }
}

// Weight storage
struct transformer_weights {
    // Attention
    std::vector<float> w_q;         // (D_MODEL, D_MODEL)
    std::vector<float> w_k;         // (D_MODEL, D_MODEL)
    std::vector<float> w_v;         // (D_MODEL, D_MODEL)
    std::vector<float> w_o;         // (D_MODEL, D_MODEL)
    std::vector<float> attn_norm;   // (D_MODEL,) RMS norm weights

    // FFN
    std::vector<float> w_gate;      // (D_MODEL, D_FF)   gate projection
    std::vector<float> w_up;        // (D_MODEL, D_FF)   up projection
    std::vector<float> w_down;      // (D_FF, D_MODEL)   down projection
    std::vector<float> ffn_norm;    // (D_MODEL,) RMS norm weights

    // Input
    std::vector<float> input;       // (D_MODEL, SEQ_LEN)

    void init() {
        w_q.resize(D_MODEL * D_MODEL);
        w_k.resize(D_MODEL * D_MODEL);
        w_v.resize(D_MODEL * D_MODEL);
        w_o.resize(D_MODEL * D_MODEL);
        attn_norm.resize(D_MODEL);

        w_gate.resize(D_MODEL * D_FF);
        w_up.resize(D_MODEL * D_FF);
        w_down.resize(D_FF * D_MODEL);
        ffn_norm.resize(D_MODEL);

        input.resize(D_MODEL * SEQ_LEN);

        // Initialize with small random values
        fill_random(w_q.data(), D_MODEL * D_MODEL);
        fill_random(w_k.data(), D_MODEL * D_MODEL);
        fill_random(w_v.data(), D_MODEL * D_MODEL);
        fill_random(w_o.data(), D_MODEL * D_MODEL);
        fill_ones(attn_norm.data(), D_MODEL);  // norm weights = 1.0

        fill_random(w_gate.data(), D_MODEL * D_FF);
        fill_random(w_up.data(), D_MODEL * D_FF);
        fill_random(w_down.data(), D_FF * D_MODEL);
        fill_ones(ffn_norm.data(), D_MODEL);

        fill_random(input.data(), D_MODEL * SEQ_LEN);
    }
};

// Build the transformer block computation graph
struct transformer_model {
    // Tensor pointers for data loading
    struct ggml_tensor * t_input;
    struct ggml_tensor * t_w_q;
    struct ggml_tensor * t_w_k;
    struct ggml_tensor * t_w_v;
    struct ggml_tensor * t_w_o;
    struct ggml_tensor * t_attn_norm;
    struct ggml_tensor * t_w_gate;
    struct ggml_tensor * t_w_up;
    struct ggml_tensor * t_w_down;
    struct ggml_tensor * t_ffn_norm;
    std::vector<uint8_t> buf;
};

struct ggml_cgraph * build_transformer_graph(transformer_model & model) {
    size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    model.buf.resize(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ model.buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph  * gf  = ggml_new_graph(ctx);

    // ---- Create weight and input tensors ----
    // ggml convention: tensor_2d(ne0, ne1) means ne0 columns, ne1 rows
    // ggml_mul_mat(A, B) computes A^T × B, requires A->ne[0] == B->ne[0]
    // Result shape: (A->ne[1], B->ne[1])

    // Input: each row is a token embedding
    model.t_input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, SEQ_LEN);

    // Attention weights: project from D_MODEL to D_MODEL
    model.t_w_q       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, D_MODEL);
    model.t_w_k       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, D_MODEL);
    model.t_w_v       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, D_MODEL);
    model.t_w_o       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, D_MODEL);
    model.t_attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D_MODEL);

    // FFN weights
    model.t_w_gate    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, D_FF);
    model.t_w_up      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, D_FF);
    model.t_w_down    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_FF,    D_MODEL);
    model.t_ffn_norm  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D_MODEL);

    struct ggml_tensor * x = model.t_input;
    // x: (D_MODEL, SEQ_LEN)   ne[0]=D_MODEL, ne[1]=SEQ_LEN

    // Step 1: RMS_NORM (pre-attention normalization)
    struct ggml_tensor * x_norm = ggml_rms_norm(ctx, x, 1e-5f);
    // Step 2: MUL   apply norm weights (element-wise, with broadcast)
    x_norm = ggml_mul(ctx, x_norm, model.t_attn_norm);
    // x_norm: (D_MODEL, SEQ_LEN)

    // Step 3: MUL_MAT   Q, K, V projections
    // mul_mat(W, x) = W^T × x → (W.ne[1], x.ne[1]) = (D_MODEL, SEQ_LEN)
    struct ggml_tensor * Q = ggml_mul_mat(ctx, model.t_w_q, x_norm);
    struct ggml_tensor * K = ggml_mul_mat(ctx, model.t_w_k, x_norm);
    struct ggml_tensor * V = ggml_mul_mat(ctx, model.t_w_v, x_norm);
    // Q, K, V: all (D_MODEL, SEQ_LEN)   ne[0]=D_MODEL, ne[1]=SEQ_LEN

    // Step 4: MUL_MAT   Attention scores = Q^T × K
    // mul_mat(Q, K) = Q^T × K → (Q.ne[1], K.ne[1]) = (SEQ_LEN, SEQ_LEN)
    // But we need Q.ne[0] == K.ne[0] → D_MODEL == D_MODEL ✓
    struct ggml_tensor * scores = ggml_mul_mat(ctx, Q, K);
    // scores: (SEQ_LEN, SEQ_LEN)

    // Step 5: SCALE   divide by sqrt(d_head)
    float scale = 1.0f / sqrtf((float)D_HEAD);
    scores = ggml_scale(ctx, scores, scale);

    // Step 6: SOFT_MAX   softmax over attention scores
    scores = ggml_soft_max(ctx, scores);
    // scores: (SEQ_LEN, SEQ_LEN)   attention probabilities

    // Step 7: MUL_MAT   weighted sum: attn_out = V × scores^T
    // mul_mat(scores, V) = scores^T × V
    // scores.ne[0]=SEQ_LEN, V.ne[0]=D_MODEL → MISMATCH!
    // Instead: mul_mat(V, scores) → V^T × scores
    // V.ne[0]=D_MODEL, scores.ne[0]=SEQ_LEN → ALSO MISMATCH
    //
    // The correct approach: transpose V so its ne[0] matches scores.ne[0]
    // Or use: V has (D_MODEL, SEQ_LEN), we want output (D_MODEL, SEQ_LEN)
    // Use ggml_cont on a permuted V:
    struct ggml_tensor * Vt = ggml_cont(ctx, ggml_transpose(ctx, V));
    // Vt: (SEQ_LEN, D_MODEL)   ne[0]=SEQ_LEN, ne[1]=D_MODEL
    // mul_mat(Vt, scores) = Vt^T × scores → (D_MODEL, SEQ_LEN) ✓
    // Need Vt.ne[0] == scores.ne[0] → SEQ_LEN == SEQ_LEN ✓
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, Vt, scores);
    // attn_out: (D_MODEL, SEQ_LEN) ✓

    // Step 8: MUL_MAT   Output projection
    attn_out = ggml_mul_mat(ctx, model.t_w_o, attn_out);
    // attn_out: (D_MODEL, SEQ_LEN)

    // Step 9: ADD   Residual connection
    struct ggml_tensor * after_attn = ggml_add(ctx, x, attn_out);

    // Step 10: RMS_NORM (pre-FFN normalization)
    struct ggml_tensor * ffn_in = ggml_rms_norm(ctx, after_attn, 1e-5f);
    ffn_in = ggml_mul(ctx, ffn_in, model.t_ffn_norm);

    // Step 11: MUL_MAT   Gate and Up projections
    struct ggml_tensor * gate = ggml_mul_mat(ctx, model.t_w_gate, ffn_in);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, model.t_w_up,   ffn_in);
    // gate, up: (D_FF, SEQ_LEN)

    // Step 12: SILU   activation on gate
    gate = ggml_silu(ctx, gate);

    // Step 13: MUL   element-wise multiply gate and up
    struct ggml_tensor * ffn_hidden = ggml_mul(ctx, gate, up);

    // Step 14: MUL_MAT   Down projection
    struct ggml_tensor * ffn_out = ggml_mul_mat(ctx, model.t_w_down, ffn_hidden);
    // ffn_out: (D_MODEL, SEQ_LEN)

    // Step 15: ADD   Residual connection
    struct ggml_tensor * output = ggml_add(ctx, after_attn, ffn_out);

    // Build the forward graph
    ggml_build_forward_expand(gf, output);
    ggml_free(ctx);

    return gf;
}

// Run the transformer on a specific backend
std::vector<float> run_transformer(ggml_backend_t backend,
                                    transformer_weights & weights) {
    transformer_model model;
    struct ggml_cgraph * gf = build_transformer_graph(model);

    // Allocate
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_gallocr_t allocr = ggml_gallocr_new(buft);
    ggml_gallocr_alloc_graph(allocr, gf);

    // Load weights and input
    ggml_backend_tensor_set(model.t_input,     weights.input.data(),     0, ggml_nbytes(model.t_input));
    ggml_backend_tensor_set(model.t_w_q,       weights.w_q.data(),       0, ggml_nbytes(model.t_w_q));
    ggml_backend_tensor_set(model.t_w_k,       weights.w_k.data(),       0, ggml_nbytes(model.t_w_k));
    ggml_backend_tensor_set(model.t_w_v,       weights.w_v.data(),       0, ggml_nbytes(model.t_w_v));
    ggml_backend_tensor_set(model.t_w_o,       weights.w_o.data(),       0, ggml_nbytes(model.t_w_o));
    ggml_backend_tensor_set(model.t_attn_norm, weights.attn_norm.data(), 0, ggml_nbytes(model.t_attn_norm));
    ggml_backend_tensor_set(model.t_w_gate,    weights.w_gate.data(),    0, ggml_nbytes(model.t_w_gate));
    ggml_backend_tensor_set(model.t_w_up,      weights.w_up.data(),      0, ggml_nbytes(model.t_w_up));
    ggml_backend_tensor_set(model.t_w_down,    weights.w_down.data(),    0, ggml_nbytes(model.t_w_down));
    ggml_backend_tensor_set(model.t_ffn_norm,  weights.ffn_norm.data(),  0, ggml_nbytes(model.t_ffn_norm));

    // Compute
    ggml_backend_graph_compute(backend, gf);

    // Get output
    struct ggml_tensor * result = ggml_graph_node(gf, -1);
    int64_t ne = ggml_nelements(result);
    std::vector<float> out(ne);
    ggml_backend_tensor_get(result, out.data(), 0, ne * sizeof(float));

    ggml_gallocr_free(allocr);
    return out;
}

// Error statistics

struct error_stats {
    float mean_rel_err;
    float max_rel_err;
    float mean_abs_err;
    float max_abs_err;
    int   n;
};

error_stats compute_errors(const float * ref, const float * test, int n) {
    error_stats stats = {};
    stats.n = n;

    float sum_rel = 0.0f, sum_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_err = fabsf(test[i] - ref[i]);
        float rel_err = (fabsf(ref[i]) > 1e-8f) ? abs_err / fabsf(ref[i]) : 0.0f;
        sum_rel += rel_err;
        sum_abs += abs_err;
        if (rel_err > stats.max_rel_err) stats.max_rel_err = rel_err;
        if (abs_err > stats.max_abs_err) stats.max_abs_err = abs_err;
    }

    stats.mean_rel_err = sum_rel / n;
    stats.mean_abs_err = sum_abs / n;
    return stats;
}

int main(void) {
    ggml_time_init();
    srand(42);

    printf("══════════════════════════════════════════════════════════\n");
    printf("  Mini Transformer Block   FP32 vs LNS Backend\n");
    printf("══════════════════════════════════════════════════════════\n\n");

    printf("  Architecture:\n");
    printf("    Sequence length : %d\n", SEQ_LEN);
    printf("    Model dimension : %d\n", D_MODEL);
    printf("    Attention heads : %d (d_head=%d)\n", N_HEADS, D_HEAD);
    printf("    FFN hidden dim  : %d\n", D_FF);
    printf("\n");
    printf("  Operations used:\n");
    printf("    Attention: RMS_NORM → MUL_MAT(Q,K,V) → SCALE → SOFT_MAX → MUL_MAT → ADD\n");
    printf("    FFN:       RMS_NORM → MUL_MAT(gate,up) → SILU → MUL → MUL_MAT(down) → ADD\n");
    printf("\n");

    // ---- Initialize weights (shared between both backends) ----
    transformer_weights weights;
    weights.init();

    // ---- Initialize backends ----
    ggml_backend_load_all();
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t lns_backend = ggml_backend_lns_init();

    printf("  CPU backend: %s\n", ggml_backend_name(cpu_backend));
    printf("  LNS backend: %s  (16-bit LNS: 1 sign + 8 int + 7 frac bits)\n\n",
           ggml_backend_name(lns_backend));

    // ---- Run transformer on both backends ----
    printf("  Running transformer block...\n");

    auto cpu_output = run_transformer(cpu_backend, weights);
    auto lns_output = run_transformer(lns_backend, weights);

    int n = (int)cpu_output.size();
    printf("  Output tensor: %d × %d = %d elements\n\n", D_MODEL, SEQ_LEN, n);

    // ---- Show first few elements ----
    printf("  First 16 output elements:\n");
    printf("  %-8s  %-12s %-12s %-10s %-10s\n", "Index", "FP32", "LNS", "AbsErr", "RelErr");
    for (int i = 0; i < 16 && i < n; i++) {
        float abs_err = fabsf(lns_output[i] - cpu_output[i]);
        float rel_err = (fabsf(cpu_output[i]) > 1e-8f)
                        ? abs_err / fabsf(cpu_output[i]) * 100.0f : 0.0f;
        printf("  [%3d]   %+11.6f  %+11.6f  %9.6f  %7.3f%%\n",
               i, cpu_output[i], lns_output[i], abs_err, rel_err);
    }
    printf("  ...\n\n");

    // ---- Aggregate error statistics ----
    error_stats stats = compute_errors(cpu_output.data(), lns_output.data(), n);

    printf("──────────────────────────────────────────────────────────\n");
    printf("  Error Statistics (LNS vs FP32)\n");
    printf("──────────────────────────────────────────────────────────\n\n");
    printf("  Total output elements  : %d\n", n);
    printf("  Mean absolute error    : %.6f\n", stats.mean_abs_err);
    printf("  Max  absolute error    : %.6f\n", stats.max_abs_err);
    printf("  Mean relative error    : %.4f%%\n", stats.mean_rel_err * 100.0f);
    printf("  Max  relative error    : %.4f%%\n", stats.max_rel_err * 100.0f);
    printf("\n");

    // ---- Per-row statistics ----
    printf("  Per-sequence-position error (mean rel %% per row):\n");
    printf("  %-6s %-12s\n", "Pos", "MeanRelErr");
    for (int row = 0; row < SEQ_LEN; row++) {
        error_stats row_stats = compute_errors(
            cpu_output.data() + row * D_MODEL,
            lns_output.data() + row * D_MODEL,
            D_MODEL
        );
        printf("  [%2d]   %.4f%%\n", row, row_stats.mean_rel_err * 100.0f);
    }

    // ---- Cleanup ----
    ggml_backend_free(lns_backend);
    ggml_backend_free(cpu_backend);
    return 0;
}
