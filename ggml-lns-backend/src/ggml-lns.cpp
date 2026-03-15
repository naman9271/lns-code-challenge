// ggml-lns.cpp   Full LNS Backend Implementation for ggml
// copyright 2025   GSoC Code Challenge
//
// Implements ggml's three backend interfaces (buffer_type, buffer, backend)
// using 16-bit LNS (xlns16) for internal computation.
// Data enters/exits as FP32   transparent to ggml.

#include "ggml-lns.h"
#include "ggml.h"
#include "ggml-backend.h"

// ggml backend implementation details
extern "C" {
#include "ggml-backend-impl.h"
}

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

// Include xlns16 library (single-header-style)
#include "xlns16.cpp"

// GUID for identifying this backend

static ggml_guid_t ggml_backend_lns_guid(void) {
    static ggml_guid guid = {
        0x4c, 0x4e, 0x53, 0x2d,  // "LNS-"
        0x42, 0x41, 0x43, 0x4b,  // "BACK"
        0x45, 0x4e, 0x44, 0x2d,  // "END-"
        0x30, 0x30, 0x30, 0x31   // "0001"
    };
    return &guid;
}

// Buffer context: simple heap allocation

struct lns_buffer_context {
    void * data;
    size_t size;
};

// BUFFER TYPE INTERFACE (ggml_backend_buffer_type_i)
static const char * lns_buft_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "LNS";
}

static ggml_backend_buffer_t lns_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    lns_buffer_context * ctx = new lns_buffer_context;
    ctx->size = size;
    ctx->data = malloc(size);
    if (!ctx->data) {
        delete ctx;
        return nullptr;
    }
    memset(ctx->data, 0, size);

    static struct ggml_backend_buffer_i lns_buffer_iface = {
        /* .free_buffer    = */ [](ggml_backend_buffer_t buffer) {
            lns_buffer_context * ctx = (lns_buffer_context *)buffer->context;
            free(ctx->data);
            delete ctx;
        },
        /* .get_base       = */ [](ggml_backend_buffer_t buffer) -> void * {
            return ((lns_buffer_context *)buffer->context)->data;
        },
        /* .init_tensor    = */ nullptr,
        /* .memset_tensor  = */ [](ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
            (void)buffer;
            memset((char *)tensor->data + offset, value, size);
        },
        /* .set_tensor     = */ [](ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            (void)buffer;
            memcpy((char *)tensor->data + offset, data, size);
        },
        /* .get_tensor     = */ [](ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
            (void)buffer;
            memcpy(data, (const char *)tensor->data + offset, size);
        },
        /* .cpy_tensor     = */ nullptr,
        /* .clear          = */ [](ggml_backend_buffer_t buffer, uint8_t value) {
            lns_buffer_context * ctx = (lns_buffer_context *)buffer->context;
            memset(ctx->data, value, ctx->size);
        },
        /* .reset          = */ nullptr,
    };

    return ggml_backend_buffer_init(buft, lns_buffer_iface, ctx, size);
}

static size_t lns_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return sizeof(float); // FP32 alignment
}

static size_t lns_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return SIZE_MAX;
}

static bool lns_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return true; // data is in host memory
}

static struct ggml_backend_buffer_type lns_buffer_type = {
    /* .iface = */ {
        /* .get_name      = */ lns_buft_get_name,
        /* .alloc_buffer  = */ lns_buft_alloc_buffer,
        /* .get_alignment = */ lns_buft_get_alignment,
        /* .get_max_size  = */ lns_buft_get_max_size,
        /* .get_alloc_size= */ nullptr,
        /* .is_host       = */ lns_buft_is_host,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

// LNS COMPUTE OPERATIONS
// ---- MUL_MAT: C = A × B^T ----
// This is the critical operation for LLM inference.
// In ggml, MUL_MAT computes: dst[i,j] = dot(src0_row_j, src1_row_i)
// where src0 is the weight matrix (ne0 × ne1) and src1 is the input (ne0 × ne10)
static void lns_compute_mul_mat(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1,
                                 struct ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0]; // K (inner dimension)
    const int64_t ne01 = src0->ne[1]; // N (rows of weight matrix)
    const int64_t ne10 = src1->ne[0]; // K (must equal ne00)
    const int64_t ne11 = src1->ne[1]; // M (rows of input / cols of output)

    assert(ne00 == ne10);

    const float * src0_data = (const float *)src0->data;
    const float * src1_data = (const float *)src1->data;
    float       * dst_data  = (float *)dst->data;

    // For each output element dst[i, j]:
    //   dst[i, j] = dot(src0_row[j], src1_row[i])
    for (int64_t i = 0; i < ne11; i++) {        // over input rows
        for (int64_t j = 0; j < ne01; j++) {    // over weight rows
            const float * a = src0_data + j * ne00;  // weight row j
            const float * b = src1_data + i * ne10;  // input row i

            // Compute dot product entirely in LNS domain
            dst_data[i * ne01 + j] = xlns16_vec_dot_f32(a, b, ne00);
        }
    }
}

// ---- ADD: element-wise addition ----
static void lns_compute_add(const struct ggml_tensor * src0,
                              const struct ggml_tensor * src1,
                              struct ggml_tensor * dst) {
    const int64_t ne = ggml_nelements(dst);
    const float * a = (const float *)src0->data;
    const float * b = (const float *)src1->data;
    float       * c = (float *)dst->data;

    // Handle broadcasting: src1 may be smaller (e.g., bias vector)
    const int64_t ne1 = ggml_nelements(src1);

    for (int64_t i = 0; i < ne; i++) {
        xlns16 la = fp2xlns16(a[i]);
        xlns16 lb = fp2xlns16(b[i % ne1]);
        c[i] = xlns162fp(xlns16_add(la, lb));
    }
}

// ---- MUL: element-wise multiplication ----
// In LNS, multiply = add the log representations (cheapest operation!)
static void lns_compute_mul(const struct ggml_tensor * src0,
                              const struct ggml_tensor * src1,
                              struct ggml_tensor * dst) {
    const int64_t ne = ggml_nelements(dst);
    const float * a = (const float *)src0->data;
    const float * b = (const float *)src1->data;
    float       * c = (float *)dst->data;

    const int64_t ne1 = ggml_nelements(src1);

    for (int64_t i = 0; i < ne; i++) {
        xlns16 la = fp2xlns16(a[i]);
        xlns16 lb = fp2xlns16(b[i % ne1]);
        c[i] = xlns162fp(xlns16_mul(la, lb));
    }
}

// ---- SCALE: multiply all elements by a scalar ----
static void lns_compute_scale(const struct ggml_tensor * src0,
                                struct ggml_tensor * dst) {
    const int64_t ne = ggml_nelements(dst);
    const float * a = (const float *)src0->data;
    float       * c = (float *)dst->data;

    // Scale factor is stored in op_params
    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    xlns16 lscale = fp2xlns16(scale);

    for (int64_t i = 0; i < ne; i++) {
        c[i] = xlns162fp(xlns16_mul(fp2xlns16(a[i]), lscale));
    }
}

// ---- SOFT_MAX: per-row softmax ----
static void lns_compute_soft_max(const struct ggml_tensor * src0,
                                   struct ggml_tensor * dst) {
    const int64_t ne0 = src0->ne[0]; // row width
    const int64_t ne1 = src0->ne[1]; // number of rows

    const float * src = (const float *)src0->data;
    float       * out = (float *)dst->data;

    // Use xlns16 softmax per row
    std::vector<xlns16> lns_in(ne0);
    std::vector<xlns16> lns_out(ne0);

    for (int64_t row = 0; row < ne1; row++) {
        const float * row_src = src + row * ne0;
        float       * row_dst = out + row * ne0;

        // Convert row to LNS
        for (int64_t i = 0; i < ne0; i++) {
            lns_in[i] = fp2xlns16(row_src[i]);
        }

        // Compute softmax in LNS domain
        xlns16_softmax(lns_in.data(), lns_out.data(), ne0);

        // Convert back to FP32
        for (int64_t i = 0; i < ne0; i++) {
            row_dst[i] = xlns162fp(lns_out[i]);
        }
    }
}

// ---- RMS_NORM: root mean square normalization ----
// In LNS: square = double the log, sqrt = halve the log
static void lns_compute_rms_norm(const struct ggml_tensor * src0,
                                   struct ggml_tensor * dst) {
    const int64_t ne0 = src0->ne[0]; // normalization dimension
    const int64_t ne1 = src0->ne[1]; // number of vectors

    const float * src = (const float *)src0->data;
    float       * out = (float *)dst->data;

    // Epsilon from op_params
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    for (int64_t row = 0; row < ne1; row++) {
        const float * row_src = src + row * ne0;
        float       * row_dst = out + row * ne0;

        // Compute sum of squares in LNS
        xlns16 sum_sq = xlns16_zero;
        for (int64_t i = 0; i < ne0; i++) {
            xlns16 val = fp2xlns16(row_src[i]);
            xlns16 sq  = xlns16_mul(val, val); // square = add logs
            sum_sq = xlns16_add(sum_sq, sq);
        }

        // mean = sum_sq / n, then 1/sqrt(mean + eps)
        float mean_sq = xlns162fp(sum_sq) / (float)ne0;
        float inv_rms = 1.0f / sqrtf(mean_sq + eps);
        xlns16 lns_inv_rms = fp2xlns16(inv_rms);

        // Normalize: out[i] = src[i] * inv_rms
        for (int64_t i = 0; i < ne0; i++) {
            xlns16 val = fp2xlns16(row_src[i]);
            row_dst[i] = xlns162fp(xlns16_mul(val, lns_inv_rms));
        }
    }
}

// ---- SILU: x * sigmoid(x) ----
static void lns_compute_silu(const struct ggml_tensor * src0,
                               struct ggml_tensor * dst) {
    const int64_t ne = ggml_nelements(dst);
    const float * a = (const float *)src0->data;
    float       * c = (float *)dst->data;

    for (int64_t i = 0; i < ne; i++) {
        c[i] = xlns162fp(xlns16_silu(fp2xlns16(a[i])));
    }
}

// ---- GELU: approximate Gaussian error linear unit ----
static void lns_compute_gelu(const struct ggml_tensor * src0,
                               struct ggml_tensor * dst) {
    const int64_t ne = ggml_nelements(dst);
    const float * a = (const float *)src0->data;
    float       * c = (float *)dst->data;

    for (int64_t i = 0; i < ne; i++) {
        c[i] = xlns162fp(xlns16_gelu(fp2xlns16(a[i])));
    }
}

// ---- CONT: copy data to contiguous layout ----
// Used after transpose/permute to make data contiguous for subsequent ops
static void lns_compute_cont(const struct ggml_tensor * src0,
                               struct ggml_tensor * dst) {
    // Copy element by element using src0's strides (which may be non-contiguous)
    // and dst's contiguous layout
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    const float * src_ptr = (const float *)((const char *)src0->data
                        + i0 * src0->nb[0]
                        + i1 * src0->nb[1]
                        + i2 * src0->nb[2]
                        + i3 * src0->nb[3]);
                    float * dst_ptr = (float *)((char *)dst->data
                        + i0 * dst->nb[0]
                        + i1 * dst->nb[1]
                        + i2 * dst->nb[2]
                        + i3 * dst->nb[3]);
                    *dst_ptr = *src_ptr;
                }
            }
        }
    }
}

// BACKEND INTERFACE (ggml_backend_i)
static const char * lns_backend_get_name(ggml_backend_t backend) {
    (void)backend;
    return "LNS";
}

static void lns_backend_free(ggml_backend_t backend) {
    delete backend;
}

static enum ggml_status lns_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    (void)backend;

    for (int i = 0; i < ggml_graph_n_nodes(cgraph); i++) {
        struct ggml_tensor * node = ggml_graph_node(cgraph, i);

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // These are metadata-only operations, no computation needed
                break;

            case GGML_OP_CONT:
                lns_compute_cont(node->src[0], node);
                break;

            case GGML_OP_MUL_MAT:
                lns_compute_mul_mat(node->src[0], node->src[1], node);
                break;

            case GGML_OP_ADD:
                lns_compute_add(node->src[0], node->src[1], node);
                break;

            case GGML_OP_MUL:
                lns_compute_mul(node->src[0], node->src[1], node);
                break;

            case GGML_OP_SCALE:
                lns_compute_scale(node->src[0], node);
                break;

            case GGML_OP_SOFT_MAX:
                lns_compute_soft_max(node->src[0], node);
                break;

            case GGML_OP_RMS_NORM:
                lns_compute_rms_norm(node->src[0], node);
                break;

            case GGML_OP_UNARY: {
                enum ggml_unary_op uop = ggml_get_unary_op(node);
                if (uop == GGML_UNARY_OP_SILU) {
                    lns_compute_silu(node->src[0], node);
                } else if (uop == GGML_UNARY_OP_GELU) {
                    lns_compute_gelu(node->src[0], node);
                } else {
                    fprintf(stderr, "LNS backend: unsupported unary op %s\n",
                            ggml_unary_op_name(uop));
                    return GGML_STATUS_FAILED;
                }
                break;
            }

            default:
                fprintf(stderr, "LNS backend: unsupported op %s\n",
                        ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

// DEVICE INTERFACE (ggml_backend_device_i)
static const char * lns_device_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    return "LNS";
}

static const char * lns_device_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return "Logarithmic Number System (16-bit xlns16) backend";
}

static void lns_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void)dev;
    // Report system memory (LNS uses host memory)
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type lns_device_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_ACCEL; // accelerator alongside CPU
}

static void lns_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = lns_device_get_name(dev);
    props->description = lns_device_get_description(dev);
    props->memory_free  = 0;
    props->memory_total = 0;
    props->type        = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->caps = {
        /* .async             = */ false,
        /* .host_buffer       = */ true,
        /* .buffer_from_host_ptr = */ false,
        /* .events            = */ false,
    };
}

static ggml_backend_t lns_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void)params;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_lns_guid(),
        /* .iface   = */ {
            /* .get_name            = */ lns_backend_get_name,
            /* .free                = */ lns_backend_free,
            /* .set_tensor_async    = */ nullptr,
            /* .get_tensor_async    = */ nullptr,
            /* .cpy_tensor_async    = */ nullptr,
            /* .synchronize         = */ nullptr,
            /* .graph_plan_create   = */ nullptr,
            /* .graph_plan_free     = */ nullptr,
            /* .graph_plan_update   = */ nullptr,
            /* .graph_plan_compute  = */ nullptr,
            /* .graph_compute       = */ lns_backend_graph_compute,
            /* .event_record        = */ nullptr,
            /* .event_wait          = */ nullptr,
            /* .graph_optimize      = */ nullptr,
        },
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };

    return backend;
}

static ggml_backend_buffer_type_t lns_device_get_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return &lns_buffer_type;
}

static bool lns_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;

    // Only support FP32 tensors
    if (op->type != GGML_TYPE_F32) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_RMS_NORM:
        case GGML_OP_CONT:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_UNARY: {
            enum ggml_unary_op uop = ggml_get_unary_op(op);
            return uop == GGML_UNARY_OP_SILU || uop == GGML_UNARY_OP_GELU;
        }

        default:
            return false;
    }
}

static bool lns_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    return buft == &lns_buffer_type;
}

// DEVICE AND REGISTRY SINGLETONS
static struct ggml_backend_device_i lns_device_iface = {
    /* .get_name            = */ lns_device_get_name,
    /* .get_description     = */ lns_device_get_description,
    /* .get_memory          = */ lns_device_get_memory,
    /* .get_type            = */ lns_device_get_type,
    /* .get_props           = */ lns_device_get_props,
    /* .init_backend        = */ lns_device_init_backend,
    /* .get_buffer_type     = */ lns_device_get_buffer_type,
    /* .get_host_buffer_type= */ nullptr,
    /* .buffer_from_host_ptr= */ nullptr,
    /* .supports_op         = */ lns_device_supports_op,
    /* .supports_buft       = */ lns_device_supports_buft,
    /* .offload_op          = */ nullptr,
    /* .event_new           = */ nullptr,
    /* .event_free          = */ nullptr,
    /* .event_synchronize   = */ nullptr,
};

static struct ggml_backend_device lns_device = {
    /* .iface   = */ lns_device_iface,
    /* .reg     = */ nullptr, // set below
    /* .context = */ nullptr,
};

static const char * lns_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return "LNS";
}

static size_t lns_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return 1;
}

static ggml_backend_dev_t lns_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg;
    (void)index;
    return &lns_device;
}

static struct ggml_backend_reg lns_reg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ {
        /* .get_name         = */ lns_reg_get_name,
        /* .get_device_count = */ lns_reg_get_device_count,
        /* .get_device       = */ lns_reg_get_device,
        /* .get_proc_address = */ nullptr,
    },
    /* .context     = */ nullptr,
};

// PUBLIC API
ggml_backend_t ggml_backend_lns_init(void) {
    lns_device.reg = &lns_reg;
    lns_buffer_type.device = &lns_device;
    return lns_device_init_backend(&lns_device, nullptr);
}

bool ggml_backend_is_lns(ggml_backend_t backend) {
    return backend != nullptr &&
           ggml_guid_matches(ggml_backend_guid(backend), ggml_backend_lns_guid());
}

ggml_backend_buffer_type_t ggml_backend_lns_buffer_type(void) {
    lns_device.reg = &lns_reg;
    lns_buffer_type.device = &lns_device;
    return &lns_buffer_type;
}
