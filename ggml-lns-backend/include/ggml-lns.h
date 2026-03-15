#pragma once

// ggml-lns: Logarithmic Number System backend for ggml
// Implements ggml's pluggable backend API using 16-bit LNS (xlns16)
// Data enters/exits as FP32; internal computation uses xlns16

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Create a new LNS backend instance
GGML_API ggml_backend_t ggml_backend_lns_init(void);

// Check if a backend is the LNS backend
GGML_API bool ggml_backend_is_lns(ggml_backend_t backend);

// Get the buffer type for the LNS backend
GGML_API ggml_backend_buffer_type_t ggml_backend_lns_buffer_type(void);

#ifdef __cplusplus
}
#endif
