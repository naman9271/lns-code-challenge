# LNS Code Challenges - GSoC 2025

Code challenges demonstrating Logarithmic Number System (LNS) arithmetic using the [xlnscpp](https://github.com/xlnsresearch/xlns) library, as part of the GSoC proposal for integrating LNS into [ggml](https://github.com/ggerganov/ggml).

---

## Challenges Overview

| # | Challenge | What it does | Precision |
|---|-----------|-------------|-----------|
| 3 | FP32 Baseline | Standard float matmul   establishes reference | Exact |
| 4 | xlns32 Matmul | Same matmul in 32-bit LNS | Matches FP32 |
| 5 | xlns16 Matmul | Same matmul in 16-bit LNS | ~0.23% error |
| 6 | **Approach A**   `#ifdef` Unified | Compile-time FP/LNS switching (preferred) | Same as above |
| Experments | **Approach B**   ggml-lns Backend | Experimental standalone backend (not preferred) | ~0.23% (matmul), ~5.6% (transformer) |

---

## Challenge 3: FP32 Baseline Matrix Multiply

Standalone FP32 matrix multiplication (C = A × B^T) using the same data from the [ggml blog tutorial](https://huggingface.co/blog/introduction-to-ggml). This establishes the reference output for all subsequent challenges.

```bash
cd challenge3_fp_matmul
g++ -o fp_matmul fp_matmul.cpp -lm
./fp_matmul
```

**Output:**
```
FP32 mul mat (3 x 4) (transposed result):
[ 60.00 55.00 50.00 110.00
 90.00 54.00 54.00 126.00
 42.00 29.00 28.00 64.00 ]
```

---

## Challenge 4: 32-bit LNS Matrix Multiply

Same matrices, computed using `xlns32_float`. Results match FP32 exactly at 2 decimal places.

```bash
cd challenge4_xlns32_matmul
g++ -o xlns32_matmul xlns32_matmul.cpp -lm
./xlns32_matmul
```

**Output:**
```
xlns32 LNS mul mat (3 x 4) (transposed result):
[ 60.00 55.00 50.00 110.00
 90.00 54.00 54.00 126.00
 42.00 29.00 28.00 64.00 ]
```

---

## Challenge 5: 16-bit LNS Matrix Multiply

Same matrices, computed using `xlns16_float`. Shows reduced precision (~0.23% mean error) due to 7-bit fractional log.

```bash
cd challenge5_xlns16_matmul
g++ -o xlns16_matmul xlns16_matmul.cpp -lm
./xlns16_matmul
```

**Output:**
```
xlns16 LNS mul mat (3 x 4) (transposed result):
[ 59.97 55.00 49.89 109.99
 89.53 53.82 53.82 125.26
 41.95 28.87 27.95 64.00 ]
```

---

## Challenge 6: Approach A   `#ifdef` Unified Matmul (Preferred)

**This is the preferred integration approach.** A single source file compiled with different `-D` flags to switch between FP32 and LNS at compile time. This mirrors the exact pattern used to modify ggml-cpu's `ggml_vec_dot_f32()`.

```bash
cd challenge6_unified_matmul

# FP32 baseline
g++ -o matmul_fp  matmul_unified.cpp -lm
./matmul_fp

# 32-bit LNS
g++ -o matmul_x32 matmul_unified.cpp -lm -DGGML_XLNS32
./matmul_x32

# 16-bit LNS
g++ -o matmul_x16 matmul_unified.cpp -lm -DGGML_XLNS16
./matmul_x16
```

**How this maps to ggml-cpu:**

| This Prototype | ggml-cpu Actual |
|---|---|
| `matmul_unified.cpp` | `vec.h` + `vec.cpp` |
| Inner loop (sum of products) | `ggml_vec_dot_f32()` |
| `-DGGML_XLNS16` flag | `cmake -DGGML_XLNS16=ON` |

---

## Approach B: Experimental ggml-lns Backend (Not Preferred)

> As per the mentor's review, this approach is **not the preferred integration path**. It was implemented as an experimental exercise to gain a deeper understanding of how the ggml backend system works from scratch   how backends are structured, how buffer/device/registry interfaces fit together, and how compute graphs get dispatched.

Instead of modifying ggml-cpu with `#ifdef`, this creates a **completely separate, standalone backend**   the same architecture used by CUDA, Metal, and Vulkan backends. The backend uses **xlns16 (16-bit LNS)** internally and implements 8 operations: `MUL_MAT`, `ADD`, `MUL`, `SCALE`, `SOFT_MAX`, `RMS_NORM`, `SILU`, `GELU`.

### Build

```bash
cd ggml-lns-backend
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Demo 1: LNS Matrix Multiply

Runs the **same matrices from Challenges 3–6** through both the CPU (FP32) and LNS backends side-by-side.

```bash
./bin/lns_matmul
```

**Result:** Mean relative error of **0.23%**   matches the standalone xlns16 results from Challenge 5.

### Demo 2: Mini Transformer Block

Runs a complete transformer layer (attention + SwiGLU FFN) through both backends, exercising all 8 LNS operations in the exact pattern used by LLMs like LLaMA. Uses random weights and random input   the purpose is to verify LNS accuracy when operations are **chained together** (20 total invocations).

```bash
./bin/mini_transformer
```

**Result:** Mean absolute error of **0.056** across 512 output elements. All 8 operations executed successfully.

---


**Key insight:** 32-bit LNS matches FP32 exactly. 16-bit LNS introduces errors up to ~0.6%, demonstrating the precision–storage tradeoff.

---

## Repository Structure

```
lns-code-challenge/
├── xlnscpp/                          # LNS library (from xlnsresearch/xlns)
│   ├── xlns16.cpp                    # 16-bit LNS (1 sign + 8 int + 7 frac bits)
│   └── xlns32.cpp                    # 32-bit LNS (1 sign + 8 int + 23 frac bits)
├── challenge3_fp_matmul/             # Challenge 3: FP32 baseline
│   └── fp_matmul.cpp
├── challenge4_xlns32_matmul/         # Challenge 4: xlns32 matmul
│   └── xlns32_matmul.cpp
├── challenge5_xlns16_matmul/         # Challenge 5: xlns16 matmul
│   └── xlns16_matmul.cpp
├── challenge6_unified_matmul/        # Challenge 6: Approach A (#ifdef)
│   └── matmul_unified.cpp
└── ggml-lns-backend/                 # Approach B: Experimental backend
    ├── CMakeLists.txt                # Fetches ggml via FetchContent
    ├── include/ggml-lns.h            # Public API (3 functions)
    ├── src/ggml-lns.cpp              # Backend implementation (~610 lines)
    └── examples/
        ├── demo_lns_matmul.cpp       # FP32 vs LNS matmul comparison
        └── demo_mini_transformer.cpp # Full transformer layer in LNS
```

## License

The xlnscpp library files are copyright Mark G. Arnold. See [xlnsresearch/xlns](https://github.com/xlnsresearch/xlns) for details.
