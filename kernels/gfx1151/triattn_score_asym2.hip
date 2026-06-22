#include <hip/hip_runtime.h>

// Native ds_swizzle_b32 Hadamard butterfly: XOR swap + add/sub.
// Replaces __shfl_xor (compiles to ds_bpermute, needs VGPR for lane index).
// ds_swizzle encodes XOR pattern as immediate — no VGPR, lower latency.
#define HADAMARD_BFLY(v, pattern, stride, tid) do {                     \
    float _p = __int_as_float(                                          \
        __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pattern)));     \
    if ((tid) & (stride)) { (v) = _p - (v); }                          \
    else                  { (v) = (v) + _p; }                           \
} while(0)

// Lloyd-Max optimal centroids for N(0, 1/128) after unit-norm + FWHT(1/sqrt(128))
__constant__ float TURBO_C2[4] = {-0.133466f, -0.040022f, 0.040022f, 0.133466f};
__constant__ float TURBO_C3[8] = {-0.190685f, -0.117832f, -0.065717f, -0.021460f, 0.021460f, 0.065717f, 0.117832f, 0.190685f};
__constant__ float TURBO_C4[16] = {
    -0.241565f, -0.182875f, -0.143012f, -0.111016f, -0.083262f, -0.057983f, -0.034295f, -0.011225f,
     0.011225f,  0.034295f,  0.057983f,  0.083262f,  0.111016f,  0.143012f,  0.182875f,  0.241565f
};

// Lloyd-Max optimal centroids for N(0, 1/256) — all values = 128-dim × 1/sqrt(2)
__constant__ float TURBO_C2_256[4] = {-0.094376f, -0.028300f, 0.028300f, 0.094376f};
__constant__ float TURBO_C3_256[8] = {-0.134860f, -0.083320f, -0.046469f, -0.015176f, 0.015176f, 0.046469f, 0.083320f, 0.134860f};
__constant__ float TURBO_C4_256[16] = {
    -0.170807f, -0.129321f, -0.101134f, -0.078505f, -0.058869f, -0.041003f, -0.024249f, -0.007938f,
     0.007938f,  0.024249f,  0.041003f,  0.058869f,  0.078505f,  0.101134f,  0.129321f,  0.170807f
};

// In-place FWHT on 128 elements in registers.
// signs1/signs2 are ±1.0f arrays in global memory (uploaded once).
__device__ void fwht_forward_128(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    // Step 1: apply signs1
    for (int i = 0; i < 128; i++) x[i] *= signs1[i];

    // Step 2: Walsh-Hadamard butterfly (7 passes for n=128)
    for (int stride = 1; stride < 128; stride <<= 1) {
        for (int i = 0; i < 128; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }

    // Step 3: scale by 1/sqrt(128)
    const float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;

    // Step 4: apply signs2
    for (int i = 0; i < 128; i++) x[i] *= signs2[i];
}

// Inverse FWHT: signs2 -> butterfly -> scale -> signs1 (reverse order)
__device__ void fwht_inverse_128(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 128; i++) x[i] *= signs2[i];
    for (int stride = 1; stride < 128; stride <<= 1) {
        for (int i = 0; i < 128; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128 * signs1[i];
}


// Register-only FWHT via __shfl_xor. Zero shared memory. Zero barriers.
// Each thread owns 4 of 128 elements in registers (a,b,c,d).
// signs1/signs2 are applied from constant memory.
// After this function, (a,b,c,d) are in the FWHT-rotated space.
__device__ void fwht_shfl_forward(float& a, float& b, float& c, float& d,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    int d0 = tid * 4;
    // Apply signs1
    a *= signs1[d0]; b *= signs1[d0+1]; c *= signs1[d0+2]; d *= signs1[d0+3];

    // Local butterfly: stride 1 (pairs 0↔1, 2↔3)
    float t;
    t = a; a = a + b; b = t - b;
    t = c; c = c + d; d = t - d;

    // Local butterfly: stride 2 (pairs 0↔2, 1↔3)
    t = a; a = a + c; c = t - c;
    t = b; b = b + d; d = t - d;

    // Wave-level butterfly via ds_swizzle_b32 (native XOR swap, no VGPR for lane index)
    HADAMARD_BFLY(a, 0x041F, 1, tid); HADAMARD_BFLY(b, 0x041F, 1, tid);
    HADAMARD_BFLY(c, 0x041F, 1, tid); HADAMARD_BFLY(d, 0x041F, 1, tid);
    HADAMARD_BFLY(a, 0x081F, 2, tid); HADAMARD_BFLY(b, 0x081F, 2, tid);
    HADAMARD_BFLY(c, 0x081F, 2, tid); HADAMARD_BFLY(d, 0x081F, 2, tid);
    HADAMARD_BFLY(a, 0x101F, 4, tid); HADAMARD_BFLY(b, 0x101F, 4, tid);
    HADAMARD_BFLY(c, 0x101F, 4, tid); HADAMARD_BFLY(d, 0x101F, 4, tid);
    HADAMARD_BFLY(a, 0x201F, 8, tid); HADAMARD_BFLY(b, 0x201F, 8, tid);
    HADAMARD_BFLY(c, 0x201F, 8, tid); HADAMARD_BFLY(d, 0x201F, 8, tid);
    HADAMARD_BFLY(a, 0x401F, 16, tid); HADAMARD_BFLY(b, 0x401F, 16, tid);
    HADAMARD_BFLY(c, 0x401F, 16, tid); HADAMARD_BFLY(d, 0x401F, 16, tid);

    // Scale by 1/sqrt(128) and apply signs2
    const float s = 0.08838834764831845f;
    a *= s * signs2[d0]; b *= s * signs2[d0+1]; c *= s * signs2[d0+2]; d *= s * signs2[d0+3];
}

// Inverse: signs2, butterfly, scale, signs1 (reverse order)
__device__ void fwht_shfl_inverse(float& a, float& b, float& c, float& d,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    int d0 = tid * 4;
    a *= signs2[d0]; b *= signs2[d0+1]; c *= signs2[d0+2]; d *= signs2[d0+3];

    HADAMARD_BFLY(a, 0x041F, 1, tid); HADAMARD_BFLY(b, 0x041F, 1, tid);
    HADAMARD_BFLY(c, 0x041F, 1, tid); HADAMARD_BFLY(d, 0x041F, 1, tid);
    HADAMARD_BFLY(a, 0x081F, 2, tid); HADAMARD_BFLY(b, 0x081F, 2, tid);
    HADAMARD_BFLY(c, 0x081F, 2, tid); HADAMARD_BFLY(d, 0x081F, 2, tid);
    HADAMARD_BFLY(a, 0x101F, 4, tid); HADAMARD_BFLY(b, 0x101F, 4, tid);
    HADAMARD_BFLY(c, 0x101F, 4, tid); HADAMARD_BFLY(d, 0x101F, 4, tid);
    HADAMARD_BFLY(a, 0x201F, 8, tid); HADAMARD_BFLY(b, 0x201F, 8, tid);
    HADAMARD_BFLY(c, 0x201F, 8, tid); HADAMARD_BFLY(d, 0x201F, 8, tid);
    HADAMARD_BFLY(a, 0x401F, 16, tid); HADAMARD_BFLY(b, 0x401F, 16, tid);
    HADAMARD_BFLY(c, 0x401F, 16, tid); HADAMARD_BFLY(d, 0x401F, 16, tid);

    float t;
    t = a; a = a + c; c = t - c;
    t = b; b = b + d; d = t - d;
    t = a; a = a + b; b = t - b;
    t = c; c = c + d; d = t - d;

    const float s = 0.08838834764831845f;
    a *= s * signs1[d0]; b *= s * signs1[d0+1]; c *= s * signs1[d0+2]; d *= s * signs1[d0+3];
}

// ============================================================================
// 256-element FWHT via __shfl_xor. Zero shared memory. Zero barriers.
// Each thread owns 8 of 256 elements in registers (v0..v7).
// 32 threads × 8 elements = 256.
// log2(256) = 8 butterfly passes:
//   Passes 1-3: register-local (strides 1, 2, 4 in element space)
//   Passes 4-8: warp shuffle    (strides 8,16,32,64,128 in element space
//                                = strides 1,2,4,8,16 in thread space)
//
// VGPR budget (RDNA, 32-lane wavefront):
//   v0-v7:  8 VGPRs  (the 8 data elements)
//   t:      1 VGPR   (butterfly temp, reused across all local passes)
//   d0:     1 VGPR   (base index for sign table loads, computed once)
//   s:      1 VGPR   (scale constant 1/sqrt(256), fused with signs2)
//   p0-p7:  0 VGPRs  (shuffle results alias v0-v7 via compiler; see note)
//   tid:    1 VGPR   (passed in, likely already live in caller)
//   signs:  0 VGPRs  (loaded directly into multiply operand from global)
//   -------
//   Total: 12 VGPRs kernel-private + 8 data = 20 VGPRs worst case
//
// Note on shuffle passes: each __shfl_xor result (p0..p7) is consumed
// immediately in the add/sub that overwrites v0..v7, so the compiler
// can alias them. At most 1 extra VGPR for the shuffle temporary if
// the compiler doesn't overlap. In practice expect 18-19 VGPRs on RDNA.
// ============================================================================

__device__ void fwht_shfl_forward_256(
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    const int d0 = tid * 8;

    // Apply signs1 (randomize input for incoherence)
    v0 *= signs1[d0];   v1 *= signs1[d0+1]; v2 *= signs1[d0+2]; v3 *= signs1[d0+3];
    v4 *= signs1[d0+4]; v5 *= signs1[d0+5]; v6 *= signs1[d0+6]; v7 *= signs1[d0+7];

    // --- Pass 1: stride 1 in element space ---
    // Butterfly pairs: (0,1), (2,3), (4,5), (6,7)
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    // --- Pass 2: stride 2 in element space ---
    // Butterfly pairs: (0,2), (1,3), (4,6), (5,7)
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    // --- Pass 3: stride 4 in element space ---
    // Butterfly pairs: (0,4), (1,5), (2,6), (3,7)
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // --- Passes 4-8: ds_swizzle_b32 (native XOR swap, no VGPR for lane index) ---
    #define HBFLY8(pat, str) \
        HADAMARD_BFLY(v0,(pat),(str),tid); HADAMARD_BFLY(v1,(pat),(str),tid); \
        HADAMARD_BFLY(v2,(pat),(str),tid); HADAMARD_BFLY(v3,(pat),(str),tid); \
        HADAMARD_BFLY(v4,(pat),(str),tid); HADAMARD_BFLY(v5,(pat),(str),tid); \
        HADAMARD_BFLY(v6,(pat),(str),tid); HADAMARD_BFLY(v7,(pat),(str),tid)
    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8

    // Scale by 1/sqrt(256) = 1/16 = 0.0625 and apply signs2
    const float s = 0.0625f;
    v0 *= s * signs2[d0];   v1 *= s * signs2[d0+1]; v2 *= s * signs2[d0+2]; v3 *= s * signs2[d0+3];
    v4 *= s * signs2[d0+4]; v5 *= s * signs2[d0+5]; v6 *= s * signs2[d0+6]; v7 *= s * signs2[d0+7];
}

// Inverse FWHT-256: signs2 -> shuffle passes -> local passes (reversed) -> scale -> signs1
__device__ void fwht_shfl_inverse_256(
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    const int d0 = tid * 8;

    // Apply signs2 (undo output randomization)
    v0 *= signs2[d0];   v1 *= signs2[d0+1]; v2 *= signs2[d0+2]; v3 *= signs2[d0+3];
    v4 *= signs2[d0+4]; v5 *= signs2[d0+5]; v6 *= signs2[d0+6]; v7 *= signs2[d0+7];

    // --- Passes 4-8: ds_swizzle_b32 ---
    #define HBFLY8(pat, str) \
        HADAMARD_BFLY(v0,(pat),(str),tid); HADAMARD_BFLY(v1,(pat),(str),tid); \
        HADAMARD_BFLY(v2,(pat),(str),tid); HADAMARD_BFLY(v3,(pat),(str),tid); \
        HADAMARD_BFLY(v4,(pat),(str),tid); HADAMARD_BFLY(v5,(pat),(str),tid); \
        HADAMARD_BFLY(v6,(pat),(str),tid); HADAMARD_BFLY(v7,(pat),(str),tid)
    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8

    // --- Pass 3 (reverse): stride 4 --- pairs (0,4), (1,5), (2,6), (3,7)
    float t;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // --- Pass 2 (reverse): stride 2 --- pairs (0,2), (1,3), (4,6), (5,7)
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    // --- Pass 1 (reverse): stride 1 --- pairs (0,1), (2,3), (4,5), (6,7)
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    // Scale by 1/sqrt(256) and apply signs1
    const float s = 0.0625f;
    v0 *= s * signs1[d0];   v1 *= s * signs1[d0+1]; v2 *= s * signs1[d0+2]; v3 *= s * signs1[d0+3];
    v4 *= s * signs1[d0+4]; v5 *= s * signs1[d0+5]; v6 *= s * signs1[d0+6]; v7 *= s * signs1[d0+7];
}

// ============================================================================
// Scalar reference FWHT-256 (for validation / non-turbo paths)
// ============================================================================
__device__ void fwht_forward_256(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 256; i++) x[i] *= signs1[i];
    for (int stride = 1; stride < 256; stride <<= 1) {
        for (int i = 0; i < 256; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_256 = 0.0625f;
    for (int i = 0; i < 256; i++) x[i] *= inv_sqrt_256 * signs2[i];
}

__device__ void fwht_inverse_256(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 256; i++) x[i] *= signs2[i];
    for (int stride = 1; stride < 256; stride <<= 1) {
        for (int i = 0; i < 256; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_256 = 0.0625f;
    for (int i = 0; i < 256; i++) x[i] *= inv_sqrt_256 * signs1[i];
}


// Branchless 2-bit quantize: returns index 0-3 (thresholds for N(0, 1/128))
__device__ int turbo_quantize_2bit(float x) {
    return (x > -0.086744f) + (x > 0.0f) + (x > 0.086744f);
}

// Branchless 3-bit quantize: returns index 0-7
__device__ int turbo_quantize_3bit(float x) {
    return (x > -0.154258f) + (x > -0.091775f) + (x > -0.043589f) + (x > 0.0f)
         + (x > 0.043589f) + (x > 0.091775f) + (x > 0.154258f);
}

// Branchless 4-bit quantize: returns index 0-15
__device__ int turbo_quantize_4bit(float x) {
    return (x > -0.212220f) + (x > -0.162944f) + (x > -0.127014f) + (x > -0.097139f)
         + (x > -0.070622f) + (x > -0.046139f) + (x > -0.022760f) + (x > 0.0f)
         + (x > 0.022760f) + (x > 0.046139f) + (x > 0.070622f) + (x > 0.097139f)
         + (x > 0.127014f) + (x > 0.162944f) + (x > 0.212220f);
}

// 256-dim quantize functions — thresholds = 128-dim × 1/sqrt(2)
__device__ int turbo_quantize_2bit_256(float x) {
    return (x > -0.061352f) + (x > 0.0f) + (x > 0.061352f);
}
__device__ int turbo_quantize_3bit_256(float x) {
    return (x > -0.109068f) + (x > -0.064903f) + (x > -0.030827f) + (x > 0.0f)
         + (x > 0.030827f) + (x > 0.064903f) + (x > 0.109068f);
}
__device__ int turbo_quantize_4bit_256(float x) {
    return (x > -0.150086f) + (x > -0.115239f) + (x > -0.089815f) + (x > -0.068691f)
         + (x > -0.049935f) + (x > -0.032626f) + (x > -0.016096f) + (x > 0.0f)
         + (x > 0.016096f) + (x > 0.032626f) + (x > 0.049935f) + (x > 0.068691f)
         + (x > 0.089815f) + (x > 0.115239f) + (x > 0.150086f);
}

// Sign flip array for cheap decorrelation (seed=42, ±1.0)
__constant__ float TURBO_SIGNS1[128] = {
  1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f,
  1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,
 -1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f,
  1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,
  1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,
  1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f,
  1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f
};

// Block-diagonal 2x2 Givens rotation primitives for givens4 KV cache.
//
// Thread tid owns 4 consecutive dims [tid*4..tid*4+3].
// Two independent Givens blocks per thread:
//   Block A: (a, b) rotated by angle[tid*2]
//   Block B: (c, d) rotated by angle[tid*2+1]
//
// cos_theta, sin_theta: [head_dim/2] float arrays (64 entries for hd=128).
// Purely register-local — no wave shuffles, no LDS, no cross-thread deps.
#pragma once

__device__ __forceinline__ void givens_forward(
    float& a, float& b, float& c, float& d,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta, int tid)
{
    int b0 = tid * 2;
    float c0 = cos_theta[b0],   s0 = sin_theta[b0];
    float c1 = cos_theta[b0+1], s1 = sin_theta[b0+1];
    // Block A: rotate (a, b)
    float a2 = a * c0 - b * s0;
    float b2 = a * s0 + b * c0;
    a = a2; b = b2;
    // Block B: rotate (c, d)
    float c2 = c * c1 - d * s1;
    float d2 = c * s1 + d * c1;
    c = c2; d = d2;
}

__device__ __forceinline__ void givens_inverse(
    float& a, float& b, float& c, float& d,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta, int tid)
{
    int b0 = tid * 2;
    float c0 = cos_theta[b0],   s0 = sin_theta[b0];
    float c1 = cos_theta[b0+1], s1 = sin_theta[b0+1];
    // Inverse = transpose: negate sin
    float a2 =  a * c0 + b * s0;
    float b2 = -a * s0 + b * c0;
    a = a2; b = b2;
    float c2 =  c * c1 + d * s1;
    float d2 = -c * s1 + d * c1;
    c = c2; d = d2;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// TriAttention scoring on an asym2 post-RoPE K cache.
//
// asym2 stores K as: [4B cnorm][head_dim/4 packed bytes, 4 × 2-bit each]
// with the same per-band Givens rotation asym{3,4} use, but a 4-centroid
// (2-bit) codebook TURBO_C2_256. Per-thread layout matches the write
// kernel (kv_cache_write_asym_k_givens2.hip): n_halves passes, each
// thread owns 4 consecutive dims (2 Givens pairs) per pass; cos_theta /
// sin_theta indexed by `block_tid = half * 32 + tid`, bands at
// `block_tid * 2 + {0, 1}`.
//
// Grid: [seq_len, n_heads, 1]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>


extern "C" __launch_bounds__(32, 16)
__global__ void triattn_score_asym2(
    const unsigned char* __restrict__ k_cache,
    const float* __restrict__ centers,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    float* __restrict__ scores,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int n_rot,
    float rope_theta,
    float p_q,
    int seq_len
) {
    const int pos = blockIdx.x;
    const int h = blockIdx.y;
    if (pos >= seq_len || h >= n_heads) return;
    const int tid = threadIdx.x;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int n_bands = head_dim / 2;
    const int n_rot_bands = n_rot / 2;
    const int n_halves = head_dim / 128;

    const int k_bytes_per_head = 4 + head_dim / 4;
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const unsigned char* k_head = k_cache + (size_t)pos * k_bytes_per_pos + kv_h * k_bytes_per_head;

    const float cnorm = *(const float*)k_head;
    const float* c_head = centers + h * n_bands * 3;

    float s_trig = 0.0f;
    float s_norm = 0.0f;

    for (int half = 0; half < n_halves; half++) {
        const int block_tid = half * 32 + tid;
        const int byte_off  = 4 + half * 32 + tid;

        const unsigned int packed = (unsigned int)k_head[byte_off];
        const int i0 = (int)((packed     ) & 0x3);
        const int i1 = (int)((packed >> 2) & 0x3);
        const int i2 = (int)((packed >> 4) & 0x3);
        const int i3 = (int)((packed >> 6) & 0x3);

        const float v0 = cnorm * TURBO_C2_256[i0];
        const float v1 = cnorm * TURBO_C2_256[i1];
        const float v2 = cnorm * TURBO_C2_256[i2];
        const float v3 = cnorm * TURBO_C2_256[i3];

        const float cA = cos_theta[block_tid * 2 + 0];
        const float sA = sin_theta[block_tid * 2 + 0];
        const float cB = cos_theta[block_tid * 2 + 1];
        const float sB = sin_theta[block_tid * 2 + 1];

        // Inverse Givens per pair.
        const float k0_re =  cA * v0 + sA * v1;
        const float k0_im = -sA * v0 + cA * v1;
        const float k1_re =  cB * v2 + sB * v3;
        const float k1_im = -sB * v2 + cB * v3;

        const float k0_mag = sqrtf(k0_re * k0_re + k0_im * k0_im);
        const float k0_phase = atan2f(k0_im, k0_re);
        const float k1_mag = sqrtf(k1_re * k1_re + k1_im * k1_im);
        const float k1_phase = atan2f(k1_im, k1_re);

        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            const int f = block_tid * 2 + sub;
            const float k_mag   = (sub == 0) ? k0_mag   : k1_mag;
            const float k_phase = (sub == 0) ? k0_phase : k1_phase;

            const float* c = c_head + f * 3;
            const float c_re  = c[0];
            const float c_im  = c[1];
            const float c_abs = c[2];
            const float c_mag   = sqrtf(c_re * c_re + c_im * c_im);
            const float c_phase = atan2f(c_im, c_re);

            float omega = 0.0f;
            if (f < n_rot_bands) {
                const float exponent = -2.0f * (float)f / (float)n_rot;
                omega = expf(exponent * logf(rope_theta));
            }

            const float angle = omega * p_q + c_phase - k_phase;
            s_trig += c_mag * k_mag * cosf(angle);

            float r = (c_abs > 1e-20f) ? (c_mag / c_abs) : 0.0f;
            if (r > 1.0f) r = 1.0f;
            s_norm += (1.0f - r) * c_abs * k_mag;
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        s_trig += __shfl_xor(s_trig, off);
        s_norm += __shfl_xor(s_norm, off);
    }
    if (tid == 0) {
        scores[(size_t)h * seq_len + pos] = s_trig + s_norm;
    }
}
