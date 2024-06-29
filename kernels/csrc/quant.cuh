#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#define FP16_MAX 6.550400e+004
#define FP16_MIN -6.000400e+004
#define WARP_SIZE 32

// #define DEBUG_KERNEL

__device__ __forceinline__ int32_t cdiv(int32_t c, int32_t divisor) { return (c + divisor - 1) / divisor; }

template <typename T, typename U>
inline __device__ T cu_round(U x) {
  uint32_t y{};
  if constexpr (std::is_same_v<U, half>) {
    if constexpr (std::is_same_v<T, uint8_t>)
      asm("cvt.rni.sat.u8.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    else if constexpr (std::is_same_v<T, uint16_t>)
      asm("cvt.rni.sat.u16.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    else if constexpr (std::is_same_v<T, uint32_t>)
      asm("cvt.rni.sat.u32.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    else
      static_assert(!std::is_same_v<T, T>, "not implemented");

  } else if constexpr (std::is_same_v<U, float>) {
    if constexpr (std::is_same_v<T, uint8_t>)
      asm("cvt.rni.sat.u8.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    else if constexpr (std::is_same_v<T, uint16_t>)
      asm("cvt.rni.sat.u16.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    else if constexpr (std::is_same_v<T, uint32_t>)
      asm("cvt.rni.sat.u32.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    else
      static_assert(!std::is_same_v<T, T>, "not implemented");
  }
  return y;
}

template <typename T>
__device__ __forceinline__ T cu_add(T a, T b) {
  return a + b;
}

template <typename T>
__device__ __forceinline__ T cu_max(T a, T b) {
  if constexpr (std::is_same_v<T, half>)
    return __hmax(a, b);
  else if constexpr (std::is_same_v<T, float>)
    return fmaxf(a, b);
  else if constexpr (std::is_same_v<T, double>)
    return fmax(a, b);
  else
    static_assert(!std::is_same_v<T, T>, "not implemented");
}

template <typename T>
__device__ __forceinline__ T cu_min(T a, T b) {
  if constexpr (std::is_same_v<T, half>)
    return __hmin(a, b);
  else if constexpr (std::is_same_v<T, float>)
    return fminf(a, b);
  else if constexpr (std::is_same_v<T, double>)
    return fmin(a, b);
  else
    static_assert(!std::is_same_v<T, T>, "not implemented");
}

template <typename T, T (*Reduce)(T, T), int32_t WarpSize = 32>
__device__ __forceinline__ T warp_reduce(T val) {
  // less than 32 and is power of 2
  static_assert(WarpSize <= 32 && ((WarpSize & (WarpSize - 1)) == 0), "warpsize must <= 32 and is power of 2");
#pragma unroll
  for (int32_t i = WarpSize / 2; i >= 1; i /= 2) {
    val = Reduce(__shfl_xor_sync(0xffffffff, val, i, WarpSize), val);
  }
  return val;
}

/// QBITS: `QBITS==1` will be interpreted as `QBITS = 1.5`, i.e. ternary quantization; 1bit is not supported
template <int32_t BLOCK_SIZE = 128, int32_t QBITS = 8, typename T_Store = uint8_t, typename T_Data = half,
          typename T_QParam = half, typename T_Sim = uint32_t>
__global__ void reorder_quant_pack(
    // input
    const T_Data* tensor, const int16_t* group_st_idx, const int16_t* pack_group_st_idx,
    // input
    const int16_t* reorder_idx, const T_Data* smooth_scale,
    // input
    const float clipping,
    // input
    const int32_t HIDDEN, const int32_t PACK_HIDDEN,
    // input
    const bool EXPLICIT_REORDER, const bool SAVE_REORDER,
    // input
    const bool FAKE_QUANT,
    // output
    T_QParam* gscale, T_QParam* gzp, T_Store* pack_res, T_Data* fake_quant_res,
    // debug output
    T_Data* gmin, T_Data* gmax, T_Sim* no_pack
    //
) {
  static_assert(QBITS == 16 || (QBITS <= 8 && QBITS >= 1), "`QBITS` must be integer in [1, 8]");
  constexpr int32_t STORE_WIDTH = sizeof(T_Store) * 8;
  // now there are waste for [7, 6, 5, 3, 1.5] bits
  constexpr auto PACK_NUM = [] {
    if constexpr (QBITS > 1)
      return STORE_WIDTH / QBITS;
    else
      return STORE_WIDTH / 2;
  }();

  static_assert(PACK_NUM <= WARP_SIZE);
  const bool USE_SMOOTH = smooth_scale != nullptr;

  // bs_token_index
  auto btidx = blockIdx.x;
  // quant group_index
  auto gidx = blockIdx.y;
  // thread_index within a block
  auto tidx    = threadIdx.x;
  auto lane_id = tidx % WARP_SIZE;
  auto warp_id = tidx / WARP_SIZE;

  auto num_groups = gridDim.y;

  constexpr auto NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  // constexpr auto MAXINT    = QBITS >= 2 ? T_Data((1 << QBITS) - 1) : T_Data(2);
  constexpr auto MAXINT = [] {
    if constexpr (QBITS > 1)
      return (1 << QBITS) - 1;
    else
      return 2;
  }();
  auto max_int = T_Data(MAXINT);

  const auto* token_data = tensor + btidx * HIDDEN;

  // if (gidx == 0 && (btidx == 0) || (btidx == 1) && tidx < 10) {
  //   printf("btidx-%d, tidx-%d, token_data val: %f\n", btidx, tidx, float(token_data[tidx]));
  // }

  auto gst   = group_st_idx[gidx];
  auto ged   = group_st_idx[gidx + 1];
  auto gsize = ged - gst;
  // how many blocks there are in a group
  auto nblocks = cdiv(gsize, BLOCK_SIZE);

  // if(btidx == 0 && tidx == 0) {
  //   printf("gst-%d, ged-%d, gsize-%d, USE_SMOOTH-%d\n", gst, ged, gsize, int(USE_SMOOTH));
  // }

  extern __shared__ T_Data group_data[];

  // minmax value within a block
  __shared__ T_Data block_min[NUM_WARPS];
  __shared__ T_Data block_max[NUM_WARPS];
  if (lane_id == 0) {
    block_min[warp_id] = T_Data{FP16_MAX};
    block_max[warp_id] = T_Data{FP16_MIN};
  }
  __syncthreads();

  for (auto i = 0; i < nblocks; i++) {
    // 1. get-minmax: load reordered data from HBM to register
    auto group_offset   = tidx + i * BLOCK_SIZE;
    auto smooth_factor  = T_Data{1.0};
    auto block_data_max = T_Data{FP16_MIN};
    auto block_data_min = T_Data{FP16_MAX};
    if (group_offset < gsize) {
      auto value_index = int16_t{};
      if (EXPLICIT_REORDER) {
        value_index = reorder_idx[gst + group_offset];
      } else {
        value_index = gst + group_offset;
      }

      if (USE_SMOOTH) {
        smooth_factor = smooth_scale[value_index];
      }

      block_data_max = token_data[value_index] * smooth_factor;
      block_data_min = block_data_max;

      group_data[group_offset] = block_data_max;
    }
    __syncthreads();

    // 2. get-minmax: warp reduction
    auto warp_max = warp_reduce<T_Data, cu_max>(block_data_max);
    auto warp_min = warp_reduce<T_Data, cu_min>(block_data_min);

    if (lane_id == 0) {
      auto prev_min      = block_min[warp_id];
      auto prev_max      = block_max[warp_id];
      block_min[warp_id] = cu_min(warp_min, prev_min);
      block_max[warp_id] = cu_max(warp_max, prev_max);

    }

    __syncthreads();
  }
  // get-minmax: 3. block reduction
  auto group_max = T_Data{FP16_MIN};
  auto group_min = T_Data{FP16_MAX};
  if (lane_id < NUM_WARPS) {
    group_max = block_max[lane_id];
    group_min = block_min[lane_id];
  }
  __syncthreads();

  group_max = warp_reduce<T_Data, cu_max>(group_max);
  group_min = warp_reduce<T_Data, cu_min>(group_min);  // zp
  group_max *= T_Data(clipping);
  group_min *= T_Data(clipping);

#ifdef DEBUG_KERNEL
  auto* token_gmin = gmin + btidx * num_groups;
  auto* token_gmax = gmax + btidx * num_groups;
  token_gmax[gidx] = group_max;
  token_gmin[gidx] = group_min;
#endif

  auto zp    = group_min;
  auto scale = (group_max - group_min) / max_int;
  scale      = scale < T_Data{1e-5} ? T_Data{1e-5} : scale;


  if (!FAKE_QUANT) {
    auto* token_gscale = gscale + btidx * num_groups;
    auto* token_gzp    = gzp + btidx * num_groups;
    if (tidx == 0) {
      token_gzp[gidx]    = T_QParam(zp);
      token_gscale[gidx] = T_QParam(scale);
    }
  }

  for (auto i = 0; i < nblocks; i++) {
    auto group_offset = tidx + i * BLOCK_SIZE;
    auto val          = T_Data{0};
    if (group_offset < gsize) {
      auto rounded = T_Sim{};
      auto val     = group_data[group_offset];

      if constexpr (QBITS >= 16) {
        rounded = *(T_Sim*)(&val);
      } else {
        auto scaled_val = cu_min(cu_max((val - zp) / scale, T_Data{0}), max_int);
        rounded = __half2uint_rn(scaled_val);
      }

#ifdef DEBUG_KERNEL
      auto* token_no_pack_sim               = no_pack + btidx * HIDDEN;
      token_no_pack_sim[gst + group_offset] = rounded;
#endif
      if (FAKE_QUANT) {
        // simulate datatype convertion loss
        zp    = T_Data(T_QParam(zp));
        scale = T_Data(T_QParam(scale));

        auto dequant = T_Data(rounded) * scale + zp;

        auto value_index = int16_t{};
        if (EXPLICIT_REORDER) {
          value_index = reorder_idx[gst + group_offset];
        } else {
          value_index = gst + group_offset;
        }

        auto smooth_val = T_Data{1.0};
        if (USE_SMOOTH) {
          smooth_val = smooth_scale[value_index];
        }
        dequant = dequant / smooth_val;

        auto* token_fake_quant        = fake_quant_res + btidx * HIDDEN;
        token_fake_quant[value_index] = dequant;
      } else {
        auto* token_pack   = pack_res + btidx * PACK_HIDDEN;
        auto pack_group_st = pack_group_st_idx[gidx];

        rounded   = rounded << ((PACK_NUM - 1 - (tidx % PACK_NUM)) * (STORE_WIDTH / PACK_NUM));
        auto pack = T_Store(warp_reduce<T_Sim, cu_add, PACK_NUM>(rounded));

        // write to HBM
        if (tidx % PACK_NUM == 0) {
          auto pack_id = group_offset / PACK_NUM + pack_group_st;
          token_pack[pack_id] = pack;
        }
        __syncthreads();
      }
    }
  }
}

/// QBITS: `QBITS==1` will be interpreted as `QBITS = 1.5`, i.e. ternary quantization; 1bit is not supported
template <int32_t BLOCK_SIZE = 128, int32_t QBITS = 8, typename T_Store = uint8_t, typename T_Data = half,
          typename T_QParam = half, typename T_Sim = uint32_t>
__global__ void reorder_dequant_unpack(
    // input
    const T_Store* pack, const int16_t* group_st_idx, const int16_t* pack_group_st_idx,
    // input
    T_QParam* gscale, T_QParam* gzp,
    // input
    const int16_t* reorder_idx, const T_Data* smooth_scale,
    // input
    const int32_t HIDDEN, const int32_t PACK_HIDDEN,
    // input
    const bool EXPLICIT_REORDER,
    // output
    T_Data* out
    //
) {
  constexpr int32_t STORE_WIDTH = sizeof(T_Store) * 8;
  constexpr auto PACK_NUM = [] {
    if constexpr (QBITS > 1)
      return STORE_WIDTH / QBITS;
    else
      return STORE_WIDTH / 2;
  }();

  static_assert(PACK_NUM <= WARP_SIZE);
  auto STORE_BITS = (STORE_WIDTH / PACK_NUM);

  const bool USE_SMOOTH = smooth_scale != nullptr;

  // bs_token_index
  auto btidx = blockIdx.x;
  // number of group in each token
  auto num_groups = gridDim.y;
  // quant group_index
  auto gidx = blockIdx.y;
  // thread_index within a block
  auto tidx = threadIdx.x;

  auto unpack_gst   = group_st_idx[gidx];
  auto unpack_gsize = group_st_idx[gidx + 1] - unpack_gst;

  const auto* pack_token = pack + btidx * PACK_HIDDEN;
  const auto* pack_group = pack_token + pack_group_st_idx[gidx];
  auto* out_token        = out + btidx * HIDDEN;

  auto nblocks = cdiv(unpack_gsize, BLOCK_SIZE);

  auto qmask = (1 << QBITS) - 1;
  auto scale = gscale[gidx + btidx * num_groups];
  auto zp    = gzp[gidx + btidx * num_groups];

  for (auto i = 0; i < nblocks; i++) {
    int16_t group_offset = tidx + i * BLOCK_SIZE;
    if (group_offset < unpack_gsize) {
      auto pack_val_offset = PACK_NUM - 1 - (group_offset % PACK_NUM);
      auto pack_val        = pack_group[group_offset / PACK_NUM];
      auto rounded         = (pack_val >> (pack_val_offset * STORE_BITS)) & qmask;
      auto smooth_factor   = T_Data{1.0};
      if (USE_SMOOTH) {
        smooth_factor = smooth_scale[unpack_gst + group_offset];
        if (EXPLICIT_REORDER) {
          smooth_factor = smooth_scale[reorder_idx[unpack_gst + group_offset]];
        }
      }

      auto dequant = (T_Data(rounded) * T_Data(scale) + T_Data(zp)) / smooth_factor;

      auto write_group_offset = group_offset;
      if (EXPLICIT_REORDER) {
        write_group_offset = reorder_idx[unpack_gst + group_offset];
      }
      out_token[write_group_offset] = dequant;
    }
  }
}
