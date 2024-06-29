#include <torch/extension.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/arange.h"
#include "ATen/ops/from_blob.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Float8_e4m3fn.h"
#include "c10/util/Half.h"
#include "quant.cuh"
#include "skvq_quant.h"

using torch::Tensor;

//<BLOCK_SIZE, QBITS, T_Store, T_Data, T_QParam, T_Sim>
auto quant_pack_q4   = &reorder_quant_pack<128, 4, uint8_t, half, half, uint32_t>;
auto quant_pack_q3   = &reorder_quant_pack<128, 3, uint8_t, half, half, uint32_t>;
auto quant_pack_q2   = &reorder_quant_pack<128, 2, uint8_t, half, half, uint32_t>;
auto quant_pack_q1_5 = &reorder_quant_pack<128, 1, uint8_t, half, half, uint32_t>;

auto quant_pack_q4_fp8   = &reorder_quant_pack<128, 4, uint8_t, half, at::Float8_e4m3fn, uint32_t>;
auto quant_pack_q3_fp8   = &reorder_quant_pack<128, 3, uint8_t, half, at::Float8_e4m3fn, uint32_t>;
auto quant_pack_q2_fp8   = &reorder_quant_pack<128, 2, uint8_t, half, at::Float8_e4m3fn, uint32_t>;
auto quant_pack_q1_5_fp8 = &reorder_quant_pack<128, 1, uint8_t, half, at::Float8_e4m3fn, uint32_t>;

auto dequant_unpack_q4   = &reorder_dequant_unpack<128, 4, uint8_t, half, half, uint32_t>;
auto dequant_unpack_q3   = &reorder_dequant_unpack<128, 3, uint8_t, half, half, uint32_t>;
auto dequant_unpack_q2   = &reorder_dequant_unpack<128, 2, uint8_t, half, half, uint32_t>;
auto dequant_unpack_q1_5 = &reorder_dequant_unpack<128, 1, uint8_t, half, half, uint32_t>;

auto dequant_unpack_q4_fp8   = &reorder_dequant_unpack<128, 4, uint8_t, half, at::Float8_e4m3fn, uint32_t>;
auto dequant_unpack_q3_fp8   = &reorder_dequant_unpack<128, 3, uint8_t, half, at::Float8_e4m3fn, uint32_t>;
auto dequant_unpack_q2_fp8   = &reorder_dequant_unpack<128, 2, uint8_t, half, at::Float8_e4m3fn, uint32_t>;
auto dequant_unpack_q1_5_fp8 = &reorder_dequant_unpack<128, 1, uint8_t, half, at::Float8_e4m3fn, uint32_t>;

using quant_func       = decltype(quant_pack_q4);
using quant_func_fp8   = decltype(quant_pack_q4_fp8);
using dequant_func     = decltype(dequant_unpack_q4);
using dequant_func_fp8 = decltype(dequant_unpack_q4_fp8);

static auto QUANT_DISPATCHER = std::unordered_map<float, quant_func>{
    {4., quant_pack_q4},
    {3., quant_pack_q3},
    {2., quant_pack_q2},
    {1.5, quant_pack_q1_5},
};
static auto QUANT_DISPATCHER_FP8 = std::unordered_map<float, quant_func_fp8>{
    {4., quant_pack_q4_fp8},
    {3., quant_pack_q3_fp8},
    {2., quant_pack_q2_fp8},
    {1.5, quant_pack_q1_5_fp8},
};

static auto DEQUANT_DISPATCHER = std::unordered_map<float, dequant_func>{
    {4., dequant_unpack_q4},
    {3., dequant_unpack_q3},
    {2., dequant_unpack_q2},
    {1.5, dequant_unpack_q1_5},
};
static auto DEQUANT_DISPATCHER_FP8 = std::unordered_map<float, dequant_func_fp8>{
    {4., dequant_unpack_q4_fp8},
    {3., dequant_unpack_q3_fp8},
    {2., dequant_unpack_q2_fp8},
    {1.5, dequant_unpack_q1_5_fp8},
};

auto skvq_quant_fake(                   //
    Tensor data,                        //
    at::optional<Tensor> group_st_idx,  //
    at::optional<Tensor> reorder_idx,   //
    at::optional<Tensor> smooth_scale,  //
    const float qbits,                  //
    const int32_t gsize,                //
    const int32_t hidden,               //
    const bool fp8,                     //
    const float clipping                //
    ) -> Tensor {
  assert(data.is_contiguous());

  auto use_reorder = reorder_idx.has_value();

  if (use_reorder) assert(group_st_idx.has_value());

  auto num_groups = use_reorder ? group_st_idx.value().size(0) - 1 : hidden / gsize;

  auto* tensor = reinterpret_cast<half*>(data.data_ptr<at::Half>());

  // get `group_start_index`
  if (!group_st_idx.has_value())
    group_st_idx = torch::arange(0, hidden + 1, gsize, at::dtype(at::kShort).device(data.device()));
  auto* gst_idx = group_st_idx.value().data_ptr<int16_t>();

  // get `reorder_index`
  auto* rod_idx = (int16_t*){nullptr};
  if (reorder_idx.has_value()) rod_idx = reorder_idx.value().data_ptr<int16_t>();

  // get `smooth_scale`
  auto* smooth = (half*){nullptr};
  if (smooth_scale.has_value()) smooth = reinterpret_cast<half*>(smooth_scale.value().data_ptr<at::Half>());

  // get `max_group_size`
  auto gst_idx_cpu     = group_st_idx.value().cpu();
  auto* gst_ptr_cpu    = gst_idx_cpu.data_ptr<int16_t>();
  auto each_group_size = std::vector<int32_t>(num_groups);
  for (auto i = 0; i < num_groups; i++)
    each_group_size[i] = gst_ptr_cpu[i + 1] - gst_ptr_cpu[i];
  auto max_group_size = *std::max_element(begin(each_group_size), end(each_group_size));

  // allocate for return tensor
  auto fake_dequant      = torch::empty_like(data);
  auto* fake_dequant_ptr = reinterpret_cast<half*>(fake_dequant.data_ptr<at::Half>());

  // clang-format on
  auto in_shape  = data.sizes();
  auto bs        = in_shape[0];
  auto seq_len   = in_shape[1];
  auto num_heads = in_shape[2];
  auto head_dim  = in_shape[3];

  auto blocks_per_grid   = dim3(bs * seq_len, num_groups);
  auto threads_per_block = dim3(128);

  if (fp8) {
    auto kernel_fp8 = QUANT_DISPATCHER_FP8[qbits];
    kernel_fp8<<<blocks_per_grid, threads_per_block, max_group_size * 2>>>(
        /*input*/ tensor,
        /*group_st_idx*/ gst_idx,
        /*packed_gst_idx*/ nullptr,
        /*reorder_index*/ rod_idx,
        /*smooth_scale*/ smooth,
        /*clipping*/ clipping,
        /*HIDDEN*/ hidden,
        /*PACK_HIDDEN*/ 0,
        /*EXPLICIT_REORDER*/ use_reorder,
        /*SAVE_REORDER*/ true,
        /*FAKE_QUANT*/ true,

        /*gscale*/ nullptr,
        /*gzp*/ nullptr,
        /*pack_res*/ nullptr,

        /*fake_quant_out*/ fake_dequant_ptr,

        /*gmin*/ nullptr,
        /*gmax*/ nullptr,
        /*no_pack*/ nullptr);
  } else {
    auto kernel = QUANT_DISPATCHER[qbits];
    kernel<<<blocks_per_grid, threads_per_block, max_group_size * 2>>>(
        /*input*/ tensor,
        /*group_st_idx*/ gst_idx,
        /*packed_gst_idx*/ nullptr,
        /*reorder_index*/ rod_idx,
        /*smooth_scale*/ smooth,
        /*clipping*/ clipping,
        /*HIDDEN*/ hidden,
        /*PACK_HIDDEN*/ 0,
        /*EXPLICIT_REORDER*/ use_reorder,
        /*SAVE_REORDER*/ true,
        /*FAKE_QUANT*/ true,

        /*gscale*/ nullptr,
        /*gzp*/ nullptr,
        /*pack_res*/ nullptr,

        /*fake_quant_out*/ fake_dequant_ptr,

        /*gmin*/ nullptr,
        /*gmax*/ nullptr,
        /*no_pack*/ nullptr);
  }

  return fake_dequant;
}

/// @return pack_res, gscale, gzp
auto skvq_quant_pack(                   //
    Tensor data,                        //
    at::optional<Tensor> group_st_idx,  //
    at::optional<Tensor> reorder_idx,   //
    at::optional<Tensor> smooth_scale,  //
    const float qbits,                  //
    const int32_t gsize,                //
    const int32_t hidden,               //
    const bool fp8,                     //
    const float clipping                //
    ) -> std::tuple<Tensor, Tensor, Tensor> {
  assert(data.is_contiguous());

  auto dev         = data.device();
  auto in_shape    = data.sizes();
  auto bs          = in_shape[0];
  auto seq_len     = in_shape[1];
  auto num_heads   = in_shape[2];
  auto head_dim    = in_shape[3];
  auto use_reorder = reorder_idx.has_value();
  auto use_smooth  = smooth_scale.has_value();

  if (use_reorder) assert(group_st_idx.has_value());

  auto num_groups = use_reorder ? group_st_idx.value().size(0) - 1 : hidden / gsize;

  auto* tensor = reinterpret_cast<half*>(data.data_ptr<at::Half>());

  // get `group_start_index`
  auto gst_option = at::dtype(at::kShort).device(dev);
  if (!group_st_idx.has_value()) group_st_idx = torch::arange(0, hidden + 1, gsize, gst_option);
  auto* gst_idx_ptr = group_st_idx.value().data_ptr<int16_t>();

  // get `reorder_index`
  auto* rod_idx = (int16_t*){nullptr};
  if (reorder_idx.has_value()) rod_idx = reorder_idx.value().data_ptr<int16_t>();

  // get `smooth_scale`
  auto* smooth = (half*){nullptr};
  if (smooth_scale.has_value()) smooth = reinterpret_cast<half*>(smooth_scale.value().data_ptr<at::Half>());

  // get `max_group_size`
  auto gst_idx_cpu  = group_st_idx.value().cpu();
  auto* gst_ptr_cpu = gst_idx_cpu.data_ptr<int16_t>();
  auto each_gsize   = std::vector<int32_t>(num_groups);
  for (auto i = 0; i < num_groups; i++)
    each_gsize[i] = gst_ptr_cpu[i + 1] - gst_ptr_cpu[i];
  auto max_group_size = *std::max_element(begin(each_gsize), end(each_gsize));

  // get `pack_gst_idx`
  auto pack_num        = sizeof(uint8_t) * 8 / qbits;
  auto each_pack_gsize = std::vector<int16_t>(num_groups);
  std::transform(begin(each_gsize), end(each_gsize), begin(each_pack_gsize),
                 [pack_num](auto gsize) { return ((gsize + pack_num - 1) / pack_num); });
  auto pack_gst_idx_cpu = std::vector<int16_t>(num_groups + 1, 0);
  std::inclusive_scan(begin(each_pack_gsize), end(each_pack_gsize), begin(pack_gst_idx_cpu) + 1);
  auto pack_hidden = pack_gst_idx_cpu.back();

  auto pack_gst_idx  = torch::from_blob(pack_gst_idx_cpu.data(), {num_groups + 1}, at::dtype(at::kShort));
  pack_gst_idx       = pack_gst_idx.to(gst_option.device());
  auto* pack_gst_ptr = pack_gst_idx.data_ptr<int16_t>();

  // allocate for return value
  auto pack_option   = at::dtype(at::kByte).device(dev);
  auto qparam_option = at::dtype(fp8 ? at::kFloat8_e4m3fn : at::kHalf).device(dev);
  auto pack          = torch::empty({bs, seq_len, pack_hidden}, pack_option);
  auto scale         = torch::empty({bs, seq_len, num_groups}, qparam_option);
  auto zp            = torch::empty({bs, seq_len, num_groups}, qparam_option);

  auto blocks_per_grid   = dim3(bs * seq_len, num_groups);
  auto threads_per_block = dim3(128);

  if (fp8) {
    auto* pack_ptr  = pack.data_ptr<uint8_t>();
    auto* scale_ptr = scale.data_ptr<at::Float8_e4m3fn>();
    auto* zp_ptr    = zp.data_ptr<at::Float8_e4m3fn>();
    auto kernel     = QUANT_DISPATCHER_FP8[qbits];
    kernel<<<blocks_per_grid, threads_per_block, max_group_size * 2>>>(
        /*input*/ tensor,
        /*group_st_idx*/ gst_idx_ptr,
        /*packed_gst_idx*/ pack_gst_ptr,
        /*reorder_index*/ rod_idx,
        /*smooth_scale*/ smooth,
        /*clipping*/ clipping,
        /*HIDDEN*/ hidden,
        /*PACK_HIDDEN*/ pack_hidden,
        /*EXPLICIT_REORDER*/ use_reorder,
        /*SAVE_REORDER*/ true,
        /*FAKE_QUANT*/ false,

        /*gscale*/ scale_ptr,
        /*gzp*/ zp_ptr,
        /*pack_res*/ pack_ptr,

        /*fake_quant_out*/ nullptr,

        /*gmin*/ nullptr,
        /*gmax*/ nullptr,
        /*no_pack*/ nullptr);
  } else {
    auto* pack_ptr  = pack.data_ptr<uint8_t>();
    auto* scale_ptr = reinterpret_cast<half*>(scale.data_ptr<at::Half>());
    auto* zp_ptr    = reinterpret_cast<half*>(zp.data_ptr<at::Half>());
    auto kernel     = QUANT_DISPATCHER[qbits];
    kernel<<<blocks_per_grid, threads_per_block, max_group_size * 2>>>(
        /*input*/ tensor,
        /*group_st_idx*/ gst_idx_ptr,
        /*packed_gst_idx*/ pack_gst_ptr,
        /*reorder_index*/ rod_idx,
        /*smooth_scale*/ smooth,
        /*clipping*/ clipping,
        /*HIDDEN*/ hidden,
        /*PACK_HIDDEN*/ pack_hidden,
        /*EXPLICIT_REORDER*/ use_reorder,
        /*SAVE_REORDER*/ true,
        /*FAKE_QUANT*/ false,

        /*gscale*/ scale_ptr,
        /*gzp*/ zp_ptr,
        /*pack_res*/ pack_ptr,

        /*fake_quant_out*/ nullptr,

        /*gmin*/ nullptr,
        /*gmax*/ nullptr,
        /*no_pack*/ nullptr);
  }

  return std::make_tuple(std::move(pack), std::move(scale), std::move(zp));
}

auto skvq_dequant_unpack(                      //
    torch::Tensor pack_data,                   //
    torch::Tensor scale,                       //
    torch::Tensor zp,                          //
    at::optional<torch::Tensor> group_st_idx,  //
    at::optional<torch::Tensor> reorder_idx,   //
    at::optional<torch::Tensor> smooth_scale,  //
    const float qbits,                         //
    const int32_t gsize,                       //
    const int32_t hidden,                      //
    const bool fp8                             //
    ) -> Tensor {
  assert(pack_data.is_contiguous());
  assert(scale.is_contiguous());
  assert(zp.is_contiguous());

  auto dev         = pack_data.device();
  auto in_shape    = pack_data.sizes();
  auto bs          = in_shape[0];
  auto seq_len     = in_shape[1];
  auto pack_hidden = in_shape[2];
  auto* pack_ptr   = pack_data.data_ptr<uint8_t>();

  auto use_reorder = reorder_idx.has_value();
  auto use_smooth  = smooth_scale.has_value();
  if (use_reorder) assert(group_st_idx.has_value());

  auto num_groups = use_reorder ? group_st_idx.value().size(0) - 1 : hidden / gsize;

  // get `group_start_index`
  auto gst_option = at::dtype(at::kShort).device(dev);
  if (!group_st_idx.has_value()) group_st_idx = torch::arange(0, hidden + 1, gsize, gst_option);
  auto* gst_idx_ptr = group_st_idx.value().data_ptr<int16_t>();

  // get `reorder_index`
  auto* rod_idx_ptr = (int16_t*){nullptr};
  if (reorder_idx.has_value()) rod_idx_ptr = reorder_idx.value().data_ptr<int16_t>();

  // get `smooth_scale`
  auto* smooth_ptr = (half*){nullptr};
  if (smooth_scale.has_value()) smooth_ptr = reinterpret_cast<half*>(smooth_scale.value().data_ptr<at::Half>());

  // get `max_group_size`
  auto gst_idx_cpu  = group_st_idx.value().cpu();
  auto* gst_ptr_cpu = gst_idx_cpu.data_ptr<int16_t>();
  auto each_gsize   = std::vector<int32_t>(num_groups);
  for (auto i = 0; i < num_groups; i++)
    each_gsize[i] = gst_ptr_cpu[i + 1] - gst_ptr_cpu[i];
  auto max_group_size = *std::max_element(begin(each_gsize), end(each_gsize));

  // get `pack_gst_idx`
  const auto pack_num  = sizeof(uint8_t) * 8 / qbits;
  auto each_pack_gsize = std::vector<int16_t>(num_groups);
  std::transform(begin(each_gsize), end(each_gsize), begin(each_pack_gsize),
                 [pack_num](auto gsize) { return ((gsize + pack_num - 1) / pack_num); });
  auto pack_gst_idx_cpu = std::vector<int16_t>(num_groups + 1, 0);
  std::inclusive_scan(begin(each_pack_gsize), end(each_pack_gsize), begin(pack_gst_idx_cpu) + 1);

  auto pack_gst_idx  = torch::from_blob(pack_gst_idx_cpu.data(), {num_groups + 1}, at::dtype(at::kShort));
  pack_gst_idx       = pack_gst_idx.to(gst_option.device());
  auto* pack_gst_ptr = pack_gst_idx.data_ptr<int16_t>();

  // allocate for return value
  auto unpack      = torch::empty({bs, seq_len, hidden}, at::dtype(at::kHalf).device(dev));
  auto* unpack_ptr = reinterpret_cast<half*>(unpack.data_ptr<torch::Half>());

  auto blocks_per_grid   = dim3(bs * seq_len, num_groups);
  auto threads_per_block = dim3(128);
  if (fp8) {
    auto kernel     = DEQUANT_DISPATCHER_FP8[qbits];
    auto* scale_ptr = scale.data_ptr<at::Float8_e4m3fn>();
    auto* zp_ptr    = zp.data_ptr<at::Float8_e4m3fn>();
    kernel<<<blocks_per_grid, threads_per_block>>>(
        /*packed data*/ pack_ptr,
        /*group_st_idx*/ gst_idx_ptr,
        /*pack_group_st_idx*/ pack_gst_ptr,
        /*scale*/ scale_ptr,
        /*zp*/ zp_ptr,
        /*reorder_index*/ rod_idx_ptr,
        /*smooth_cale*/ smooth_ptr,
        /*hidden*/ hidden,
        /*packed_hidden*/ pack_hidden,
        /*use_reorder*/ use_reorder,
        /*output*/ unpack_ptr);

  } else {
    auto kernel     = DEQUANT_DISPATCHER[qbits];
    auto* scale_ptr = reinterpret_cast<half*>(scale.data_ptr<at::Half>());
    auto* zp_ptr    = reinterpret_cast<half*>(zp.data_ptr<at::Half>());

    kernel<<<blocks_per_grid, threads_per_block>>>(
        /*packed data*/ pack_ptr,
        /*group_st_idx*/ gst_idx_ptr,
        /*pack_group_st_idx*/ pack_gst_ptr,
        /*scale*/ scale_ptr,
        /*zp*/ zp_ptr,
        /*reorder_index*/ rod_idx_ptr,
        /*smooth_cale*/ smooth_ptr,
        /*hidden*/ hidden,
        /*packed_hidden*/ pack_hidden,
        /*use_reorder*/ use_reorder,
        /*output*/ unpack_ptr);
  }

  return unpack;
}