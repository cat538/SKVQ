#pragma once
#include <torch/extension.h>

// clang-format off

/// data -> quant -> dequant
/// @return: dequant data
auto skvq_quant_fake(
    // input
    torch::Tensor data,
    at::optional<torch::Tensor> gst_idx,
    at::optional<torch::Tensor> reorder_idx,
    at::optional<torch::Tensor> smooth_scale,
    const float qbits,
    const int32_t gsize,
    const int32_t hidden,
    const bool fp8,
    const float clipping
) -> torch::Tensor;


/// @return: [quant_data, scale, zp]
auto skvq_quant_pack(
    torch::Tensor data,
    at::optional<torch::Tensor> gst_idx,
    at::optional<torch::Tensor> reorder_idx,
    at::optional<torch::Tensor> smooth_scale,
    const float qbits,
    const int32_t gsize,
    const int32_t hidden,
    const bool fp8,
    const float clipping
) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;


/// @return: dequant data
auto skvq_dequant_unpack(
    // input
    torch::Tensor pack_data,
    torch::Tensor scale,
    torch::Tensor zp,
    at::optional<torch::Tensor> gst_idx,
    at::optional<torch::Tensor> reorder_idx,
    at::optional<torch::Tensor> smooth_scale,
    const float qbits,
    const int32_t gsize,
    const int32_t hidden,
    const bool fp8
) -> torch::Tensor;