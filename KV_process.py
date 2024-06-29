import math
import torch
import torch.nn as nn
import operator

import skvq_quant

from typing import Literal, TypeAlias
from transformers import PreTrainedModel, LlamaForCausalLM, MistralForCausalLM

RodMeta: TypeAlias = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]

class SKVQuantProcessor(nn.Module):
    def __init__(
        self,
        K_target_bitwidth,
        V_target_bitwidth,
        gsize: int,
        hidden: int,
        clipping: list[float] = None,
        reorder_meta: RodMeta = None,
        smooth_scale: RodMeta = None,
        KIVI_mode: bool = False,
        fp8:bool = False,
        fake_quant: bool = True,
        # model: PreTrainedModel = None,
    ) -> None:
        """
        reorder_meta: [reorder_indices, cluster_st_inds]
        """
        super().__init__()
        self.K_target_bitwidth = K_target_bitwidth
        self.V_target_bitwidth = V_target_bitwidth
        self.clipping = clipping
        if reorder_meta:
            reorder_meta[0]["k"] = reorder_meta[0]["k"].to(torch.int16)
            reorder_meta[0]["v"] = reorder_meta[0]["v"].to(torch.int16)
            reorder_meta[1]["k"] = reorder_meta[1]["k"].to(torch.int16)
            reorder_meta[1]["v"] = reorder_meta[1]["v"].to(torch.int16)

        self.reorder_idx = reorder_meta[0] if reorder_meta else None
        self.group_st_idx = reorder_meta[1] if reorder_meta else None
        self.smooth_scale = smooth_scale
        self.gsize = gsize
        self.fake_quant = fake_quant
        self.KIVI_mode = KIVI_mode
        self.fp8 = fp8
        self.pack_sim = True
        self.hidden = hidden
        self.layer_idx:int = None

    @torch.no_grad()
    def reorder_weight(
        self, model: PreTrainedModel, reorder_indices: list[torch.Tensor]
    ):
        proj_map = {
            LlamaForCausalLM: {
                "q": operator.attrgetter("self_attn.q_proj"),
                "k": operator.attrgetter("self_attn.k_proj"),
                "v": operator.attrgetter("self_attn.v_proj"),
                "o": operator.attrgetter("self_attn.o_proj"),
            },
            MistralForCausalLM: {
                "q": operator.attrgetter("self_attn.q_proj"),
                "k": operator.attrgetter("self_attn.k_proj"),
                "v": operator.attrgetter("self_attn.v_proj"),
                "o": operator.attrgetter("self_attn.o_proj"),
            },
        }
        wname_map: dict[str, str] = proj_map.get(model.__class__, None)
        assert wname_map is not None, f"Not supported for {model.__class__}"

        for idx, layer in enumerate(model.model.layers):
            for wtype, wgetter in wname_map.items():
                weight: nn.Parameter = wgetter(layer)
                dev = weight.weight.device
                # if wtype in ["q", "k"] and idx == 0:
                if wtype in ["k"]:
                    reorder_index = reorder_indices["k"][idx].to(dev)
                    weight.weight.data = weight.weight.data[reorder_index, :]
                if wtype == "v":
                    reorder_index = reorder_indices["v"][idx].to(dev)
                    weight.weight.data = weight.weight.data[reorder_index, :]
                # elif wtype == "o":
                #     reorder_index = reorder_indices["v"][idx].to(dev)
                #     weight.weight.data = weight.weight.data[:, reorder_index]

    def quant_pytorch(
        self,
        ttype: Literal["k", "v"],
        tensor: torch.Tensor,
    )->tuple[torch.Tensor, None, None]:
        bs, num_heads, seqlen, head_dim = tensor.shape
        assert num_heads * head_dim == self.hidden
        if seqlen == 0:
            return tensor, None, None

        dtype = tensor.dtype
        qbits = self.K_target_bitwidth if ttype == "k" else self.V_target_bitwidth
        per_channel = self.KIVI_mode and (ttype == "k")
        
        if round(qbits) == qbits:
            max_int = (1 << int(qbits)) - 1
        else:
            assert qbits == 1.5
            max_int = 2

        def pack_tensor(t: torch.Tensor, store_type=torch.uint8):
            '''
            pack along the last dim
            '''
            store_bits = torch.tensor([], dtype=torch.uint8).element_size() * 8
            val_bits = int(math.ceil(qbits))
            pack_num = store_bits // val_bits
            pack_gst_lis = [0]

            if self.group_st_idx is None:
                gst_lis = torch.arange(0, num_heads*head_dim+1, self.gsize)
            else:
                gst_lis = self.group_st_idx["k"]

            pgst = 0
            res = []
            for i in range(len(gst_lis)-1):
                gst, ged = gst_lis[i], gst_lis[i+1]
                gsize = ged - gst
                pgst += int(math.ceil(gsize / pack_num))
                pack_gst_lis.append(pgst)

                pack_group = []
                for j in range(gst, ged, pack_num):
                    pack_val = torch.zeros((bs, seqlen, 1), dtype=store_type).to(t.device)
                    for k in range(pack_num):
                        if j + k < ged:
                            pack_val += t[:,:, j + k:j+k+1].to(store_type) << ((pack_num - k - 1) * val_bits)
                    pack_group.append(pack_val)

                res.append(torch.cat(pack_group, dim=-1))

            res = torch.cat(res, dim=-1)

            return res

        def quant(t: torch.Tensor):
            """
            Asymmetric Dynamic Quantiztion
            Assume the last dim is reduction dim (group dim)

            return: (quant_tensor, scale, zp), shape of quant param: [bs, seqlen, 1]
            """
            if qbits == 16:
                return t, None, None
            gmin, gmax = t.aminmax(dim=-1, keepdim=True)
            clip_scale = self.clipping[self.layer_idx]
            gmin = gmin * clip_scale
            gmax = gmax * clip_scale
            zp = gmin
            scale = torch.clamp((gmax - gmin) / max_int, min=1e-5)
            # quant
            res = (
                t.sub(zp)
                .div(scale)
                .clamp(0, max_int)
                .round()
            )
            if self.fp8:
                scale = scale.to(torch.float8_e4m3fn).to(dtype)
                zp = zp.to(torch.float8_e4m3fn).to(dtype)

            if self.fake_quant:
                # dequant
                res = (
                    res.to(dtype)
                    .mul(scale)
                    .add(zp)
                )
                return res, None, None
            else:
                return res, scale, zp

        def back_to_original(t: torch.Tensor, reorder_indices: torch.Tensor):
            # https://stackoverflow.com/questions/52127723/pytorch-better-way-to-get-back-original-tensor-order-after-torch-sort
            return t.gather(-1, reorder_indices.argsort(-1).expand(t.shape))

        if per_channel:
            assert seqlen % self.gsize == 0
            assert self.fake_quant
            # [bs, num_heads, head_dim, seqlen]
            t_reshape = tensor.transpose(2, 3)
        else:
            t_reshape = tensor.transpose(1, 2).reshape(bs, seqlen, num_heads * head_dim)

        if self.smooth_scale is not None:
            assert not per_channel
            smooth_scale = self.smooth_scale[ttype].to(tensor.dtype).to(tensor.device)
            assert smooth_scale.shape[0] == self.hidden
            t_reshape = t_reshape.mul(smooth_scale)

        if self.reorder_idx is not None:
            assert not per_channel
            # reorder KV
            reorder_idx = self.reorder_idx[ttype].long().to(tensor.device)
            assert reorder_idx.shape[0] == self.hidden
            reordered_kv = t_reshape[..., reorder_idx]

            res = torch.empty_like(reordered_kv)
            gst = self.group_st_idx[ttype].long()

            if not self.fake_quant:
                gscale = []
                gzp = []

            # quant reordered KV
            for i in range(len(gst) - 1):
                qdata, scale, zp = quant(reordered_kv[..., gst[i] : gst[i + 1]])
                res[..., gst[i] : gst[i + 1]] = qdata

                if not self.fake_quant:
                    gscale.append(scale)
                    gzp.append(zp)
            
            if not self.fake_quant:
                scale = torch.cat(gscale, dim = -1)
                zp = torch.cat(gscale, dim = -1)

            if self.fake_quant:
                # back to original(ordered KV)
                res = back_to_original(res, reorder_idx)
                if self.smooth_scale is not None:
                    res = res.div(smooth_scale)

            kv_quant = res.reshape(bs, seqlen, num_heads, head_dim).transpose(1, 2)

            # if not self.fake_quant and self.pack_sim:
            #     kv_quant = pack_tensor(kv_quant.transpose(1,2).reshape(bs,seqlen, -1))

            return kv_quant, scale, zp

        else:
            t_reshape = t_reshape.reshape(bs, seqlen, num_heads * head_dim // self.gsize, self.gsize)
            qdata, scale, zp = quant(t_reshape)

            if not self.fake_quant:
                scale, zp = scale.squeeze(-1), zp.squeeze(-1)

            if self.fake_quant and self.smooth_scale is not None:
                qdata = qdata.reshape(bs, seqlen, -1).div(smooth_scale)

            if per_channel:
                kv_quant = qdata.reshape(bs, num_heads, head_dim, seqlen).transpose(2,3)
            else:
                kv_quant = qdata.reshape(bs, seqlen, num_heads, head_dim).transpose(1,2)

            # if not self.fake_quant and self.pack_sim:
            #     kv_quant = pack_tensor(kv_quant.transpose(1,2).reshape(bs,seqlen, -1))

            return kv_quant, scale, zp

    def quant_cuda(
        self,
        ttype: Literal["k", "v"],
        tensor: torch.Tensor,
    )->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        qbits = self.K_target_bitwidth if ttype == "k" else self.V_target_bitwidth
        bs, num_heads, seqlen, head_dim = tensor.shape
        if seqlen == 0:
            return tensor, None, None

        qbits = self.K_target_bitwidth if ttype == "k" else self.V_target_bitwidth
        assert not (self.KIVI_mode and (ttype == "k"))
        if qbits == 16:
            return tensor, None, None

        # [bs, seqlen, num_heads, head_dim]
        t_reshape = tensor.transpose(1, 2).contiguous()

        smooth_scale = None
        reorder_idx = None
        gst_idx = None
        if self.smooth_scale is not None:
            smooth_scale = self.smooth_scale[ttype].to(tensor.dtype).to(tensor.device).contiguous()
            assert smooth_scale.shape[0] == num_heads * head_dim
        if self.reorder_idx is not None:
            reorder_idx = self.reorder_idx[ttype].to(torch.int16).to(tensor.device)

            # print(f"cache device: {tensor.device}")
            gst_idx = self.group_st_idx[ttype].to(torch.int16).to(tensor.device)
            assert reorder_idx.shape[0] == num_heads * head_dim

        if self.fake_quant:
            # [bs, seq_len, num_heads, head_dim]
            fake_quant = skvq_quant.skvq_quant_fake(
                t_reshape,
                gst_idx,
                reorder_idx,
                smooth_scale,
                qbits, self.gsize, self.hidden, self.fp8, self.clipping[self.layer_idx],
            )
            # [bs, num_heads, seq_len, head_dim]
            fake_quant = fake_quant.transpose(1,2).contiguous()

            return fake_quant, None, None

        # [bs, seq_len, pack_hidden], [bs, seq_len, num_groups]
        pack, scale, zp = skvq_quant.skvq_quant_pack(
            t_reshape,
            gst_idx,
            reorder_idx,
            smooth_scale,
            qbits, self.gsize, self.hidden, self.fp8, self.clipping[self.layer_idx],
        )
        
        return pack, scale, zp

    def dequant_pytorch(
        self,
        ttype: Literal["k", "v"],
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zp: torch.Tensor,
    ):
        raise NotImplementedError()

    def dequant_cuda(
        self,
        ttype: Literal["k", "v"],
        pack: torch.Tensor,
        scale: torch.Tensor,
        zp: torch.Tensor,
    ):
        # [bs, seq_len, pack_hidden]
        assert len(pack.shape) == 3

        qbits = self.K_target_bitwidth if ttype == "k" else self.V_target_bitwidth

        smooth_scale = None
        reorder_idx = None
        gst_idx = None
        if self.smooth_scale is not None:
            smooth_scale = self.smooth_scale[ttype].to(scale.dtype).to(scale.device)
        if self.reorder_idx is not None:
            reorder_idx = self.reorder_idx[ttype].to(scale.device)
            gst_idx = self.group_st_idx[ttype].to(scale.device)
        
        dequant = skvq_quant.skvq_dequant_unpack(
            pack, scale, zp,
            gst_idx,
            reorder_idx,
            smooth_scale,
            qbits, self.gsize, self.hidden, self.fp8,
        )

        return dequant


    def quantization(
        self,
        ttype: Literal["k", "v"],
        tensor: torch.Tensor,
        impl: Literal["py", "cuda", "triton"] = "cuda",
    ):
        assert ttype in ["k", "v"]
        assert impl in ["py", "cuda"]

        if self.KIVI_mode:
            impl = "py"

        if tensor is None:
            return None, None, None

        if impl == "py":
            return self.quant_pytorch(ttype, tensor)
        elif impl == "cuda":
            return self.quant_cuda(ttype, tensor)
        elif impl == "triton":
            raise ValueError(f"{impl} not supported")
        else:
            raise ValueError(f"{impl} not supported")

    def forward(self, K, V):
        quantized_K = self.quantization("k", K)
        quantized_V = self.quantization("v", V)
        return quantized_K, quantized_V
    
    def dequant(
        self,
        ttype: Literal["k", "v"],
        quant_data: torch.Tensor,
        scale: torch.Tensor,
        zp: torch.Tensor,
        impl: Literal["py", "cuda", "triton"] = "cuda",
    ):
        assert ttype in ["k", "v"]
        assert impl in ["py", "cuda", "triton"]

        if impl == "py":
            return self.dequant_pytorch(ttype, quant_data, scale, zp)
        elif impl == "cuda":
            return self.dequant_cuda(ttype, quant_data, scale, zp)
        elif impl == "triton":
            raise ValueError(f"{impl} not supported")
        else:
            raise ValueError(f"{impl} not supported")
        
