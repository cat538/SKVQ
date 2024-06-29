# TODO: Pack Unpack Memory allocation ...
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from KV_process import SKVQuantProcessor


class SlidingKVCacheManager(nn.Module):
    def __init__(
        self,
        pre_rope: bool,
        window_size=0,
        attention_sink=0,
        processor_config: dict = None,
        full_prefill=True,
        KIVI_mode=False,
        fake_quant=True,
        use_acc_score:float=0,
        use_random:float=0,
    ) -> None:
        """
        if `pre_rope` is True, we apply rope each step
        """
        super().__init__()
        self.window_size = window_size

        # for heavy hitter
        self.acc_scores = None  # [n_heads, k_len]
        # for attention sink
        self.attention_sink = attention_sink

        self.attention_mask = None
        self.KV_processor = SKVQuantProcessor(**processor_config)
        self.pre_rope = pre_rope
        self.full_prefill = full_prefill
        self.active = True
        self.fake_quant = fake_quant
        self.KIVI_mode = KIVI_mode
        self.use_acc_score = use_acc_score
        self.use_random = use_random
        # if use_acc_score:
        #     self.acc_scores = None

    def clear(self):
        self.acc_scores = None

    def get_ctx_len(self, k_past):
        if k_past is None:
            return 0, 0, 0
        k_sink, k_quant, k_window, _, _ = k_past

        k_sink_len = k_sink.shape[-2] if k_sink is not None else 0
        k_quant_len = k_quant.shape[-2] if k_quant is not None else 0
        k_window_len = k_window.shape[-2] if k_window is not None else 0

        return k_sink_len, k_quant_len, k_window_len

    def forward(self, k_past, v_past, k_fresh: torch.Tensor, v_fresh: torch.Tensor, layer_idx: int):
        """
        k_past shape : [bs, n_heads, ctx_len, head_hidden]
        v_past shape : [bs, n_heads, ctx_len, head_hidden]

        in prefill phase, `ctx_len == 0`, `q_len == prompt_len`

        k_fresh shape: [bs, n_heads, q_len, head_hidden]
        v_fresh shape: [bs, n_heads, q_len, head_hidden]

        return: [k_quant, v_quant, k_full, v_full]
        """

        self.KV_processor.layer_idx = layer_idx

        is_prefill = k_past is None

        if self.KIVI_mode:
            residual_length = self.window_size

            k_sink, v_sink = None, None
            kscale, vscale = None, None
            kzp, vzp = None, None

            if is_prefill:
                ctx_len = k_fresh.shape[-2]

                # 1. quant K
                if ctx_len % residual_length != 0:
                    if ctx_len < residual_length:
                        k_quant = None
                        k_window = k_fresh
                    else:
                        k_quant = k_fresh[
                            :, :, : -(ctx_len % residual_length), :
                        ].contiguous()
                        k_window = k_fresh[
                            :, :, -(ctx_len % residual_length) :, :
                        ].contiguous()
                else:
                    k_quant = k_fresh
                    k_window = None

                # 2. quant V
                if ctx_len <= residual_length:
                    v_quant = None
                    v_window = v_fresh
                else:
                    v_quant = v_fresh[:, :, :-residual_length, :].contiguous()
                    v_window = v_fresh[:, :, -residual_length:, :].contiguous()

                (k_quant, _, _), (v_quant, _, _) = self.KV_processor(
                    k_quant,
                    v_quant,
                )

            else:
                (k_sink, k_quant, k_window, kscale, kzp) = k_past
                (v_sink, v_quant, v_window, vscale, vzp) = v_past

                if k_window is not None:
                    k_window = torch.cat([k_window, k_fresh], dim=2)
                else:
                    k_window = k_fresh
                # 1. quant K
                if k_window.shape[-2] == residual_length:
                    (k_quant_new, _, _), (_, _, _) = self.KV_processor(
                        k_window.contiguous(),
                        None,
                    )
                    if k_quant is not None:
                        k_quant = torch.cat([k_quant, k_quant_new], dim=2)
                    else:
                        k_quant = k_quant_new
                    k_window = None

                v_window = torch.cat([v_window, v_fresh], dim=2)
                value_full_length = v_window.shape[-2]
                # 2. quant V
                if value_full_length > residual_length:
                    assert value_full_length == residual_length + 1

                    v_to_be_processed = v_window[:, :, :1, :].contiguous()
                    (_, _, _), (v_quant_new, _, _) = self.KV_processor(
                        None,
                        v_window[:, :, :1, :].contiguous(),
                    )
                    if v_quant is not None:
                        v_quant = torch.cat([v_quant, v_quant_new], dim=2)
                    else:
                        v_quant = v_quant_new
                    v_window = v_window[:, :, 1:, :].contiguous()

            return (k_sink, k_quant, k_window, kscale, kzp), (
                v_sink,
                v_quant,
                v_window,
                vscale,
                vzp,
            )

        if is_prefill:
            ctx_len = k_fresh.shape[-2]
            n_quantize = ctx_len - (self.attention_sink + self.window_size)

            if n_quantize > 0:
                quant_range = (self.attention_sink, self.attention_sink + n_quantize)

                (k_quant, kscale, kzp), (v_quant, vscale, vzp) = self.KV_processor(
                    k_fresh[:, :, quant_range[0] : quant_range[1], :],
                    v_fresh[:, :, quant_range[0] : quant_range[1], :],
                )
                k_sink = (
                    k_fresh[:, :, : self.attention_sink, :]
                    if self.attention_sink
                    else None
                )
                k_window = (
                    k_fresh[:, :, -self.window_size :, :] if self.window_size else None
                )
                v_sink = (
                    v_fresh[:, :, : self.attention_sink, :]
                    if self.attention_sink
                    else None
                )
                v_window = (
                    v_fresh[:, :, -self.window_size :, :] if self.window_size else None
                )

            else:
                (k_quant, kscale, kzp), (v_quant, vscale, vzp) = (None, None, None), (
                    None,
                    None,
                    None,
                )
                if ctx_len <= self.attention_sink:
                    k_window, v_window = None, None
                else:
                    k_window = k_fresh[:, :, self.attention_sink :, :]
                    v_window = v_fresh[:, :, self.attention_sink :, :]

                k_sink = (
                    k_fresh[:, :, : self.attention_sink, :]
                    if self.attention_sink
                    else None
                )
                v_sink = (
                    v_fresh[:, :, : self.attention_sink, :]
                    if self.attention_sink
                    else None
                )

            return (k_sink, k_quant, k_window, kscale, kzp), (
                v_sink,
                v_quant,
                v_window,
                vscale,
                vzp,
            )
        else:
            (k_sink, k_quant, k_window, kscale, kzp) = k_past
            (v_sink, v_quant, v_window, vscale, vzp) = v_past
            k_sink_len, k_quant_len, k_window_len = self.get_ctx_len(k_past)

            if k_sink_len > 0 and k_sink_len < self.attention_sink:
                k_sink = torch.cat((k_sink, k_fresh), dim=-2)
                v_sink = torch.cat((v_sink, v_fresh), dim=-2)
                return (k_sink, k_quant, k_window, kscale, kzp), (
                    v_sink,
                    v_quant,
                    v_window,
                    vscale,
                    vzp,
                )
            if k_window_len < self.window_size:
                k_window = (
                    k_fresh
                    if k_window is None
                    else torch.cat((k_window, k_fresh), dim=-2)
                )
                v_window = (
                    v_fresh
                    if v_window is None
                    else torch.cat((v_window, v_fresh), dim=-2)
                )
                return (k_sink, k_quant, k_window, kscale, kzp), (
                    v_sink,
                    v_quant,
                    v_window,
                    vscale,
                    vzp,
                )

            k_to_be_processed = k_window[:, :, :1, :] if self.window_size else k_fresh
            v_to_be_processed = v_window[:, :, :1, :] if self.window_size else v_fresh
            (k_quant_new, kscale_new, kzp_new), (v_quant_new, vscale_new, vzp_new) = (
                self.KV_processor(
                    k_to_be_processed,
                    v_to_be_processed,
                )
            )
            k_quant = (
                torch.cat((k_quant, k_quant_new), dim=-2)
                if k_quant_len
                else k_quant_new
            )
            v_quant = (
                torch.cat((v_quant, v_quant_new), dim=-2)
                if k_quant_len
                else v_quant_new
            )

            # [bs, seq_len, num_groups]
            if kscale_new is not None:
                kscale = (
                    torch.cat((kscale, kscale_new), dim=-2)
                    if k_quant_len
                    else kscale_new
                )
                vscale = (
                    torch.cat((vscale, vscale_new), dim=-2)
                    if k_quant_len
                    else vscale_new
                )
                kzp = torch.cat((kzp, kzp_new), dim=-2) if k_quant_len else kzp_new
                vzp = torch.cat((vzp, vzp_new), dim=-2) if k_quant_len else vzp_new

            k_window = (
                torch.cat((k_window[:, :, 1:, :], k_fresh), dim=-2)
                if self.window_size
                else None
            )
            v_window = (
                torch.cat((v_window[:, :, 1:, :], v_fresh), dim=-2)
                if self.window_size
                else None
            )

            return (k_sink, k_quant, k_window, kscale, kzp), (
                v_sink,
                v_quant,
                v_window,
                vscale,
                vzp,
            )

    def dequant(
        self,
        k_quant: torch.Tensor,
        kscale: torch.Tensor,
        kzp: torch.Tensor,
        v_quant: torch.Tensor,
        vscale: torch.Tensor,
        vzp: torch.Tensor,
    ):
        k_dequant = self.KV_processor.dequant("k", k_quant, kscale, kzp)
        v_dequant = self.KV_processor.dequant("v", v_quant, vscale, vzp)
        return k_dequant, v_dequant


class ModelKVCacheManager:
    def __init__(
        self, smooth_file, reorder_file, kv_managers: list[SlidingKVCacheManager]
    ) -> None:
        self.kv_managers = kv_managers
        self.smooth_file = smooth_file
        self.reorder_file = reorder_file

    @classmethod
    def create(
        cls,
        model: PreTrainedModel,
        kbits,
        vbits,
        gsize: int,
        reorder_file: str = None,
        smooth_file: str = None,
        window_size: int = 32,
        pre_rope: bool = False,
        clipping: float = 1.0,
        attn_sink: int = 0,
        full_prefill: bool = True,
        KIVI_mode: bool = False,
        fp8: bool = False,
        fake_quant: bool = True,
        use_acc_score:float = 0,
        use_random:float=0,
    ):
        if KIVI_mode:
            assert not pre_rope
            assert window_size >= gsize and window_size % gsize == 0
            assert smooth_file == None
            assert reorder_file == None
            assert full_prefill
            assert fake_quant

        kv_managers = []
        layers = model.model.layers
        cfg = model.config
        hidden = cfg.num_key_value_heads * cfg.hidden_size // cfg.num_attention_heads

        if reorder_file is not None:
            rod_meta = torch.load(reorder_file)
            print(f"* Load reorder cache from {reorder_file} ...")
        if smooth_file is not None:
            smooth_scale = torch.load(smooth_file)
            print(f"* Load smooth cache from {smooth_file} ...")

        for i, _ in enumerate(layers):
            if reorder_file is not None:
                rod_idx = rod_meta["reorder_indices"]
                group_st = rod_meta["cluster_st_inds"]
                layer_rod_meta = (
                    {"k": rod_idx[i][0], "v": rod_idx[i][1]},
                    {"k": group_st[i][0], "v": group_st[i][1]},
                )
            else:
                layer_rod_meta = None

            if smooth_file is not None:
                layer_smooth_scale = {
                    "k": smooth_scale["k"][i],
                    "v": smooth_scale["v"][i],
                }
            else:
                layer_smooth_scale = None

            processor_cfg = {
                "K_target_bitwidth": kbits,
                "V_target_bitwidth": vbits,
                "gsize": gsize,
                "reorder_meta": layer_rod_meta,
                "smooth_scale": layer_smooth_scale,
                "clipping": clipping,
                "KIVI_mode": KIVI_mode,
                "fp8": fp8,
                "hidden": hidden,
                "fake_quant": fake_quant,
            }
            kv_managers.append(
                SlidingKVCacheManager(
                    pre_rope=pre_rope,
                    window_size=window_size,
                    processor_config=processor_cfg,
                    full_prefill=full_prefill,
                    KIVI_mode=KIVI_mode,
                    attention_sink=attn_sink,
                    fake_quant=fake_quant,
                    use_acc_score=use_acc_score,
                    use_random=use_random,
                )
            )

        return cls(smooth_file, reorder_file, kv_managers)

    def full_prefill(self, is_full_prefill: bool):
        for m in self.kv_managers:
            m.full_prefill = is_full_prefill

    def active(self, is_active: bool):
        for m in self.kv_managers:
            m.active = is_active

    def clear(self):
        for m in self.kv_managers:
            m.clear()

    def __str__(self) -> str:
        manager = self.kv_managers[0]
        return (
            f">>> `{len(self.kv_managers)}` kv managers\n"
            f">>> group_size:   {manager.KV_processor.gsize}\n"
            f">>> window_size:  {manager.window_size}\n"
            f">>> pre_rope:     {manager.pre_rope}\n"
            f">>> full_prefill: {manager.full_prefill}\n"
            f">>> smooth:       {self.smooth_file}\n"
            f">>> reorder:      {self.reorder_file}\n"
            f">>> clipping:     {manager.KV_processor.clipping}\n"
            f">>> sink:         {manager.attention_sink}\n"
            f">>> KIVI_mode:    {manager.KIVI_mode}\n"
            f">>> FP8:          {manager.KV_processor.fp8}\n"
            f">>> fake_quant:   {manager.KV_processor.fake_quant}\n"
            f">>> heavy_hitter: {manager.use_acc_score}\n"
            f">>> random:       {manager.use_random}\n"
        )

    def tag(self) -> str:
        manager = self.kv_managers[0]
        processor = self.kv_managers[0].KV_processor

        kbits = processor.K_target_bitwidth
        vbits = processor.V_target_bitwidth
        gsize = processor.gsize
        window = manager.window_size

        pre_rope = "-pre_rope" if manager.pre_rope else ""
        reorder = "" if self.reorder_file is None else "-rod"
        smooth = "" if self.smooth_file is None else "-smooth"
        clipping = f"-clip{processor.clipping[0]}"
        KIVI_mode = "-KIVI" if manager.KIVI_mode else ""
        fp8 = "-fp8" if processor.fp8 else ""
        sink = f"-sink{manager.attention_sink}" if manager.attention_sink else ""
        h2o = f"-h2o{manager.use_acc_score}" if manager.use_acc_score != 0 else ""
        rand = f"-random{manager.use_random}" if manager.use_random != 0 else ""

        return f"k{kbits}-v{vbits}-g{gsize}-w{window}{reorder}{smooth}{clipping}{pre_rope}{KIVI_mode}{sink}{fp8}{h2o}{rand}"
