# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    get_cuda_rng_tracker,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

# TODO: Implement GatedDeltaNetContextParallel
# from .gated_delta_net_context_parallel import GatedDeltaNetContextParallel

try:
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    raise ImportError(
        "FLA is required for GatedDeltaNet. "
        "Please install it: pip3 install flash-linear-attention==0.4.0"
    )

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    raise ImportError(
        "causal_conv1d is required for GatedDeltaNet. "
        "Please install it: pip3 install causal-conv1d==1.5.3.post1"
    )


logger = logging.getLogger(__name__)


@dataclass
class GatedDeltaNetSubmodules:
    """
    Contains the module specs for the input linear, output norm, and output linear layers.
    """

    qkvz_proj: Union[ModuleSpec, type] = IdentityOp
    ba_proj: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class GatedDeltaNet(MegatronModule):
    """Gated Delta Net (GDN) layer class

    GDN layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNetSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this GDN layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization in the kernel of the gated delta rule.
            A_init_range: The initialization range for the attention weights.
            pg_collection: The required process groups to use for tensor model parallel and context
                parallel.
        """

        if not HAVE_FLA:
            raise ImportError("FLA is not installed. Please install it with `pip install fla`.")

        super().__init__(config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
        self.pg_collection = pg_collection
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads

        # TP compatibility checks
        assert self.num_key_heads % self.tp_size == 0, (
            f"num_key_heads ({self.num_key_heads}) must be divisible by tp_size ({self.tp_size})"
        )
        assert self.num_value_heads % self.num_key_heads == 0, (
            f"num_value_heads ({self.num_value_heads}) must be divisible by "
            f"num_key_heads ({self.num_key_heads}) for GQA grouping"
        )

        # Local dimensions after TP split
        self.num_key_heads_local = self.num_key_heads // self.tp_size
        self.num_value_heads_local = self.num_value_heads // self.tp_size

        # Number of value heads per key head group (for GQA-style grouping)
        self.v_heads_per_kv_group = self.num_value_heads // self.num_key_heads

        # Size of one interleaved group in qkvz projection:
        # [q_head_dim, k_head_dim, v_head_dim * v_heads_per_group, z_head_dim * v_heads_per_group]
        self.qkvz_group_dim = (
            self.key_head_dim
            + self.key_head_dim
            + self.value_head_dim * self.v_heads_per_kv_group
            + self.value_head_dim * self.v_heads_per_kv_group
        )

        # Size of one interleaved group in ba projection:
        # [beta_heads_per_group, alpha_heads_per_group]
        self.ba_group_dim = self.v_heads_per_kv_group * 2

        # Input projection (hidden_states -> q, k, v, gate, beta, alpha)
        # TODO: for now, output gate is forced for GDN.
        # We may remove this restriction in the future.
        self.qkvz_proj_dim = self.qkvz_group_dim * self.num_key_heads
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.qkvz_proj_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                "QKVZ projection output tensor must be a multiple of 16."
            )
        self.qkvz_proj = build_module(
            submodules.qkvz_proj,
            self.hidden_size,
            self.qkvz_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkvz_fc",
            tp_group=self.pg_collection.tp,
        )

        self.ba_proj_dim = self.ba_group_dim * self.num_key_heads
        self.ba_proj = build_module(
            submodules.ba_proj,
            self.hidden_size,
            self.ba_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="ba_fc",
            tp_group=self.pg_collection.tp,
        )

        # Conv1d for Q, K, V (3 separate conv1d instead of one combined)
        # Each conv1d handles its own component - this naturally works with TP
        # since each rank has local heads
        # causal_conv1d with channel last layout requires dim % 8 == 0

        # Q conv1d - operates on query heads
        # weight shape: [q_dim_local, 1, d_conv]
        # bias shape: [q_dim_local]
        self.q_dim_local = self.key_head_dim * self.num_key_heads_local
        assert self.q_dim_local % 8 == 0, f"q_dim_local must be divisible by 8, got {self.q_dim_local}"
        self.q_conv1d = nn.Conv1d(
            in_channels=self.q_dim_local,
            out_channels=self.q_dim_local,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.q_dim_local,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.q_conv1d.weight, "tensor_model_parallel", True)
        if conv_bias:
            setattr(self.q_conv1d.bias, "tensor_model_parallel", True)

        # K conv1d - operates on key heads
        self.k_dim_local = self.key_head_dim * self.num_key_heads_local
        self.k_conv1d = nn.Conv1d(
            in_channels=self.k_dim_local,
            out_channels=self.k_dim_local,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.k_dim_local,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.k_conv1d.weight, "tensor_model_parallel", True)
        if conv_bias:
            setattr(self.k_conv1d.bias, "tensor_model_parallel", True)

        # V conv1d - operates on value heads
        self.v_dim_local = self.value_head_dim * self.num_value_heads_local
        assert self.v_dim_local % 8 == 0, f"v_dim_local must be divisible by 8, got {self.v_dim_local}"
        self.v_conv1d = nn.Conv1d(
            in_channels=self.v_dim_local,
            out_channels=self.v_dim_local,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.v_dim_local,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.v_conv1d.weight, "tensor_model_parallel", True)
        if conv_bias:
            setattr(self.v_conv1d.bias, "tensor_model_parallel", True)

        # Time step projection (discretization)
        # dt_bias parameter
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_value_heads_local,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        # A_log parameter
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_value_heads_local,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)

        # Output layernorm before projection
        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.value_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )

        # TODO: support CP

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                # conv1d.weight
                if self.conv_init is not None:
                    nn.init.uniform_(self.q_conv1d.weight, -self.conv_init, self.conv_init)
                    nn.init.uniform_(self.k_conv1d.weight, -self.conv_init, self.conv_init)
                    nn.init.uniform_(self.v_conv1d.weight, -self.conv_init, self.conv_init)
                # dt_bias
                torch.ones(
                    self.num_value_heads_local,
                    out=self.dt_bias.data,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                )
                # A_log
                A = torch.empty(
                    self.num_value_heads_local,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(A)

    def _deinterleave_qkvz(
        self, qkvz: Tensor, batch: int, seq_len: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Deinterleave QKVZ from grouped format to separate Q, K, V, Z tensors.

        Input layout (after TP split): [b, s, num_key_heads_local * qkvz_group_dim]
        Each group contains: [q_head, k_head, v_heads..., z_heads...]

        Returns:
            query: [b, s, num_key_heads_local, key_head_dim]
            key: [b, s, num_key_heads_local, key_head_dim]
            value: [b, s, num_value_heads_local, value_head_dim]
            gate: [b, s, num_value_heads_local, value_head_dim]
        """
        # Reshape to expose groups: [b, s, num_key_heads_local, qkvz_group_dim]
        qkvz = qkvz.reshape(batch, seq_len, self.num_key_heads_local, self.qkvz_group_dim)

        # Split each group into q, k, v, z components
        q, k, v, z = torch.split(
            qkvz,
            [
                self.key_head_dim,
                self.key_head_dim,
                self.value_head_dim * self.v_heads_per_kv_group,
                self.value_head_dim * self.v_heads_per_kv_group,
            ],
            dim=-1,
        )

        # v, z need reshape: [b, s, num_key_heads_local, v_heads_per_group * v_head_dim]
        #                 -> [b, s, num_value_heads_local, value_head_dim]
        v = v.reshape(batch, seq_len, self.num_value_heads_local, self.value_head_dim)
        z = z.reshape(batch, seq_len, self.num_value_heads_local, self.value_head_dim)

        return q, k, v, z

    def _deinterleave_ba(self, ba: Tensor, batch: int, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Deinterleave BA from grouped format to separate beta, alpha tensors.

        Input layout (after TP split): [b, s, num_key_heads_local * ba_group_dim]
        Each group contains: [beta_heads..., alpha_heads...]

        Returns:
            beta: [b, s, num_value_heads_local]
            alpha: [b, s, num_value_heads_local]
        """
        # Reshape to expose groups: [b, s, num_key_heads_local, ba_group_dim]
        ba = ba.reshape(batch, seq_len, self.num_key_heads_local, self.ba_group_dim)

        # Split each group
        beta, alpha = torch.split(
            ba,
            [self.v_heads_per_kv_group, self.v_heads_per_kv_group],
            dim=-1,
        )

        # Reshape to [b, s, num_value_heads_local]
        beta = beta.reshape(batch, seq_len, self.num_value_heads_local)
        alpha = alpha.reshape(batch, seq_len, self.num_value_heads_local)

        return beta, alpha

    def _apply_conv1d(self, x: Tensor, conv: nn.Conv1d, seq_idx: Optional[Tensor] = None) -> Tensor:
        """Apply causal conv1d to input tensor.

        Args:
            x: Input tensor [b, s, d]
            conv: Conv1d module
            seq_idx: Sequence index for packed sequences

        Returns:
            Output tensor [b, s, d]
        """
        x = x.contiguous().transpose(1, 2)  # [b, s, d] -> [b, d, s]
        x = causal_conv1d_fn(
            x=x,
            weight=conv.weight.squeeze(1),  # d, 1, w -> d, w
            bias=conv.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
        x = x.transpose(1, 2)  # [b, d, s] -> [b, s, d]
        return x

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        Perform a forward pass through the GDN module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) GDN output and bias.

        """
        # TODO: Deal with attention_mask

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")

        # Handle packed sequences
        cu_seqlens = None
        seq_idx = None
        if packed_seq_params is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q
            # Build seq_idx for causal_conv1d: (batch, seqlen)
            total_tokens = seq_len
            num_seqs = cu_seqlens.shape[0] - 1
            seq_idx = torch.zeros((batch, total_tokens), dtype=torch.int32, device=hidden_states.device)
            for i in range(num_seqs):
                seq_idx[0, cu_seqlens[i]:cu_seqlens[i+1]] = i

        # Input projections
        nvtx_range_push(suffix="qkvz_proj")
        qkvz, _ = self.qkvz_proj(hidden_states)
        nvtx_range_pop(suffix="qkvz_proj")

        nvtx_range_push(suffix="ba_proj")
        ba, _ = self.ba_proj(hidden_states)
        nvtx_range_pop(suffix="ba_proj")

        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvz = qkvz.transpose(0, 1)
        ba = ba.transpose(0, 1)

        # Deinterleave into separate components
        query, key, value, gate = self._deinterleave_qkvz(qkvz, batch, seq_len)
        beta, alpha = self._deinterleave_ba(ba, batch, seq_len)

        # Flatten for conv1d: [b, s, num_heads, head_dim] -> [b, s, dim]
        query = query.reshape(batch, seq_len, -1)
        key = key.reshape(batch, seq_len, -1)
        value = value.reshape(batch, seq_len, -1)

        # Convolution on q, k, v
        nvtx_range_push(suffix="conv1d")
        # TODO: support deterministic_mode for causal_conv1d
        assert self.activation in ["silu", "swish"]
        query = self._apply_conv1d(query, self.q_conv1d, seq_idx)
        key = self._apply_conv1d(key, self.k_conv1d, seq_idx)
        value = self._apply_conv1d(value, self.v_conv1d, seq_idx)
        nvtx_range_pop(suffix="conv1d")

        # Reshape back to head format
        query = query.reshape(batch, seq_len, self.num_key_heads_local, self.key_head_dim)
        key = key.reshape(batch, seq_len, self.num_key_heads_local, self.key_head_dim)
        value = value.reshape(batch, seq_len, self.num_value_heads_local, self.value_head_dim)

        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())
        if self.v_heads_per_kv_group > 1:
            query = query.repeat_interleave(self.v_heads_per_kv_group, dim=2)
            key = key.repeat_interleave(self.v_heads_per_kv_group, dim=2)

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        # Calculate g and beta
        nvtx_range_push(suffix="g_and_beta")
        g = -self.A_log.exp() * F.softplus(alpha.float() + self.dt_bias)  # In fp32
        beta = beta.sigmoid()
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        # TODO: support deterministic_mode for chunk_gated_delta_rule
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        # RMSNorm
        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    @jit_fuser
    def _apply_gated_norm(self, x, gate):
        # Output Norm
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        # Output gate
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * self.act_fn(gate.float())
        y = y.to(x_dtype)
        return y

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                "A_log": 0,
                "dt_bias": 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Submodules
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            if name in ["q_conv1d", "k_conv1d", "v_conv1d"]:
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {"weight": 0}
                if self.conv_bias:
                    tp_sharding_map["bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )

            sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict


def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim], (
            split_start,
            orig_sh_ten_no_data.local_shape[split_dim],
        )
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens,
            t.shape,
        )
        return chunk_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        orig_sh_ten.key, orig_sh_ten.data, sh_ten_build_fn, sh_ten_merge_fn, orig_sh_ten.replica_id
    )


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    # pylint: disable=line-too-long
    '''
    Torch-native implementation of chunked gated delta rule for deterministic mode.
    Need this because FLA is not deterministic.

    Reference: https://github.com/huggingface/transformers/blob/144c8ce2809a2e21914017652700e1ecb450501e/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L470-L547
    '''

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
