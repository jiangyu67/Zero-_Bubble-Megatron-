# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Quantization metrics for monitoring quantization error.

This module provides SignRate and CosSim metrics to monitor quantization error
during QAT (Quantization-Aware Training). These metrics are used to drive
Dynamic Low-Rank Compensation (DLRC).

SignRate (符号一致率): Measures the ratio of weight signs that remain unchanged
after quantization. Sign flip is the most critical quantization error as it
directly causes gradient direction reversal.

    SignRate = (1/N) * sum(sign(W) == sign(W_hat))

CosSim (余弦相似度): Measures the directional deviation between original and
quantized weight vectors in the vector space.

    CosSim(W, W_hat) = (W . W_hat) / (||W|| * ||W_hat||)
"""

import torch
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class QuantizationMetrics:
    """Container for quantization metrics of a single layer."""
    
    layer_name: str
    sign_rate: float = 1.0
    cos_sim: float = 1.0
    needs_lrc: bool = False  # Low-rank compensation flag
    
    # Historical metrics for trend analysis
    sign_rate_history: list = field(default_factory=list)
    cos_sim_history: list = field(default_factory=list)


class QuantizationMetricsCollector:
    """Collects and manages quantization metrics across layers.
    
    This class provides:
    - Async computation of SignRate and CosSim
    - TensorBuffer for storing metrics without blocking main computation
    - DLRC trigger logic based on thresholds
    """
    
    # DLRC trigger thresholds
    SIGN_RATE_THRESHOLD = 0.95
    COS_SIM_THRESHOLD = 0.98
    
    def __init__(self, device: torch.device = torch.device('cuda')):
        self.device = device
        self._metrics: Dict[str, QuantizationMetrics] = {}
        self._tensor_buffer: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Use dedicated CUDA stream for async computation
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
    def compute_metrics_async(
        self,
        original_weight: torch.Tensor,
        quantized_weight: torch.Tensor,
        layer_name: str,
    ) -> Tuple[float, float]:
        """Compute SignRate and CosSim asynchronously.

        Returns the computed (sign_rate, cos_sim). The collector history is updated
        later via `sync_and_update()` to avoid blocking the main stream.
        """
        if not self._can_compute_metrics(original_weight, quantized_weight):
            print(
                f"[QuantMetrics] Layer: {layer_name}, skip metrics because the candidate quantized weight does not appear to be an FP8 tensor."
            )
            return 1.0, 1.0

        if self._stream is None:
            return self._compute_metrics_sync(original_weight, quantized_weight, layer_name)
            
        with torch.cuda.stream(self._stream):
            sign_rate, cos_sim = self._compute_metrics_impl(
                original_weight, quantized_weight, layer_name
            )
            
        # Store in buffer
        self._tensor_buffer[layer_name] = (
            torch.tensor([sign_rate], device=self.device),
            torch.tensor([cos_sim], device=self.device)
        )
        return sign_rate, cos_sim
        
    def _compute_metrics_sync(
        self,
        original_weight: torch.Tensor,
        quantized_weight: torch.Tensor,
        layer_name: str,
    ) -> Tuple[float, float]:
        """Synchronous computation fallback."""
        if not self._can_compute_metrics(original_weight, quantized_weight):
            print(
                f"[QuantMetrics] Layer: {layer_name}, skip metrics because the candidate quantized weight does not appear to be an FP8 tensor."
            )
            return 1.0, 1.0

        sign_rate, cos_sim = self._compute_metrics_impl(
            original_weight, quantized_weight, layer_name
        )
        self._update_metrics(layer_name, sign_rate, cos_sim)
        return sign_rate, cos_sim
        
    def _can_compute_metrics(
        self,
        original_weight: torch.Tensor,
        quantized_weight: torch.Tensor,
    ) -> bool:
        """Check whether the candidate tensor is a plausible quantized weight tensor."""
        if quantized_weight is None:
            return False
        if original_weight.numel() != quantized_weight.numel():
            return False
        if original_weight.device != quantized_weight.device:
            return False
        return True

    def _compute_metrics_impl(
        self,
        original_weight: torch.Tensor,
        quantized_weight: torch.Tensor,
        layer_name: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Internal implementation of metrics computation."""
        # Flatten tensors for computation
        orig = original_weight.flatten().float()
        if hasattr(quantized_weight, "dequantize"):
            quant = quantized_weight.dequantize().flatten().float()
        else:
            quant = quantized_weight.flatten().float()
        
        # Compute SignRate
        orig_sign = torch.sign(orig)
        quant_sign = torch.sign(quant)
        # Handle zeros: sign(0) = 0, but we treat them as matching
        orig_sign = torch.where(orig_sign == 0, torch.ones_like(orig_sign), orig_sign)
        quant_sign = torch.where(quant_sign == 0, torch.ones_like(quant_sign), quant_sign)
        sign_match = (orig_sign == quant_sign).float()
        sign_rate = sign_match.mean().item()
        
        # Compute CosSim
        dot_product = torch.dot(orig, quant)
        norm_orig = torch.norm(orig)
        norm_quant = torch.norm(quant)
        
        # Avoid division by zero
        if norm_orig.item() > 0 and norm_quant.item() > 0:
            cos_sim = (dot_product / (norm_orig * norm_quant)).item()
        else:
            cos_sim = 1.0
        if layer_name is not None:
            print(
                f"[QuantMetrics] Layer: {layer_name}, SignRate: {sign_rate:.4f}, "
                f"CosSim: {cos_sim:.4f}"
            )
            
        return sign_rate, cos_sim
    
    def _update_metrics(self, layer_name: str, sign_rate: float, cos_sim: float):
        """Update metrics for a layer and check DLRC trigger."""
        if layer_name not in self._metrics:
            self._metrics[layer_name] = QuantizationMetrics(layer_name=layer_name)
            
        metrics = self._metrics[layer_name]
        metrics.sign_rate = sign_rate
        metrics.cos_sim = cos_sim
        
        # Update history
        metrics.sign_rate_history.append(sign_rate)
        metrics.cos_sim_history.append(cos_sim)
        
        # Keep history bounded
        max_history = 100
        if len(metrics.sign_rate_history) > max_history:
            metrics.sign_rate_history = metrics.sign_rate_history[-max_history:]
        if len(metrics.cos_sim_history) > max_history:
            metrics.cos_sim_history = metrics.cos_sim_history[-max_history:]
            
        # DLRC trigger logic
        metrics.needs_lrc = (
            sign_rate < self.SIGN_RATE_THRESHOLD or 
            cos_sim < self.COS_SIM_THRESHOLD
        )
        
    def sync_and_update(self) -> None:
        """Synchronize CUDA stream and update metrics."""
        if self._stream is not None:
            torch.cuda.synchronize()
            for layer_name, (sign_tensor, cos_tensor) in self._tensor_buffer.items():
                self._update_metrics(
                    layer_name, 
                    sign_tensor.item(), 
                    cos_tensor.item()
                )
            # Prevent re-applying the same buffered values on subsequent syncs.
            self._tensor_buffer.clear()
                
    def get_metrics(self, layer_name: str) -> Optional[QuantizationMetrics]:
        """Get metrics for a specific layer."""
        return self._metrics.get(layer_name)
    
    def get_layers_needing_lrc(self) -> list:
        """Get list of layers that need low-rank compensation."""
        return [
            name for name, metrics in self._metrics.items() 
            if metrics.needs_lrc
        ]
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._tensor_buffer.clear()


# Global metrics collector instance
_global_collector: Optional[QuantizationMetricsCollector] = None


def get_global_collector() -> QuantizationMetricsCollector:
    """Get or create the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = QuantizationMetricsCollector()
    return _global_collector


def compute_quantization_metrics(
    original_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    layer_name: str,
    async_compute: bool = True,
) -> Tuple[float, float]:
    """Convenience function to compute quantization metrics."""
    collector = get_global_collector()
    
    if not collector._can_compute_metrics(original_weight, quantized_weight):
        print(
            f"[QuantMetrics] Layer: {layer_name}, skip metrics because the candidate quantized weight does not appear to be an FP8 tensor."
        )
        return 1.0, 1.0

    if async_compute:
        # Async path computes once and buffers results; history is updated in sync_and_update().
        sign_rate, cos_sim = collector.compute_metrics_async(original_weight, quantized_weight, layer_name)
    else:
        # Sync path computes once and updates history immediately.
        sign_rate, cos_sim = collector._compute_metrics_impl(original_weight, quantized_weight, layer_name)
        collector._update_metrics(layer_name, sign_rate, cos_sim)
        
    return sign_rate, cos_sim