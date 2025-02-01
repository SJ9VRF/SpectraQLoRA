import torch
import torch.nn as nn

class SNRAnalyzer:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def get_snr(self, layer):
        """Compute Signal-to-Noise Ratio (SNR) for a given layer."""
        if hasattr(layer, 'weight') and layer.weight is not None:
            weights = layer.weight.data
            signal = torch.mean(weights).item()
            noise = torch.std(weights).item()
            return signal / noise if noise != 0 else 0
        return 0

    def get_high_snr_layers(self):
        """Identify layers with high SNR above threshold."""
        high_snr_layers = []
        snr_values = {}
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LayerNorm, nn.Embedding)):
                snr = self.get_snr(layer)
                snr_values[name] = snr
                if snr > self.threshold:
                    high_snr_layers.append(name)
        
        # Compute layer similarity for Spectral Gradient Merging (SGM)
        merged_layers = self.merge_similar_layers(snr_values)
        return merged_layers

    def merge_similar_layers(self, snr_values):
        """Group layers with similar SNR values to share gradient updates."""
        merged_layers = []
        sorted_layers = sorted(snr_values.items(), key=lambda x: x[1], reverse=True)
        grouped = {}
        
        for i, (layer_name, snr) in enumerate(sorted_layers):
            if i == 0:
                grouped[layer_name] = [layer_name]
            else:
                prev_layer, prev_snr = sorted_layers[i - 1]
                if abs(snr - prev_snr) < 0.1:  # Threshold for merging
                    grouped[prev_layer].append(layer_name)
                else:
                    grouped[layer_name] = [layer_name]
        
        for key, group in grouped.items():
            merged_layers.append(group)
        
        return merged_layers

