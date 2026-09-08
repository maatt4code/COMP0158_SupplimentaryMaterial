import torch
import torch.nn as nn
import numpy as np

class DifferentiableDDSPSynth(nn.Module):
    def __init__(self, sample_rate=16000, num_harmonics=32, num_filter_bands=65):
        """
        Differentiable Harmonic + Filtered Noise Synthesizer (DDSP).
        Args:
            sample_rate: Audio sampling rate.
            num_harmonics: Number of sinusoidal harmonics to synthesize.
            num_filter_bands: Number of frequency bands for noise filtering.
        """
        super(DifferentiableDDSPSynth, self).__init__()
        self.sr = sample_rate
        self.num_harmonics = num_harmonics
        self.num_filter_bands = num_filter_bands

    def synthesize_harmonic(self, f0, amplitude, harmonic_distribution):
        """
        Synthesizes the harmonic component.
        Args:
            f0: Fundamental frequency trajectory, shape (Batch_Size, Num_Samples, 1)
            amplitude: Global amplitude envelope, shape (Batch_Size, Num_Samples, 1)
            harmonic_distribution: Distribution over harmonics, shape (Batch_Size, Num_Samples, Num_Harmonics)
        Returns:
            Harmonic signal, shape (Batch_Size, Num_Samples)
        """
        device = f0.device
        batch_size, num_samples, _ = f0.shape
        
        # Create harmonic multipliers: [1, 2, ..., num_harmonics]
        harmonics = torch.arange(1, self.num_harmonics + 1, device=device, dtype=torch.float32)
        harmonics = harmonics.view(1, 1, self.num_harmonics) # Shape: (1, 1, Num_Harmonics)
        
        # Calculate instantaneous frequencies for all harmonics
        inst_freqs = f0 * harmonics # Shape: (Batch_Size, Num_Samples, Num_Harmonics)
        
        # Clip frequencies above Nyquist limit to prevent aliasing
        nyquist = self.sr / 2.0
        inst_freqs = torch.where(inst_freqs < nyquist, inst_freqs, torch.zeros_like(inst_freqs))
        
        # Integrate frequency to get phase: phase = 2 * pi * cumsum(freq) / sr
        phases = 2.0 * np.pi * torch.cumsum(inst_freqs, dim=1) / self.sr # Shape: (Batch_Size, Num_Samples, Num_Harmonics)
        
        # Generate individual harmonic waveforms
        harmonic_waves = torch.sin(phases) # Shape: (Batch_Size, Num_Samples, Num_Harmonics)
        
        # Scale individual harmonics by the distribution and global amplitude
        # harmonic_distribution is normalized (softmaxed) over the last dimension
        harmonic_amplitudes = amplitude * harmonic_distribution # Shape: (Batch_Size, Num_Samples, Num_Harmonics)
        
        # Sum all harmonics to create the composite signal
        harmonic_signal = torch.sum(harmonic_waves * harmonic_amplitudes, dim=2) # Shape: (Batch_Size, Num_Samples)
        
        return harmonic_signal

    def synthesize_noise(self, noise_filter_magnitudes):
        """
        Synthesizes filtered noise by filtering white noise in the frequency domain.
        Args:
            noise_filter_magnitudes: Time-varying filter frequency responses.
                                     Shape: (Batch_Size, Num_Frames, Num_Filter_Bands)
        Returns:
            Filtered noise signal, shape (Batch_Size, Num_Samples)
        """
        device = noise_filter_magnitudes.device
        batch_size, num_frames, num_bands = noise_filter_magnitudes.shape
        
        # Parameters for STFT windowing matching the frames count
        n_fft = (num_bands - 1) * 2
        hop_length = n_fft // 2
        num_samples = num_frames * hop_length
        
        # 1. Generate white noise
        white_noise = torch.randn(batch_size, num_samples, device=device)
        
        # 2. Convert white noise to frequency domain (STFT)
        window = torch.hann_window(n_fft, device=device)
        Zxx = torch.stft(white_noise, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        # Zxx shape: (Batch_Size, Num_Filter_Bands, Num_Frames)
        
        # 3. Apply the time-varying filter magnitude response
        # Match dimensions: transpose magnitudes to (Batch_Size, Num_Filter_Bands, Num_Frames)
        filter_gains = noise_filter_magnitudes.transpose(1, 2)
        
        # Interpolate or match frames if there is a shape mismatch
        if filter_gains.shape[2] != Zxx.shape[2]:
            filter_gains = torch.nn.functional.interpolate(
                filter_gains, size=Zxx.shape[2], mode='linear', align_corners=True
            )
            
        filtered_Zxx = Zxx * filter_gains
        
        # 4. Convert back to time domain (inverse STFT)
        filtered_noise = torch.istft(
            filtered_Zxx, n_fft=n_fft, hop_length=hop_length, window=window, length=num_samples
        )
        
        return filtered_noise

    def forward(self, f0, amplitude, harmonic_distribution, noise_filter_magnitudes):
        """
        Combines harmonic and noise components to generate the full audio.
        """
        harmonic_part = self.synthesize_harmonic(f0, amplitude, harmonic_distribution)
        noise_part = self.synthesize_noise(noise_filter_magnitudes)
        
        # Ensure they have the same length (noise_part might be slightly longer/shorter due to STFT hops)
        min_len = min(harmonic_part.shape[1], noise_part.shape[1])
        harmonic_part = harmonic_part[:, :min_len]
        noise_part = noise_part[:, :min_len]
        
        # Scale noise part by the amplitude envelope to prevent it from overwhelming the harmonics
        # A 0.20 scaling factor keeps the noise-to-harmonic ratio balanced (approx -14dB)
        amp_slice = amplitude[:, :min_len, 0]
        noise_part = noise_part * amp_slice * 0.20
        
        mixed_signal = harmonic_part + noise_part
        
        # Normalize
        max_val = torch.max(torch.abs(mixed_signal), dim=1, keepdim=True)[0] + 1e-8
        return mixed_signal / max_val

if __name__ == "__main__":
    # Test DDSP Synth instantiation and forward pass
    synth = DifferentiableDDSPSynth()
    
    batch_size = 2
    num_samples = 32000 # 2 seconds
    num_frames = num_samples // 256
    
    f0 = 100.0 + torch.rand(batch_size, num_samples, 1) * 200.0 # 100Hz to 300Hz
    amplitude = torch.rand(batch_size, num_samples, 1)
    
    # Harmonic distribution must sum to 1 across harmonics
    raw_dist = torch.rand(batch_size, num_samples, 32)
    harmonic_distribution = torch.softmax(raw_dist, dim=2)
    
    # Filter bands
    noise_filter_magnitudes = torch.rand(batch_size, num_frames, 65)
    
    out = synth(f0, amplitude, harmonic_distribution, noise_filter_magnitudes)
    print("DDSP Differentiable Synthesizer compiled successfully!")
    print(f"Output audio shape: {out.shape}")
