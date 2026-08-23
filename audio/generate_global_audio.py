import os
import wave
import numpy as np
from scipy.signal import butter, lfilter

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BEDS_DIR = "/cs/student/msc/dsml/2023/myeung/THESIS/Claude/ucl_dsml_thesis_claude/FINAL_CONDUCTOR/app/SideProjects/beds"

SR = 16000          # 16 kHz mono sampling rate
DURATION = 10.0      # 10.0 seconds per clip
N_SAMPLES = int(SR * DURATION)
t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def butter_highpass(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

def apply_fade(signal, sr, fade_len=0.5):
    n_fade = int(sr * fade_len)
    env = np.ones_like(signal)
    if len(signal) > 2 * n_fade:
        env[:n_fade] = np.linspace(0.0, 1.0, n_fade)
        env[-n_fade:] = np.linspace(1.0, 0.0, n_fade)
    return signal * env

def save_wav(filename, signal, sr):
    signal = apply_fade(signal, sr, fade_len=0.4)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = 0.92 * (signal / max_val)
    int_signal = (signal * 32767).astype(np.int16)
    filepath = os.path.join(OUT_DIR, filename)
    with wave.open(filepath, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(int_signal.tobytes())
    print(f"Generated: {filepath}")

# Simple dark room convolution impulse response
def dark_reverb(signal, sr, rt60=2.5, wet=0.35):
    ir_len = int(sr * rt60)
    ir_t = np.linspace(0, rt60, ir_len)
    ir = np.random.normal(0, 1, ir_len) * np.exp(-3.0 * ir_t / rt60)
    ir = butter_lowpass(ir, 1800, sr, order=2)
    ir = ir / (np.max(np.abs(ir)) + 1e-6)
    wet_signal = np.convolve(signal, ir, mode='full')[:len(signal)]
    return (1.0 - wet) * signal + wet * wet_signal

# ---------------------------------------------------------------------------
# 1. Continuant Morphology: Organic LFO Swell, Micro-Movement & Pitch Drift
# ---------------------------------------------------------------------------
def make_continuant_morphology():
    f0 = 110.0  # A2
    harmonics = np.arange(1, 33)
    h_weights = np.exp(-harmonics / 2.5)
    h_weights /= np.sum(h_weights)
    
    # Slow pitch drift (cents wander +- 1.5 cents)
    drift = 1.5 * np.sin(2 * np.pi * 0.12 * t)
    f_t = f0 * (2 ** (drift / 1200.0))
    phase = 2 * np.pi * np.cumsum(f_t) / SR
    
    # Fundamental + Fifth + Octave
    voice_root = np.zeros(N_SAMPLES)
    for k, w in zip(harmonics, h_weights):
        voice_root += w * np.sin(k * phase)
        
    voice_fifth = 0.65 * np.sin(1.500 * phase)
    voice_octave = 0.40 * np.sin(2.000 * phase)
    
    drone = voice_root + voice_fifth + voice_octave
    
    # Slow breathing LFO swell envelope (0.1 Hz)
    lfo_swell = 0.65 + 0.35 * np.sin(2 * np.pi * 0.10 * t)
    drone = drone * lfo_swell
    
    # Subtle noise texture
    noise = np.random.normal(0, 1.0, N_SAMPLES)
    noise_filtered = 0.08 * butter_lowpass(noise, 1200, SR)
    
    out = dark_reverb(drone + noise_filtered, SR, rt60=2.0, wet=0.3)
    save_wav("global_continuant_morphology.wav", out, SR)

# ---------------------------------------------------------------------------
# 2. Tape Loops / Feedback: Recursive Delay Network with Tape Saturation
# ---------------------------------------------------------------------------
def make_tape_loops():
    # Base short ambient chord burst at t=0.5s and t=4.5s
    f0 = 130.81  # C3
    f_third = 130.81 * (2 ** (4/12))  # E3
    f_fifth = 130.81 * 1.5           # G3
    
    base_signal = np.zeros(N_SAMPLES)
    # Short 2-second chord swell as input to the loop
    pulse_t = np.linspace(0, 2.5, int(SR * 2.5))
    pulse_env = np.sin(np.pi * pulse_t / 2.5) ** 2
    chord = (np.sin(2 * np.pi * f0 * pulse_t) + 
             0.7 * np.sin(2 * np.pi * f_third * pulse_t) + 
             0.8 * np.sin(2 * np.pi * f_fifth * pulse_t)) * pulse_env
    
    base_signal[int(SR * 0.2): int(SR * 0.2) + len(chord)] += 0.8 * chord
    base_signal[int(SR * 4.5): int(SR * 4.5) + len(chord)] += 0.6 * chord
    
    # Tape delay buffer with feedback and high-frequency loss
    delay_samples = int(SR * 1.6)
    feedback_gain = 0.68
    buffer = np.zeros(N_SAMPLES + delay_samples * 2)
    output = np.zeros(N_SAMPLES)
    
    # Wow and flutter modulation
    flutter = 0.002 * np.sin(2 * np.pi * 3.5 * t) + 0.004 * np.sin(2 * np.pi * 0.3 * t)
    
    for i in range(N_SAMPLES):
        delayed_idx = i - delay_samples + int(flutter[i] * SR)
        delayed_val = buffer[delayed_idx] if delayed_idx >= 0 else 0.0
        # Tape saturation (tanh) and lowpass filtering on feedback
        loop_val = base_signal[i] + feedback_gain * np.tanh(delayed_val * 1.2)
        buffer[i] = loop_val
        output[i] = loop_val
        
    output = butter_lowpass(output, 2400, SR, order=2)
    out = dark_reverb(output, SR, rt60=3.0, wet=0.4)
    save_wav("global_tape_loops.wav", out, SR)

# ---------------------------------------------------------------------------
# 3. Vertical Time: Beatless Static Root-Fifth Harmonic Suspension
# ---------------------------------------------------------------------------
def make_vertical_time():
    f0 = 98.0  # G2
    harmonics = np.arange(1, 33)
    h_weights = (harmonics ** -1.2) / np.sum(harmonics ** -1.2)
    
    phase_root = 2 * np.pi * f0 * t
    voice_root = np.zeros(N_SAMPLES)
    for k, w in zip(harmonics, h_weights):
        # Microtonal phase offset to create rich natural interference
        voice_root += w * np.sin(k * phase_root + k * 0.15)
        
    # Perfect fifth (D3 = 147 Hz) and octave doubling (G3 = 196 Hz)
    phase_fifth = 2 * np.pi * (f0 * 1.500) * t + 0.4
    voice_fifth = 0.70 * np.sin(phase_fifth) + 0.35 * np.sin(2 * phase_fifth)
    
    phase_oct = 2 * np.pi * (f0 * 2.000) * t + 0.8
    voice_oct = 0.45 * np.sin(phase_oct)
    
    static_drone = voice_root + voice_fifth + voice_oct
    # Static room reverb
    out = dark_reverb(static_drone, SR, rt60=3.5, wet=0.45)
    save_wav("global_vertical_time.wav", out, SR)

# ---------------------------------------------------------------------------
# 4. Soundscape Composition: Bimodal Environmental Bed + DDSP Drone
# ---------------------------------------------------------------------------
def make_soundscape_composition():
    # Load authentic environmental nature bed
    bed_path = os.path.join(BEDS_DIR, "r_0nature118777_1708550-hq.wav")
    bed_signal = np.zeros(N_SAMPLES)
    
    if os.path.exists(bed_path):
        with wave.open(bed_path, 'rb') as w:
            n_frames = min(w.getnframes(), N_SAMPLES)
            frames = w.readframes(n_frames)
            raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            bed_sr = w.getframerate()
            # If stereo, convert to mono
            if w.getnchannels() == 2:
                raw = (raw[0::2] + raw[1::2]) * 0.5
            # Simple resample / match length
            bed_len = min(len(raw), N_SAMPLES)
            bed_signal[:bed_len] = raw[:bed_len]
            # Loop if bed is shorter than 10s
            if bed_len < N_SAMPLES:
                rem = N_SAMPLES - bed_len
                bed_signal[bed_len:] = raw[:rem]
    else:
        # Fallback gentle rain simulation if bed file not found
        rain = np.random.normal(0, 1.0, N_SAMPLES)
        bed_signal = 0.3 * butter_highpass(butter_lowpass(rain, 3500, SR), 400, SR)
        
    # High-pass filter bed to make space for drone low end
    bed_signal = butter_highpass(bed_signal, 250, SR, order=2)
    bed_signal = 0.35 * (bed_signal / (np.max(np.abs(bed_signal)) + 1e-6))
    
    # Warm DDSP low-register drone (C2 = 65.4 Hz + fifth G2 = 98.1 Hz)
    f0 = 65.41
    harmonics = np.arange(1, 33)
    h_weights = np.exp(-harmonics / 1.8)
    h_weights /= np.sum(h_weights)
    
    drone = np.zeros(N_SAMPLES)
    for k, w in zip(harmonics, h_weights):
        drone += w * np.sin(2 * np.pi * k * f0 * t)
    drone += 0.6 * np.sin(2 * np.pi * (f0 * 1.5) * t)
    
    drone = dark_reverb(drone, SR, rt60=2.5, wet=0.35)
    drone = 0.65 * (drone / (np.max(np.abs(drone)) + 1e-6))
    
    # Bimodal sum
    bimodal = drone + bed_signal
    save_wav("global_soundscape_composition.wav", bimodal, SR)

# ---------------------------------------------------------------------------
# 5. Affective Trajectory: S-Curve Emotional Transition (Dark/Tense -> Warm/Serene)
# ---------------------------------------------------------------------------
def make_affective_trajectory():
    # S-curve transition parameter s(t) from 0 to 1 between t=2.5s and t=7.5s
    s_t = np.zeros(N_SAMPLES)
    for i, curr_t in enumerate(t):
        if curr_t < 2.5:
            s_t[i] = 0.0
        elif curr_t > 7.5:
            s_t[i] = 1.0
        else:
            s_t[i] = 0.5 - 0.5 * np.cos(np.pi * (curr_t - 2.5) / 5.0)
            
    # State A (Dark, Tense): Low D2 (73.4 Hz), minor third (87.3 Hz), low tilt (darker)
    f0_A = 73.42
    f_third_A = 73.42 * (2 ** (3/12))  # Minor 3rd
    
    # State B (Warm, Serene): A2 (110.0 Hz), major third (138.6 Hz), fifth (165.0 Hz)
    f0_B = 110.0
    f_third_B = 110.0 * (2 ** (4/12))  # Major 3rd
    f_fifth_B = 110.0 * 1.5
    
    # Morphing fundamental frequency and chord voices
    f0_curr = (1.0 - s_t) * f0_A + s_t * f0_B
    phase_main = 2 * np.pi * np.cumsum(f0_curr) / SR
    
    # Harmonics morph from dark sparse to rich warm
    harmonics = np.arange(1, 33)
    voice_main = np.zeros(N_SAMPLES)
    for k in harmonics:
        # Interpolate harmonic roll-off
        w_A = np.exp(-k / 1.5)
        w_B = (k ** -1.2)
        w_k = (1.0 - s_t) * w_A + s_t * w_B
        voice_main += w_k * np.sin(k * phase_main)
        
    # Minor 3rd voice fades out as Major 3rd & 5th fade in
    phase_min3 = 2 * np.pi * f_third_A * t
    phase_maj3 = 2 * np.pi * f_third_B * t
    phase_5th = 2 * np.pi * f_fifth_B * t
    
    voice_min3 = (1.0 - s_t) * 0.75 * np.sin(phase_min3)
    voice_maj3 = s_t * 0.65 * np.sin(phase_maj3)
    voice_5th = s_t * 0.70 * np.sin(phase_5th)
    
    synth_morph = voice_main + voice_min3 + voice_maj3 + voice_5th
    out = dark_reverb(synth_morph, SR, rt60=2.8, wet=0.35)
    save_wav("global_affective_trajectory.wav", out, SR)

if __name__ == "__main__":
    make_continuant_morphology()
    make_tape_loops()
    make_vertical_time()
    make_soundscape_composition()
    make_affective_trajectory()
    print("All 5 Global-Level Composition audio clips generated successfully!")
