from math import atan2, degrees
from unittest import signals as unittest_signals
import librosa
from scipy.signal import chirp, firwin, lfilter, correlate, spectrogram
import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks, correlation_lags
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import chirp, correlate, spectrogram, firwin, lfilter
from filterpy.kalman import KalmanFilter
from threading import Thread, Event
from queue import Queue
import time
import matplotlib
matplotlib.use('TkAgg')


class AudioCapture:
    def __init__(self, fs=44100, channels=None, blocksize=1024, device=None):
        # Query available devices
        devices = sd.query_devices()
        input_devices = [i for i, dev in enumerate(devices)
                         if dev['max_input_channels'] > 0]

        # Find best device (prioritizing requested device if specified)
        selected_device = None
        if device is not None:
            if isinstance(device, int) and device < len(devices):
                selected_device = device
            elif isinstance(device, str):
                for i, dev in enumerate(devices):
                    if device.lower() in dev['name'].lower():
                        selected_device = i
                        break

        # Fallback to default input device
        if selected_device is None:
            selected_device = sd.default.device[0]

        # Get device info
        device_info = sd.query_devices(selected_device, 'input')
        max_channels = device_info['max_input_channels']

        # Set channels (respecting hardware limits)
        if channels is None:
            self.channels = min(4, max_channels)  # Default to up to 4 channels
        else:
            self.channels = min(channels, max_channels)

        self.fs = fs
        self.blocksize = blocksize
        self.input_queue = Queue()
        self.stream = None
        self.stop_event = Event()
        self.device = selected_device

        print(f"Using device: {device_info['name']}")
        print(
            f"Using {self.channels} of {max_channels} available input channels")

    def callback(self, indata, frames, time, status):
        """Sounddevice callback for real-time audio capture"""
        if status:
            print(f"Audio stream status: {status}")
        self.input_queue.put(indata.copy())

    def start(self):
        """Start audio capture stream"""
        try:
            self.stream = sd.InputStream(
                samplerate=self.fs,
                channels=self.channels,
                callback=self.callback,
                blocksize=self.blocksize,
                device=self.device
            )
            self.stream.start()
        except sd.PortAudioError as e:
            print(f"Failed to start audio stream: {e}")
            print("Try reducing the number of channels")
            raise

    def stop(self):
        """Stop audio capture"""
        self.stop_event.set()
        if self.stream:
            self.stream.stop()
            self.stream.close()


class RadarVisualization:
    def __init__(self, processor):
        self.processor = processor
        self.fig = plt.figure(figsize=(10, 8), facecolor='black')
        self.setup_plots()
        self.animation = None
        self.sweep_angle = 0
        self.sweep_speed = 0.15  # Faster sweep speed
        self._is_running = False
        self.last_update_time = time.time()

    def setup_plots(self):
        """Initialize radar display with better layout"""
        self.fig.patch.set_facecolor('black')

        # Main radar display (larger and centered)
        self.ax_radar = self.fig.add_subplot(2, 2, (1, 3), projection='polar')
        self.ax_radar.set_facecolor('navy')
        self.ax_radar.set_title('Active Sonar Display', color='white', pad=20)

        # Information panel (top right)
        self.ax_info = self.fig.add_subplot(2, 2, 2)
        self.ax_info.set_facecolor('black')
        self.ax_info.axis('off')

        # Signal waveform (bottom right)
        self.ax_signal = self.fig.add_subplot(2, 2, 4)
        self.ax_signal.set_facecolor('black')
        self.ax_signal.set_title('Signal Waveform', color='white')
        self.ax_signal.tick_params(colors='white')
        self.ax_signal.xaxis.label.set_color('white')
        self.ax_signal.yaxis.label.set_color('white')

        # Configure radar display
        self.ax_radar.set_theta_zero_location('N')
        self.ax_radar.set_theta_direction(-1)  # Clockwise rotation
        self.ax_radar.set_ylim(0, 10)  # 10 meter range
        self.ax_radar.grid(color='lime', linestyle='--', alpha=0.7)

        # Draw range circles
        for r in range(2, 11, 2):
            circle = plt.Circle((0, 0), r, color='lime', fill=False, alpha=0.3)
            self.ax_radar.add_artist(circle)

        # Add compass labels
        self.ax_radar.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        self.ax_radar.set_xticklabels(
            ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

        plt.tight_layout()

    def update_plots(self, frame):
        """Update radar display with new data"""
        # Get latest data from processor
        positions, spectrograms, _ = self.processor.get_latest_data()

        # Clear and update radar display
        self.ax_radar.clear()
        self.setup_radar_base()

        # Enhanced direction indicators
        if positions and len(positions[0]) == 2:
            distance, angle = positions[0]

            # Convert angle to degrees for display
            angle_deg = np.degrees(angle) % 360
            compass_dir = self.get_compass_direction(angle_deg)

            # Big red detection dot
            self.ax_radar.scatter(angle, distance, s=300, color='red',
                                  edgecolors='white', zorder=10,
                                  label=f'Source: {compass_dir}')

            # Direction arrow
            self.ax_radar.annotate('',
                                   xy=(angle, distance*0.8),
                                   xytext=(angle, distance*0.2),
                                   arrowprops=dict(arrowstyle="->", color='red', lw=2))

            # Distance ring highlight
            circle = plt.Circle((0, 0), distance, color='red',
                                fill=False, alpha=0.3, linestyle='--')
            self.ax_radar.add_artist(circle)

            self.sweep_angle = angle
        else:
            # Continue sweeping when no detection
            self.sweep_angle = (self.sweep_angle +
                                self.sweep_speed) % (2*np.pi)

        # Draw sweeping line
        self.ax_radar.plot([self.sweep_angle, self.sweep_angle], [0, 10],
                           color='cyan', linewidth=3, alpha=0.8)

        # Update info panel with more detailed direction info
        self.update_info_panel(positions)

        # Update signal plot
        if hasattr(self.processor, 'last_signal'):
            self.update_signal_plot(self.processor.last_signal)

    def get_compass_direction(self, angle_deg):
        """Convert angle to precise compass direction"""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        index = round(angle_deg / (360./len(directions))) % len(directions)
        return directions[index]
    
    def setup_radar_base(self):
        """Redraw constant radar elements"""
        self.ax_radar.set_theta_zero_location('N')
        self.ax_radar.set_theta_direction(-1)
        self.ax_radar.set_ylim(0, 10)
        self.ax_radar.grid(color='lime', linestyle='--', alpha=0.7)

        # Redraw range circles
        for r in range(2, 11, 2):
            circle = plt.Circle((0, 0), r, color='lime', fill=False, alpha=0.3)
            self.ax_radar.add_artist(circle)

        # Redraw compass labels
        self.ax_radar.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        self.ax_radar.set_xticklabels(
            ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    def update_info_panel(self, positions):
        """Enhanced information display"""
        self.ax_info.clear()
        self.ax_info.set_facecolor('black')
        self.ax_info.axis('off')

        if positions and len(positions[0]) == 2:
            distance, angle = positions[0]
            angle_deg = np.degrees(angle) % 360
            compass_dir = self.get_compass_direction(angle_deg)
            status = "LOCKED"
            status_color = "lime"
        else:
            distance = 0.0
            angle_deg = np.degrees(self.sweep_angle) % 360
            compass_dir = self.get_compass_direction(angle_deg)
            status = "SCANNING"
            status_color = "yellow"

        info_lines = [
            f"STATUS: {status}",
            f"Direction: {compass_dir} ({angle_deg:.1f}°)",
            f"Distance: {distance:.1f} meters",
            f"Sound Type: {self.processor.last_sound_type}",
            "",
            "SWEEP INFORMATION:",
            f"Current Angle: {angle_deg:.1f}°",
            f"Microphones: {self.processor.mic_positions.shape[0]} active"
        ]

        # Display with color coding
        for idx, line in enumerate(info_lines):
            y_pos = 0.9 - idx * 0.08
            color = "white"
            weight = "normal"
            size = 10

            if line.startswith("STATUS:"):
                color = status_color
                weight = "bold"
                size = 12
            elif line.startswith("Direction:"):
                color = "cyan"
            elif line.startswith("SWEEP"):
                color = "yellow"
                size = 9

            self.ax_info.text(0.05, y_pos, line,
                              transform=self.ax_info.transAxes,
                              color=color, fontsize=size, weight=weight,
                              family='monospace')

    def update_signal_plot(self, signal):
        """Update signal waveform plot"""
        self.ax_signal.clear()
        self.ax_signal.set_facecolor('black')

        t = np.arange(len(signal)) / self.processor.fs
        self.ax_signal.plot(t, signal, color='cyan')
        self.ax_signal.set_xlabel('Time [sec]', color='white')
        self.ax_signal.set_ylabel('Amplitude', color='white')
        self.ax_signal.tick_params(colors='white')
        self.ax_signal.set_title('Signal Waveform', color='white')

    def start_animation(self):
        """Start radar animation"""
        if not self._is_running:
            try:
                self._is_running = True
                self.animation = FuncAnimation(
                    self.fig,
                    self.update_plots,
                    interval=50,  # Faster update for smoother sweep
                    cache_frame_data=False,
                    blit=False
                )
                plt.show()
            except Exception as e:
                print(f"Animation error: {e}")
                self._is_running = False

    def stop_animation(self):
        """Stop radar display"""
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
        self._is_running = False
        plt.close(self.fig)

    def update_data(self, positions, spectrograms):
        """Store the latest data for visualization"""
        self.positions = positions
        self.spectrograms = spectrograms


class SignalProcessor:
    def __init__(self, fs: int = 44100, mic_positions: np.ndarray = None):
        self.fs = fs
        self.speed_of_sound = 343  # Speed of sound in m/s
        self.mic_positions = mic_positions or self.default_mic_positions()
        self.ping = self.generate_ping()
        self.matched_filter = self.ping[::-1]
        self.kalman_filters = {}
        self.last_signal = None
        self.last_direction = "No source detected"
        self.last_distance = 0
        self.last_direction = "Unknown"
        self.last_distance = 0.0
        self.last_sound_type = "Unknown"

    @staticmethod
    def default_mic_positions(num_channels=4) -> np.ndarray:
        """Returns microphone positions with configurable channels"""
        mic_distance = 0.2  # 20cm between mics
        positions = [
            [0, 0, 0],          # Mic 1 at origin
            [mic_distance, 0, 0],  # Mic 2 along x-axis
            [0, mic_distance, 0],  # Mic 3 along y-axis (optional)
            [0, 0, mic_distance]   # Mic 4 along z-axis (optional)
        ]
        return np.array(positions[:num_channels])

    def classify_sound(self, signal):
        """Basic sound classification based on spectral characteristics"""
        if len(signal) == 0:
            return "Unknown"

        # Compute FFT
        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1/self.fs)

        # Find dominant frequency
        dominant_freq = freqs[np.argmax(fft)]

        if dominant_freq < 300:
            return "Low frequency (e.g., engine)"
        elif 300 <= dominant_freq < 2000:
            return "Mid frequency (e.g., voice)"
        else:
            return "High frequency (e.g., chirp)"

    def compute_direction(self, signal):
        """Compute direction from signal characteristics"""
        # This should use your TDOA calculations
        if not hasattr(self, 'last_direction'):
            return "Unknown"
        return self.last_direction

    def estimate_distance(self, signals, tdoas):
        """Estimate distance using TDOA and known mic positions."""
        if len(tdoas) < 1 or len(self.mic_positions) < 2:
            return 0.0

        # Distance = (speed_of_sound * time_delay) / mic_separation
        mic1, mic2 = self.mic_positions[:2]
        d_mic = np.linalg.norm(mic2 - mic1)  # Distance between mics
        tdoa = tdoas[0]  # Time delay between mic1 and mic2

        # Basic triangulation (simplified)
        distance = (self.speed_of_sound * abs(tdoa)) / d_mic
        return distance

    def get_direction_description(self, angle_deg: float) -> str:
        """Convert angle to directional description"""
        if -45 <= angle_deg < 45:
            return "Front"
        elif 45 <= angle_deg < 135:
            return "Right"
        elif -135 <= angle_deg < -45:
            return "Left"
        else:
            return "Back"

    def estimate_position(self, tdoas: list) -> np.ndarray:
        """Return consistent polar coordinates (distance, azimuth)"""
        if len(tdoas) < 1 or len(self.mic_positions) < 2:
            return np.array([0, 0])  # (distance, angle)
        
        # For 2 mics - basic direction estimation
        mic1, mic2 = self.mic_positions[:2]
        d_mic = np.linalg.norm(mic2 - mic1)  # Distance between mics
        tdoa = tdoas[0]
        
        # Calculate angle (in radians)
        angle = np.arcsin((tdoa * self.speed_of_sound) / d_mic)
        
        # Simple distance estimation (could be improved)
        distance = 1.0  # Placeholder - should use signal strength
        
        return np.array([distance, angle])

    def process_frame(self, audio_frame: np.ndarray):
        """Enhanced frame processing with better detection"""
        # Convert to 2D array if needed
        if len(audio_frame.shape) == 1:
            audio_frame = audio_frame[np.newaxis, :]

        # Basic quality check
        if audio_frame.size == 0 or np.max(np.abs(audio_frame)) < 0.01:
            self.last_direction = "No signal"
            self.last_distance = 0.0
            self.last_sound_type = "Silence"
            return [np.zeros(3)], [], []

        # Apply processing
        cleaned_signals = self.noise_reduction(audio_frame)
        filtered_signals = self.bandpass_filter(cleaned_signals)
        filtered_signals = [self.apply_matched_filter(
            sig) for sig in filtered_signals]

        # Classify sound from first channel
        self.last_sound_type = self.classify_sound(filtered_signals[0])

        # Calculate TDOAs
        tdoas = self.compute_tdoas(filtered_signals)
        sources = self.detect_multiple_sources(filtered_signals, tdoas)

        positions = []
        for source in sources:
            pos = self.estimate_position(source['tdoas'])
            if np.any(pos):  # Only track non-zero positions
                pos = self.track_position(pos, source['id'])
                positions.append(pos)

        spectrograms = [self.compute_spectrogram(
            sig) for sig in filtered_signals]

        return positions, spectrograms, tdoas

    def generate_ping(self, duration: float = 0.01, f0: int = 2000, f1: int = 8000) -> np.ndarray:
        """Generate a linear chirp ping signal."""
        t = np.linspace(0, duration, int(self.fs * duration))
        return chirp(t, f0=f0, f1=f1, t1=duration, method='linear')

    def bandpass_filter(self, signals: list, lowcut: int = 1500, highcut: int = 8500, order: int = 5) -> list:
        """Apply a bandpass filter to each audio channel."""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b = firwin(order, [low, high], pass_zero=False)
        return [lfilter(b, 1, sig) for sig in signals]

    def apply_matched_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply matched filter for pulse compression."""
        return correlate(signal, self.matched_filter, mode='same')

    def compute_tdoas(self, signals, ref_mic=0):
        if not isinstance(signals, (list, np.ndarray)):
            raise ValueError("`signals` must be a list or numpy array.")
        tdoas = []
        ref_signal = signals[ref_mic]

        for i, sig in enumerate(signals):
            if i == ref_mic:
                continue

            # Compute cross-correlation
            corr = correlate(ref_signal, sig, mode='full')
            lags = correlation_lags(len(ref_signal), len(sig), mode='full')

            # Find integer peak index
            peak_idx = np.argmax(corr)
            tdoa = lags[peak_idx] / self.fs  # Convert to time

            tdoas.append(tdoa)

        return tdoas

    def detect_multiple_sources(self, signals: list, tdoas: list, threshold: float = 0.5) -> list:
        """Detect multiple sound sources using peak detection."""
        sources = []
        ref_signal = signals[0]
        peaks, _ = self.find_peaks(ref_signal, threshold=threshold)

        for peak in peaks:
            source_tdoas = []
            for i in range(1, len(signals)):
                start = np.clip(peak - 100, 0, len(signals[i]) - 1)
                end = np.clip(peak + 100, start + 1, len(signals[i]))
                corr = correlate(signals[0][start:end],
                                 signals[i][start:end], mode='full')
                lag = np.argmax(corr) - (end - start) // 2
                source_tdoas.append(lag / self.fs)

            sources.append({
                'id': f"src_{peak}",
                'tdoas': source_tdoas,
                'amplitude': np.max(ref_signal[max(peak-10, 0):min(peak+10, len(ref_signal))])
            })

        return sources

    def estimate_position(self, tdoas):
        """Return (distance, angle) instead of full 3D position when limited"""
        if len(tdoas) < 1 or len(self.mic_positions) < 2:
            return (0, 0)  # (distance, angle)

        # For 2 mics - basic direction estimation
        mic1, mic2 = self.mic_positions[:2]
        d_mic = np.linalg.norm(mic2 - mic1)  # Distance between mics
        tdoa = tdoas[0]

        # Calculate angle (in radians)
        angle = np.arcsin((tdoa * self.speed_of_sound) / d_mic)

        # Simple distance estimation (amplitude-based)
        distance = 1.0

        return (distance, angle)

    def track_position(self, position: np.ndarray, obj_id: str) -> np.ndarray:
        """Track object position using a Kalman filter."""
        if obj_id not in self.kalman_filters:
            kf = KalmanFilter(dim_x=6, dim_z=3)
            kf.x = np.hstack((position, [0, 0, 0]))
            kf.P *= 1000
            kf.R = np.eye(3) * 0.1
            kf.Q = np.eye(6) * 0.01
            kf.F = np.eye(6)
            for i in range(3):
                kf.F[i, i+3] = 1
            kf.H = np.zeros((3, 6))
            kf.H[:3, :3] = np.eye(3)
            self.kalman_filters[obj_id] = kf

        kf = self.kalman_filters[obj_id]
        kf.predict()
        kf.update(position)
        return kf.x[:3]

    def compute_spectrogram(self, signal: np.ndarray, nperseg: int = 256) -> dict:
        """Compute a spectrogram with validation."""
        try:
            f, t, Sxx = spectrogram(signal, fs=self.fs, nperseg=nperseg)
            Sxx_db = 10 * np.log10(Sxx) if Sxx.size > 0 else np.array([[]])
            return {'freqs': f, 'times': t, 'spectrogram': Sxx_db}
        except Exception as e:
            print(f"Spectrogram error: {e}")
            return {'freqs': np.array([]), 'times': np.array([]), 'spectrogram': np.array([[]])}

    @staticmethod
    def find_peaks(signal: np.ndarray, threshold: float = 0.5, min_distance: int = 100):
        peaks, _ = scipy_find_peaks(
            signal, height=threshold, distance=min_distance)
        return peaks, signal[peaks]

    def get_latest_data(self):
        """Return dummy latest data format."""
        dummy_spec = {
            'freqs': np.array([]),
            'times': np.array([]),
            'spectrogram': np.array([[]])
        }
        return [np.zeros(3)], [dummy_spec], []

    def process(self, signal):
        # Example placeholders:
        self.last_direction = self.compute_direction(signal)
        self.last_distance = self.estimate_distance(signal)
        self.last_sound_type = self.classify_sound(signal)

    def noise_reduction(self, signals):
        """Apply basic noise reduction to audio signals"""
        if len(signals.shape) == 1:
            signals = signals[np.newaxis, :]  # Ensure 2D array

        cleaned_signals = []
        for sig in signals:
            # Simple moving average filter for noise reduction
            window_size = 5
            if len(sig) > window_size:
                cleaned = np.convolve(sig, np.ones(
                    window_size)/window_size, mode='same')
            else:
                cleaned = sig.copy()
            cleaned_signals.append(cleaned)

        return np.array(cleaned_signals)

    def spectral_gating(self, signal):
        """More advanced noise reduction using spectral gating"""
        # Requires librosa - install with: pip install librosa
        try:
            S = np.abs(librosa.stft(signal))
            mask = librosa.util.softmask(S, 0.2 * S, power=2)
            return librosa.istft(S * mask)
        except ImportError:
            return signal


class AudioRadarSystem:
    def __init__(self):
        # Let AudioCapture determine the correct number of channels
        self.audio = AudioCapture()
        self.processor = SignalProcessor()
        # Only use as many mic positions as we have channels
        default_positions = self.processor.default_mic_positions()
        self.processor.mic_positions = default_positions[:self.audio.channels]
        self.visualizer = RadarVisualization(self.processor)
        self.running = False

    def start(self):
        """Start the complete system"""
        self.running = True

        # Start audio capture in separate thread
        audio_thread = Thread(target=self.audio_capture.start)
        audio_thread.daemon = True
        audio_thread.start()

        # Start processing thread
        processing_thread = Thread(target=self.process_audio)
        processing_thread.daemon = True
        processing_thread.start()

        # Start visualization
        self.visualization.start_animation()

    def process_audio(self):
        """Process incoming audio frames"""
        while self.running:
            if not self.audio_capture.input_queue.empty():
                audio_frame = self.audio_capture.input_queue.get()

                # Store last signal for visualization
                self.signal_processor.last_signal = audio_frame[:, 0]

                # Process the frame
                positions, spectrograms, tdoas = self.signal_processor.process_frame(
                    audio_frame.T)

                # Update visualization data
                self.visualization.update_data(positions, spectrograms)

    def stop(self):
        """Stop the system"""
        self.running = False
        self.audio_capture.stop()
        plt.close('all')


class RadarSystem:
    def __init__(self):
        # Initialize audio with automatic channel detection
        self.audio = AudioCapture(channels=None)

        # Initialize processor with correct number of mics
        self.processor = SignalProcessor()
        self.processor.mic_positions = self.processor.default_mic_positions(
            num_channels=self.audio.channels)

        self.visualizer = RadarVisualization(self.processor)
        self.running = False

    def start(self):
        """Start radar system"""
        self.audio.start()
        self.running = True
        processing_thread = Thread(target=self.process_loop)
        processing_thread.start()
        self.visualizer.start_animation()

    def process_loop(self):
        """Background loop for processing audio frames"""
        while not self.audio.stop_event.is_set():
            try:
                frame = self.audio.input_queue.get(timeout=1)
                frame = frame.T
                self.processor.last_signal = frame[0]
                positions, spectrograms, tdoas = self.processor.process_frame(
                    frame)
                self.visualizer.update_data(positions, spectrograms)
            except Exception as e:
                print(f"Processing error: {e}")
