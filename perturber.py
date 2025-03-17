import os
import sys
import random
import torch
import opuspy
import librosa
import numpy as np
import signal
import torchaudio
import torch.nn.functional as F
import typing as tp
import julius
import scipy.signal as signal
from numpy.typing import NDArray
from transformers import EncodecModel, AutoProcessor
from copy import deepcopy
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_float_samples_to_int16, get_max_abs_amplitude,
)
import tempfile
import uuid

class AudioPerturbation:
    """
    A class that provides various audio perturbation methods for audio signals.
    """

    def __init__(self, sample_rate: int = 16000, device=None):
        """
        Initialize the AudioPerturbation class.

        Args:
            sample_rate: The sample rate of the audio signals (default: 16000)
            device: The device to use for torch operations (default: None)
        """
        self.sample_rate = sample_rate
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize EncodecModel and processor if needed
        self._model_encodec = None
        self._processor_encodec = None

        # Download noise sample for background noise perturbation if needed
        self._noise_sample_path = None

    def _audio_effect_return(self, tensor: torch.Tensor, mask: tp.Optional[torch.Tensor]
                             ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return the mask if it was in the input otherwise only the output tensor"""
        if mask is None:
            return tensor
        else:
            return tensor, mask

    def _compute_snr(self, clean_signal, noisy_signal):
        """Compute Signal-to-Noise Ratio between clean and noisy signals"""

        signal_power = torch.mean(clean_signal ** 2)

        if clean_signal.size(0) > noisy_signal.size(0):
            clean_signal = clean_signal[:noisy_signal.size(0)]
        else:
            noisy_signal = noisy_signal[:clean_signal.size(0)]

        noise = noisy_signal - clean_signal
        noise_power = torch.mean(noise ** 2)
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr

    def echo(self,
             tensor: torch.Tensor,
             volume: float = 0.4,
             duration: float = 0.1,
             mask: tp.Optional[torch.Tensor] = None,
             ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Attenuating the audio volume by a factor of volume, delaying it by duration seconds,
        and then overlaying it with the original.

        Args:
            tensor: 3D Tensor representing the audio signal [bsz, channels, frames]
            volume: Volume of the echo signal (default: 0.4)
            duration: Duration of the echo delay in seconds (default: 0.1)
            mask: Optional mask tensor

        Returns:
            Audio signal with echo effect applied
        """
        tensor = tensor.unsqueeze(0)

        duration = torch.Tensor([duration])
        volume = torch.Tensor([volume])
        n_samples = int(self.sample_rate * duration)
        impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

        impulse_response[0] = 1.0  # Direct sound
        impulse_response[int(self.sample_rate * duration) - 1] = volume

        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

        reverbed_signal = julius.fft_conv1d(tensor, impulse_response)

        reverbed_signal = (
                reverbed_signal
                / torch.max(torch.abs(reverbed_signal))
                * torch.max(torch.abs(tensor))
        )
        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
        reverbed_signal = tmp
        reverbed_signal = reverbed_signal.squeeze(0)

        return self._audio_effect_return(tensor=reverbed_signal, mask=mask)

    def gaussian_noise(self, waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add Gaussian noise to the waveform with a specified SNR.

        Args:
            waveform: Input audio waveform
            snr_db: Signal-to-Noise Ratio in decibels

        Returns:
            Noisy waveform
        """
        # Calculate signal power
        signal_power = torch.mean(waveform ** 2).to(device=waveform.device)

        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate noise with calculated noise power
        noise = torch.randn(waveform.size(), device=waveform.device) * torch.sqrt(noise_power)
        waveform_noisy = waveform + noise

        return waveform_noisy

    def background_noise(self, waveform: torch.Tensor, snr_db: float, noise_path: str = None) -> torch.Tensor:
        """
        Add background noise to the waveform with a specified SNR.

        Args:
            waveform: Input audio waveform
            snr_db: Signal-to-Noise Ratio in decibels
            noise_path: Path to noise sample file (optional)

        Returns:
            Waveform with background noise
        """
        if noise_path is None:
            if self._noise_sample_path is None:
                from torchaudio.utils import download_asset
                self._noise_sample_path = download_asset(
                    "tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
            noise_path = self._noise_sample_path

        noise, _ = torchaudio.load(noise_path)
        noise = noise.to(waveform.device)

        if noise.size(1) > waveform.size(1):
            noise = noise[:, :waveform.size(1)]
        else:
            repeat_times = waveform.size(1) // noise.size(1) + 1
            noise = noise.repeat(1, repeat_times)
            noise = noise[:, :waveform.size(1)]

        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)

        snr_linear = 10 ** (snr_db / 10)
        scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
        noisy_waveform = waveform + noise * scaling_factor

        return noisy_waveform

    def opus(self, waveform: torch.Tensor, bitrate: int, quality: int=10, cache: str="./cache/") -> torch.Tensor:
        """
        Apply Opus compression to the waveform.

        Args:
            waveform: Input audio waveform
            bitrate: Bitrate for Opus compression
            quality: Encoder complexity/quality (0-10)
            cache: Directory to store temporary files

        Returns:
            Opus-compressed waveform
        """
        os.makedirs(cache,exist_ok=True)
        waveform_scaled = waveform * 32768
        waveform_scaled = waveform_scaled.reshape(-1, 1).numpy()
        cache_file = os.path.join(cache, "temp.opus")
        opuspy.write(cache_file, waveform_scaled, sample_rate=16000,
                     bitrate=bitrate, signal_type=0, encoder_complexity=quality)
        pert_waveform, sampling_rate = opuspy.read(cache_file)
        os.remove(cache_file)
        resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
        pert_waveform = torch.tensor(pert_waveform, dtype=torch.float32).reshape(1, -1)
        pert_waveform /= 32768

        return resampler(pert_waveform)

    def highpass(self, waveform: torch.Tensor, cutoff_ratio: float, order: int = 5) -> torch.Tensor:
        """
        Apply a highpass filter to the waveform.

        Args:
            waveform: Input audio waveform
            cutoff_ratio: Cutoff frequency ratio (0 to 1) of the Nyquist frequency
            order: Filter order (default: 5)

        Returns:
            Highpass filtered waveform
        """
        waveform_np = waveform.cpu().numpy()
        b, a = signal.butter(order, cutoff_ratio, btype="high", analog=False)
        waveform_pert = signal.lfilter(b, a, waveform_np)

        return torch.from_numpy(waveform_pert).to(waveform.device)

    def lowpass(self, waveform: torch.Tensor, cutoff_ratio: float, order: int = 5) -> torch.Tensor:
        """
        Apply a lowpass filter to the waveform.

        Args:
            waveform: Input audio waveform
            cutoff_ratio: Cutoff frequency ratio (0 to 1) of the Nyquist frequency
            order: Filter order (default: 5)

        Returns:
            Lowpass filtered waveform
        """
        waveform_np = waveform.cpu().numpy()
        b, a = signal.butter(order, cutoff_ratio, btype="low", analog=False)
        waveform_pert = signal.lfilter(b, a, waveform_np)

        return torch.from_numpy(waveform_pert).to(waveform.device)

    def smooth(self,
               waveform: torch.Tensor,
               window_size: int = 5,
               mask: tp.Optional[torch.Tensor] = None,
               ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply smoothing to the waveform using a moving average filter.

        Args:
            waveform: Input audio waveform
            window_size: Size of the smoothing window (default: 5)
            mask: Optional mask tensor

        Returns:
            Smoothed waveform
        """
        waveform = waveform.unsqueeze(0)
        window_size = int(window_size)
        # Create a uniform smoothing kernel
        kernel = torch.ones(1, 1, window_size).type(waveform.type()) / window_size
        kernel = kernel.to(waveform.device)

        smoothed = julius.fft_conv1d(waveform, kernel)
        # Ensure tensor size is not changed
        tmp = torch.zeros_like(waveform)
        tmp[..., : smoothed.shape[-1]] = smoothed
        smoothed = tmp

        return self._audio_effect_return(tensor=smoothed, mask=mask).squeeze().unsqueeze(0)

    def time_stretch(self, waveform: torch.Tensor, speed_factor: float) -> tuple:
        """
        Apply time stretching to the waveform.

        Args:
            waveform: Input audio waveform
            speed_factor: Factor to stretch/compress the waveform (>1 speeds up, <1 slows down)

        Returns:
            Tuple of (time-stretched waveform, sample rate)
        """
        waveform_np = waveform.cpu().numpy()
        if waveform_np.shape[0] == 1:
            waveform_np = waveform_np.squeeze()

        waveform_stretched = librosa.effects.time_stretch(waveform_np, rate=speed_factor)
        time_stretched_waveform = torch.from_numpy(waveform_stretched).unsqueeze(0).float().to(waveform.device)

        if time_stretched_waveform.shape[1] < waveform.shape[1]:
            time_stretched_waveform = F.pad(time_stretched_waveform,
                                            (0, waveform.shape[1] - time_stretched_waveform.shape[1]))
        elif time_stretched_waveform.shape[1] > waveform.shape[1]:
            time_stretched_waveform = time_stretched_waveform[:, :waveform.shape[1]]

        return time_stretched_waveform, self.sample_rate

    def pitch_shift(self, waveform: torch.Tensor, pitch_factor: float = 0.0) -> torch.Tensor:
        """
        Apply pitch shifting to the waveform.

        Args:
            waveform: Input audio waveform
            pitch_factor: Number of semitones to shift the pitch

        Returns:
            Pitch-shifted waveform
        """
        waveform_np = waveform.cpu().numpy().squeeze()
        shifted = librosa.effects.pitch_shift(y=waveform_np, sr=self.sample_rate, n_steps=pitch_factor)
        shifted_waveform = torch.from_numpy(shifted).unsqueeze(0).to(waveform.device)
        return shifted_waveform

    def quantization(self, waveform: torch.Tensor, quantization_bit: int) -> torch.Tensor:
        """
        Apply quantization to the waveform.

        Args:
            waveform: Input audio waveform
            quantization_bit: Number of quantization levels

        Returns:
            Quantized waveform
        """
        # Normalize the waveform to the range of the quantization levels
        min_val, max_val = waveform.min(), waveform.max()
        normalized_waveform = (waveform - min_val) / (max_val - min_val)

        # Quantize the normalized waveform
        quantized_waveform = torch.round(normalized_waveform * (quantization_bit - 1))

        # Rescale the quantized waveform back to the original range
        rescaled_waveform = (quantized_waveform / (quantization_bit - 1)) * (max_val - min_val) + min_val

        return rescaled_waveform

    def mp3(self, waveform: torch.Tensor, bitrate: int) -> torch.Tensor:
        mp3_compressor = Mp3Compression(
            min_bitrate=bitrate,  # Set the minimum bitrate
            max_bitrate=bitrate,  # Set the maximum bitrate
            backend="pydub",  # Choose the backend
            p=1.0)
        waveform = waveform.detach().cpu().numpy()
        mp3_compressor.randomize_parameters(waveform, self.sample_rate)
        waveform_pert = mp3_compressor.apply(waveform, self.sample_rate)
        return torch.tensor(waveform_pert)

    def _load_encodec_model(self):
        """Load EncodecModel and processor if not already loaded"""
        if self._model_encodec is None:
            self._model_encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(self.device)
        if self._processor_encodec is None:
            self._processor_encodec = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    def encodec(self, waveform: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """
        Apply EncodecModel encoding/decoding to the waveform.

        Args:
            waveform: Input audio waveform
            bandwidth: Bandwidth parameter for Encodec

        Returns:
            Processed waveform
        """
        # Load model if not already loaded
        self._load_encodec_model()
        encodec_sample_rate = 24000

        waveform_np = waveform.squeeze().cpu().numpy()
        inputs = self._processor_encodec(raw_audio=waveform_np, sampling_rate=encodec_sample_rate,
                                         return_tensors="pt").to(self.device)

        # Encode and decode the audio sample using ENCODeC
        encoder_outputs = self._model_encodec.encode(inputs["input_values"],
                                                     inputs["padding_mask"],
                                                     bandwidth)
        audio_values = self._model_encodec.decode(encoder_outputs.audio_codes,
                                                  encoder_outputs.audio_scales,
                                                  inputs["padding_mask"])[0]

        processed_waveform = audio_values.clone().detach().cpu().squeeze().unsqueeze(0)

        if encodec_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=encodec_sample_rate,
                new_freq=self.sample_rate
            )
            processed_waveform = resampler(processed_waveform)

        return processed_waveform


class Mp3Compression(BaseWaveformTransform):
    """Compress the audio using an MP3 encoder to lower the audio quality.
    This may help machine learning models deal with compressed, low-quality audio.

    This transform depends on either lameenc or pydub/ffmpeg.

    Note that bitrates below 32 kbps are only supported for low sample rates (up to 24000 Hz).

    Note: When using the lameenc backend, the output may be slightly longer than the input due
    to the fact that the LAME encoder inserts some silence at the beginning of the audio.

    Warning: This transform writes to disk, so it may be slow. Ideally, the work should be done
    in memory. Contributions are welcome.
    """

    supports_multichannel = True

    SUPPORTED_BITRATES = [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        192,
        224,
        256,
        320,
    ]

    def __init__(
            self,
            min_bitrate: int = 8,
            max_bitrate: int = 64,
            backend: str = "pydub",
            p: float = 0.5,
    ):
        """
        :param min_bitrate: Minimum bitrate in kbps
        :param max_bitrate: Maximum bitrate in kbps
        :param backend: "pydub" or "lameenc".
            Pydub may use ffmpeg under the hood.
                Pros: Seems to avoid introducing latency in the output.
                Cons: Slower than lameenc.
            lameenc:
                Pros: You can set the quality parameter in addition to bitrate.
                Cons: Seems to introduce some silence at the start of the audio.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_bitrate < self.SUPPORTED_BITRATES[0]:
            raise ValueError(
                "min_bitrate must be greater than or equal to"
                f" {self.SUPPORTED_BITRATES[0]}"
            )
        if max_bitrate > self.SUPPORTED_BITRATES[-1]:
            raise ValueError(
                "max_bitrate must be less than or equal to"
                f" {self.SUPPORTED_BITRATES[-1]}"
            )
        if max_bitrate < min_bitrate:
            raise ValueError("max_bitrate must be >= min_bitrate")

        is_any_supported_bitrate_in_range = any(
            min_bitrate <= bitrate <= max_bitrate for bitrate in self.SUPPORTED_BITRATES
        )
        if not is_any_supported_bitrate_in_range:
            raise ValueError(
                "There is no supported bitrate in the range between the specified"
                " min_bitrate and max_bitrate. The supported bitrates are:"
                f" {self.SUPPORTED_BITRATES}"
            )

        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        if backend not in ("pydub", "lameenc"):
            raise ValueError('backend must be set to either "pydub" or "lameenc"')
        self.backend = backend
        self.post_gain_factor = None

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            bitrate_choices = [
                bitrate
                for bitrate in self.SUPPORTED_BITRATES
                if self.min_bitrate <= bitrate <= self.max_bitrate
            ]
            self.parameters["bitrate"] = random.choice(bitrate_choices)

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        if self.backend == "lameenc":
            return self.apply_lameenc(samples, sample_rate)
        elif self.backend == "pydub":
            return self.apply_pydub(samples, sample_rate)
        else:
            raise Exception("Backend {} not recognized".format(self.backend))

    def maybe_pre_gain(self, samples):
        """
        If the audio is too loud, gain it down to avoid distortion in the audio file to
        be encoded.
        """
        greatest_abs_sample = get_max_abs_amplitude(samples)
        if greatest_abs_sample > 1.0:
            self.post_gain_factor = greatest_abs_sample
            samples = samples * (1.0 / greatest_abs_sample)
        else:
            self.post_gain_factor = None
        return samples

    def maybe_post_gain(self, samples):
        """If the audio was pre-gained down earlier, post-gain it up to compensate here."""
        if self.post_gain_factor is not None:
            samples = samples * self.post_gain_factor
        return samples

    def apply_lameenc(self, samples: NDArray[np.float32], sample_rate: int):
        try:
            import lameenc
        except ImportError:
            print(
                (
                    "Failed to import the lame encoder. Maybe it is not installed? "
                    "To install the optional lameenc dependency of audiomentations,"
                    " do `pip install audiomentations[extras]` or simply"
                    " `pip install lameenc`"
                ),
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        samples = self.maybe_pre_gain(samples)

        int_samples = convert_float_samples_to_int16(samples).T

        num_channels = 1 if samples.ndim == 1 else samples.shape[0]

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(self.parameters["bitrate"])
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(num_channels)
        encoder.set_quality(7)  # 2 = highest, 7 = fastest
        encoder.silence()

        mp3_data = encoder.encode(int_samples.tobytes())
        mp3_data += encoder.flush()

        # Write a temporary MP3 file that will then be decoded
        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(
            tmp_dir, "tmp_compressed_{}.mp3".format(str(uuid.uuid4())[0:12])
        )
        with open(tmp_file_path, "wb") as f:
            f.write(mp3_data)

        degraded_samples, _ = librosa.load(tmp_file_path, sr=sample_rate, mono=False)

        os.unlink(tmp_file_path)

        degraded_samples = self.maybe_post_gain(degraded_samples)

        if num_channels == 1:
            if int_samples.ndim == 1 and degraded_samples.ndim == 2:
                degraded_samples = degraded_samples.flatten()
            elif int_samples.ndim == 2 and degraded_samples.ndim == 1:
                degraded_samples = degraded_samples.reshape((1, -1))

        return degraded_samples

    def apply_pydub(self, samples: NDArray[np.float32], sample_rate: int):
        try:
            import pydub
        except ImportError:
            print(
                (
                    "Failed to import pydub. Maybe it is not installed? "
                    "To install the optional pydub dependency of audiomentations,"
                    " do `pip install audiomentations[extras]` or simply"
                    " `pip install pydub`"
                ),
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        samples = self.maybe_pre_gain(samples)

        int_samples = convert_float_samples_to_int16(samples).T
        num_channels = 1 if samples.ndim == 1 else samples.shape[0]
        audio_segment = pydub.AudioSegment(
            int_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=int_samples.dtype.itemsize,
            channels=num_channels,
        )

        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(
            tmp_dir, "tmp_compressed_{}.mp3".format(str(uuid.uuid4())[0:12])
        )

        bitrate_string = "{}k".format(self.parameters["bitrate"])
        file_handle = audio_segment.export(tmp_file_path, bitrate=bitrate_string)
        file_handle.close()

        degraded_samples, _ = librosa.load(tmp_file_path, sr=sample_rate, mono=False)

        os.unlink(tmp_file_path)

        degraded_samples = self.maybe_post_gain(degraded_samples)

        if num_channels == 1:
            if int_samples.ndim == 1 and degraded_samples.ndim == 2:
                degraded_samples = degraded_samples.flatten()
            elif int_samples.ndim == 2 and degraded_samples.ndim == 1:
                degraded_samples = degraded_samples.reshape((1, -1))

        return degraded_samples