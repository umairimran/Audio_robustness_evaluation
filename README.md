# Audio_robustness_evaluation

## Enviroment

- ``conda env export > environment.yml``
- ``conda activate audiopert``

## Datasets

Please download the following datasets and extract them in ``/data/``  or change the database path correspondingly in the code.

- [Wavefake](https://zenodo.org/records/5642694) 
- [LJ-Speech](https://keithito.com/LJ-Speech-Dataset/)

The structure should look like

```
data
├── LJSpeech-1.1
│ 	├── wavs
│	├── metadata.csv
│ 	└── README
├── wavefake
│ 	├── ljspeech_full_band_melgan
│	├── ljspeech_hifiGAN
│	├── ...
│ 	└── ljspeech_waveglow
```

## Audio Perturber

Multiple audio perturbations can be applied with the ``AudioPerturbation`` class provided in ``perturber.py``. Example usage looks like the following:

```python
import torchaudio
from perturber import AudioPerturbation

perturber = AudioPerturbation(sample_rate=16000)
waveform, sample_rate = torchaudio.load("input.wav")
# Apply gaussian noise
noisy_waveform = perturber.gaussian_noise(waveform, snr_db=10)
# Apply MP3 compression
mp3_waveform = perturber.mp3(waveform, bitrate=8)
```

## Evaluation

To evaluate foundation models, run ``main_fm.py``. For example, to evaluate robustness to different perturbation scales of Gaussian noise:

```
python main_fm.py --eval \
--model hubert \
--eval_ckpt $PATH_TO_MODEL_CHECKPOINT$ \
--pert_method gaussian_noise \ 
```



To evaluate traditional deep learning detectors, run main_tm.py

```
python main_tm.py --eval \
--config ./config/AASIST.conf \
--pert_method gaussian_noise
```
