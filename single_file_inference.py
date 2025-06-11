import torch
import torchaudio
from transformers import AutoFeatureExtractor
from models import Wav2Vec2BERT
import torch.nn.functional as F
import argparse

def load_audio(audio_path, target_sr=16000):
    """Load and preprocess audio file"""
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform

def main():
    parser = argparse.ArgumentParser(description="Single audio file inference")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--model_path", type=str, default="./ckpt/wave2vec2bert_wavefake.pth", help="Path to the model checkpoint")
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model_name = "facebook/w2v-bert-2.0"
    model = Wav2Vec2BERT(model_name)
    model = model.to(device)
    
    # Load checkpoint
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # Load and preprocess audio
    waveform = load_audio(args.audio_path)
    
    with torch.no_grad():
        # Extract features
        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=16000, 
            return_attention_mask=True, 
            padding_value=0, 
            return_tensors="pt"
        ).to(device)

        # Run model
        outputs = model(**inputs)

        # Softmax over logits to get probabilities
        probs = F.softmax(outputs.logits, dim=-1)

        # Average across all chunks (rows)
        avg_probs = probs.mean(dim=0)

        # Predict class and confidence
        pred_class = torch.argmax(avg_probs).item()
        confidence = avg_probs[pred_class].item()

        # Print results
        result = "FAKE" if pred_class == 0 else "REAL"
        print(f"\nFinal Prediction: {result}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nDetailed averaged probabilities:")
        print(f"FAKE probability: {avg_probs[0].item():.2%}")
        print(f"REAL probability: {avg_probs[1].item():.2%}")



if __name__ == "__main__":
    main()