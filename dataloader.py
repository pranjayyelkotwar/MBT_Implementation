import os
import glob
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as Tv
from PIL import Image
import torchaudio.transforms as Ta

class CombinedDataset(Dataset):
    def __init__(self, root_dir, num_frames_per_clip=8, spec_mean=0.0, spec_std=1.0,max_samples=50):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            num_frames_per_clip (int): Number of frames to sample per video.
            spec_mean (float): Mean for spectrogram normalization.
            spec_std (float): Std for spectrogram normalization.
        """
        self.sampling_frequency = 16000  # Fixed sampling frequency for audio

        self.root_dir = root_dir
        self.num_frames_per_clip = num_frames_per_clip
        self.spec_mean = spec_mean
        self.spec_std = spec_std
        self.max_samples = max_samples

        self.visual_transforms = Tv.Compose([
            Tv.ToTensor(),
            Tv.ConvertImageDtype(torch.float32),
            Tv.Resize((224, 224)),  # Resize frames to 224x224 for ViT
            Tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.mel = Ta.MelSpectrogram(
            sample_rate=self.sampling_frequency,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=128,
        )
        self.a2d = Ta.AmplitudeToDB()

        self.data = self._load_data()

    def _load_data(self):
        """
        Parse the dataset directory structure.
        Returns a list of tuples (video_path, audio_path, label).
        """
        data = []
        video_files_dir = os.path.join(self.root_dir, "video")
        audio_files_dir = os.path.join(self.root_dir, "audio")
        class_dirs = os.listdir(video_files_dir)

        for class_idx, class_dir in enumerate(class_dirs):
            video_dir = os.path.join(video_files_dir, class_dir)
            audio_dir = os.path.join(audio_files_dir, class_dir)
            video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
            audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
            
            video_files = video_files[:self.max_samples]
            audio_files = audio_files[:self.max_samples]
            
            for vf, af in zip(video_files, audio_files):
                id = os.path.basename(vf).split('.')[0]
                data.append((vf, af, class_idx, id))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, audio_path, label, id = self.data[idx]

        # Load video frames
        video_frames = self._load_video_as_frames(video_path)
        video_frames = self.transform_video(video_frames)

        # Load audio spectrogram
        audio_spectrogram = self._load_audio(audio_path)
        audio_spectrogram = self.transform_audio(audio_spectrogram)

        return {"video": video_frames, "audio": audio_spectrogram, "label": label, "id": id}

    def _load_video_as_frames(self, video_path):
        """
        Load video as frames, sample a fixed number of frames, and stack them.
        """
        from torchvision.io import read_video
        video, _, _ = read_video(video_path, pts_unit='sec')  # Use pts_unit='sec' to avoid warning
        frame_count = video.shape[0]
        indices = np.linspace(0, frame_count - 1, self.num_frames_per_clip, dtype=int)
        sampled_frames = video[indices]
        return sampled_frames  # Shape: (T, H, W, C)

    def _load_audio(self, audio_path):
        """
        Load audio as waveform, process it into a spectrogram, and normalize.
        """
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        # Use mono audio (left channel if stereo)
        waveform = waveform[0]
        # Resample if necessary
        if sample_rate != self.sampling_frequency:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_frequency)
        # Normalize waveform
        waveform = (waveform - torch.mean(waveform)) / (torch.std(waveform) + 1e-6)
        # Generate mel spectrogram and convert amplitude to decibels
        spectrogram = self.a2d(self.mel(waveform))
        # Normalize spectrogram.
        spectrogram = (spectrogram - self.spec_mean) / self.spec_std
        spectrogram = spectrogram.type(torch.float32)

        # Resize spectrogram to 128x128 for AST input
        spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions: (1, 1, n_mels, time_steps)
        spectrogram = F.interpolate(spectrogram, size=(128, 128), mode='bilinear', align_corners=False)
        spectrogram = spectrogram.squeeze(0)  # Remove batch and channel dimensions: (128, 128)
        return spectrogram

    def transform_video(self, video_frames):
        """
        Apply visual transformations to video frames.
        """
        transformed_frames = []
        for frame in video_frames:
            frame = frame.numpy()  # Convert to numpy array
            frame = (frame * 255).astype(np.uint8)  # Scale to 0-255
            if frame.shape[-1] == 1:  # Handle grayscale frames
                frame = frame.squeeze(-1)  # Remove the last dimension
            frame = Image.fromarray(frame, mode='RGB' if frame.ndim == 3 else 'L')  # Convert to PIL Image
            frame = self.visual_transforms(frame)  # Apply transformations (including resize to 224x224)
            transformed_frames.append(frame)
        return torch.stack(transformed_frames, 0).type(torch.float32)

    def transform_audio(self, spectrogram):
        """
        Apply audio transformations to spectrogram.
        """
        return spectrogram



# # Example usage
# dataset = CombinedDataset(
#     root_dir="final_dataset"
# )

# # Check dataset size
# print(len(dataset))

# # Fetch a sample
# sample = dataset[0]

# # See dimensions
# print(sample["video"].shape)
# print(sample["audio"].shape)
# print(sample["label"])