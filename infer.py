import os

import hydra
import torch
import torchaudio
import torchvision
from lightning import ModelModule
from datamodule.transforms import AudioTransform, VideoTransform


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="mediapipe"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")
        elif self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector(device="cuda:0")
                self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(subset="test")

        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
        )
        self.modelmodule.eval()


    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)
            with torch.no_grad():
                transcript = self.modelmodule(audio)

        if self.modality == "video":
            landmarks = self.landmarks_detector(data_filename)
            video = self.load_video(data_filename)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video)
            with torch.no_grad():
                transcript = self.modelmodule(video)

        return transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.infer_path)
    print(f"transcript: {transcript}")


if __name__ == "__main__":
    main()
