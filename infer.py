import os

import hydra
import torch
from lightning import ModelModule
from pipelines.data.data_module import AVSRDataLoader


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="mediapipe"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        self.dataloader = AVSRDataLoader(self.modality, detector=detector)
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from pipelines.detectors.mediapipe.detector import LandmarksDetector

                self.landmarks_detector = LandmarksDetector()
            if detector == "retinaface":
                from pipelines.detectors.retinaface.detector import LandmarksDetector

                self.landmarks_detector = LandmarksDetector(device="cuda:0")

        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
        )
        self.modelmodule.eval()

    def process_landmarks(self, data_filename, landmarks_filename):
        if self.modality == "audio":
            return None
        if self.modality in ["video", "audiovisual"]:
            landmarks = self.landmarks_detector(data_filename)
            return landmarks

    def forward(self, data_filename, landmarks_filename=None):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(
            data_filename
        ), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        with torch.no_grad():
            transcript = self.modelmodule(data)
        return transcript


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.infer_file)
    print(f"transcript: {transcript}")


if __name__ == "__main__":
    main()
