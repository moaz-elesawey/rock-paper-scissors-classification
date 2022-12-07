import PIL
import torch
from torchvision import transforms

import config

classes = {
    0: "Rock",
    1: "Paper",
    2: "Scissor",
}

class Prediction:
    def __init__(self, image_data: PIL.Image):
        self.image_data = image_data

        self._transformations = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __apply_transforms(self):
        transformed_image = self._transformations(self.image_data)\
                        .view(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

        return transformed_image


    def predict(self, model: torch.nn.Module):
        transformed_image = self.__apply_transforms()
        prediction_logit = model(transformed_image)
        prediction_probs = torch.softmax(prediction_logit, dim=1)

        self._prediction = classes[torch.argmax(prediction_probs).item()]
        self._confidance = torch.max(prediction_probs).item()

        return self

    @classmethod
    def from_file(cls, filename: str):
        image = PIL.Image.open(filename)

        return cls(image)


    def results(self):
        return f"Prediction: {self._prediction}, Confidance: {self._confidance*100:.2f}%"


