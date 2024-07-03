import unittest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from main import *


# Dummy data and model for testing
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128), antialias=True),
                # transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(3, 128, 128)
        target = [{"category_id": 1, "bbox": [10, 10, 50, 50]}]
        if self.transform:
            image = self.transform(image)
        return image, target


class TestDeformCapsNet(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeformCapsNet(num_classes=80, image_size=[3, 128, 128]).to(
            self.device
        )
        self.loss_fn = CapsuleLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.dataset = DummyDataset()
        self.dataloader = DataLoader(
            self.dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
        )

    def test_model_forward(self):
        data_iter = iter(self.dataloader)
        images, targets = next(data_iter)
        images = images.to(self.device)
        output = self.model(images)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.size(0), images.size(0))

    def test_capsule_loss(self):
        data_iter = iter(self.dataloader)
        images, targets = next(data_iter)
        images = images.to(self.device)
        output, bboxes, reconstructions = self.model(images)
        loss = self.loss_fn(images, targets, output, bboxes, reconstructions)
        self.assertIsInstance(loss, torch.Tensor)

    def test_train_step(self):
        data_iter = iter(self.dataloader)
        images, targets = next(data_iter)
        images = images.to(self.device)
        self.optimizer.zero_grad()
        output, bboxes, reconstructions = self.model(images)
        loss = self.loss_fn(images, targets, output, bboxes, reconstructions)
        loss.backward()
        self.optimizer.step()
        self.assertIsInstance(loss.item(), float)

    def test_collate_fn(self):
        data_iter = iter(self.dataloader)
        images, targets = next(data_iter)
        self.assertIsInstance(images, torch.Tensor)
        self.assertIsInstance(targets, tuple)
        self.assertEqual(len(images), 2)

    def test_data_pipeline(self):
        for images, targets in self.dataloader:
            self.assertIsInstance(images, torch.Tensor)
            self.assertIsInstance(targets, tuple)
            self.assertEqual(images.size(0), 2)
            break


if __name__ == "__main__":
    unittest.main()
