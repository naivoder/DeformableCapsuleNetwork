import unittest
import torch
from torch import nn
import numpy as np
from main import *


class TestDeformableCapsuleNet(unittest.TestCase):
    def setUp(self):
        self.num_classes = 80
        self.image_size = (3, 128, 128)
        self.model = DeformableCapsuleNet(
            num_classes=self.num_classes, image_size=self.image_size
        )
        self.model.cuda()
        self.capsule_loss = CapsuleLoss()
        self.batch_size = 4

    def test_model_architecture(self):
        # Check if the model's architecture is correct
        self.assertIsInstance(self.model.conv1, nn.Conv2d)
        self.assertIsInstance(self.model.primary_capsules, DeformableCapsuleLayer)
        self.assertIsInstance(
            self.model.object_instantiation_capsules, DeformableCapsuleLayer
        )
        self.assertIsInstance(
            self.model.class_presence_capsules, DeformableCapsuleLayer
        )
        self.assertIsInstance(self.model.se_routing, SE_Routing)
        self.assertEqual(
            self.model.class_presence_capsules.num_capsules, self.num_classes
        )
        self.assertEqual(
            self.model.class_presence_capsules.out_channels, self.num_classes
        )

    def test_forward_pass(self):
        # Test the forward pass of the model
        input_tensor = torch.randn(self.batch_size, *self.image_size).cuda()
        classes, reconstructions = self.model(input_tensor)

        self.assertEqual(classes.size(0), self.batch_size)
        self.assertEqual(classes.size(1), self.num_classes)
        self.assertEqual(reconstructions.size(0), self.batch_size)
        self.assertEqual(
            reconstructions.size(1), np.prod(self.image_size)
        )  # Flattened reconstruction size

    def test_loss_computation(self):
        # Test the loss computation
        input_tensor = torch.randn(self.batch_size, *self.image_size).cuda()
        labels = (
            torch.eye(self.num_classes)
            .index_select(
                dim=0, index=torch.randint(0, self.num_classes, (self.batch_size,))
            )
            .cuda()
        )
        classes, reconstructions = self.model(input_tensor, labels)

        loss = self.capsule_loss(input_tensor, labels, classes, reconstructions)
        self.assertIsInstance(loss.item(), float)


if __name__ == "__main__":
    unittest.main()
