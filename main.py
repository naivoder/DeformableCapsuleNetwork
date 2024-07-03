import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import warnings

warnings.simplefilter("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeformableCapsuleLayer(nn.Module):
    def __init__(
        self,
        num_capsules,
        num_route_nodes,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        num_iterations=3,
    ):
        super(DeformableCapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_route_nodes, in_channels, out_channels)
            )
            self.spatial_offsets = nn.Parameter(
                torch.randn(num_capsules, kernel_size, kernel_size, in_channels)
            )
        else:
            self.capsules = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=0,
                    )
                    for _ in range(num_capsules)
                ]
            )

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            priors = priors + self.spatial_offsets[:, None, :, :, None]
            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class SE_Routing(nn.Module):
    def __init__(self, in_capsules, out_capsules, num_iterations=3, reduction_ratio=16):
        super(SE_Routing, self).__init__()
        self.num_iterations = num_iterations
        self.reduction_ratio = reduction_ratio

        # Squeeze-and-Excitation layers for dynamic routing
        self.fc1 = nn.Linear(in_capsules, in_capsules // reduction_ratio)
        self.fc2 = nn.Linear(in_capsules // reduction_ratio, out_capsules)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_capsules, _ = x.size()
        x_mean = x.mean(dim=1, keepdim=True)
        x_reduced = F.relu(self.fc1(x_mean))
        routing_weights = self.sigmoid(self.fc2(x_reduced))
        routing_weights = routing_weights.view(batch_size, num_capsules, -1)
        return routing_weights


class DeformableCapsuleNet(nn.Module):
    def __init__(self, num_classes=80, image_size=(3, 128, 128)):
        super(DeformableCapsuleNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = image_size[0]  # RGB or grayscale

        self.conv1 = nn.Conv2d(
            in_channels=self.num_channels, out_channels=256, kernel_size=9, stride=1
        )
        self.primary_capsules = DeformableCapsuleLayer(
            num_capsules=8,
            num_route_nodes=-1,
            in_channels=256,
            out_channels=32,
            kernel_size=9,
            stride=2,
        )
        self.object_instantiation_capsules = DeformableCapsuleLayer(
            num_capsules=num_classes,
            num_route_nodes=2048,
            in_channels=8,
            out_channels=64,  # Increased dimensions to accommodate class-specific variations
            kernel_size=5,
        )
        self.class_presence_capsules = DeformableCapsuleLayer(
            num_capsules=num_classes,
            num_route_nodes=2048,
            in_channels=8,
            out_channels=num_classes,  # Number of classes for class presence
            kernel_size=5,
        )
        self.bbox_capsules = DeformableCapsuleLayer(
            num_capsules=num_classes,
            num_route_nodes=2048,
            in_channels=8,
            out_channels=4,  # Bounding box coordinates (x, y, w, h)
            kernel_size=5,
        )
        self.se_routing = SE_Routing(in_capsules=64, out_capsules=num_classes)

        flattened_input_size = 64 * num_classes
        flattened_output_size = np.prod(image_size)

        self.decoder = nn.Sequential(
            nn.Linear(flattened_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, flattened_output_size),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)

        # Object instantiation capsules
        obj_inst_capsules = self.object_inst_capsules(x).squeeze().transpose(0, 1)

        # Class presence capsules
        class_pres_capsules = self.class_presence_capsules(x).squeeze().transpose(0, 1)

        # Bounding box capsules
        bbox_capsules = self.bbox_capsules(x).squeeze().transpose(0, 1)

        # SE-Routing for dynamic weight assignment
        routing_weights = self.se_routing(obj_inst_capsules)
        obj_inst_capsules = obj_inst_capsules * routing_weights

        classes = (class_pres_capsules**2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = (
                Variable(torch.eye(self.num_classes))
                .cuda()
                .index_select(dim=0, index=max_length_indices.data)
            )

        reconstructions = self.decoder(
            (obj_inst_capsules * y[:, :, None]).view(x.size(0), -1)
        )
        reconstructions = reconstructions.view(
            -1, self.num_channels, self.image_size[0], self.image_size[1]
        )

        return classes, bbox_capsules, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction="sum")
        self.bbox_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, images, targets, classes, bboxes, reconstructions):
        # Extract labels and bounding boxes from targets
        labels = torch.zeros(classes.size()).cuda()
        target_bboxes = torch.zeros(bboxes.size()).cuda()
        for i, target in enumerate(targets):
            for obj in target:
                labels[i, obj["category_id"]] = 1
                target_bboxes[i, obj["category_id"]] = torch.tensor(obj["bbox"]).cuda()

        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        bbox_loss = self.bbox_loss(bboxes, target_bboxes)

        return (margin_loss + 0.0005 * reconstruction_loss + bbox_loss) / images.size(0)


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack([img for img in images], dim=0)
    return images, targets


def train(model, train_loader, optimizer, capsule_loss, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_id, (data, targets) in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            output, bboxes, reconstructions = model(data)
            loss = capsule_loss(data, targets, output, bboxes, reconstructions)
            loss.backward()
            optimizer.step()
            if batch_id % 100 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_id * len(data)}/{len(train_loader.dataset)} ({100. * batch_id / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )


def evaluate(model, test_loader, capsule_loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(DEVICE)
            output, bboxes, reconstructions = model(data)
            test_loss += capsule_loss(
                data, targets, output, bboxes, reconstructions
            ).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack([img for img in images], dim=0)
    return images, targets


if __name__ == "__main__":
    import torchvision.datasets.utils as utils
    from torchvision.datasets import CocoDetection
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import os

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    train_zip = os.path.join(data_dir, "train2017.zip")
    val_zip = os.path.join(data_dir, "val2017.zip")
    ann_zip = os.path.join(data_dir, "annotations_trainval2017.zip")
    train_dir = os.path.join(data_dir, "train2017")
    val_dir = os.path.join(data_dir, "val2017")
    ann_dir = os.path.join(data_dir, "annotations")

    if not os.path.exists(train_dir):
        utils.download_and_extract_archive(
            url="http://images.cocodataset.org/zips/train2017.zip",
            download_root=data_dir,
            extract_root=data_dir,
        )

    if not os.path.exists(val_dir):
        utils.download_and_extract_archive(
            url="http://images.cocodataset.org/zips/val2017.zip",
            download_root=data_dir,
            extract_root=data_dir,
        )

    if not os.path.exists(ann_dir):
        utils.download_and_extract_archive(
            url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            download_root=data_dir,
            extract_root=data_dir,
        )

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    print("\n-> Building Train Dataset")
    train_dataset = CocoDetection(
        root=os.path.join(data_dir, "train2017"),
        annFile=os.path.join(data_dir, "annotations", "instances_train2017.json"),
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate_fn
    )

    print("\n-> Building Test Dataset")
    test_dataset = CocoDetection(
        root=os.path.join(data_dir, "val2017"),
        annFile=os.path.join(data_dir, "annotations", "instances_val2017.json"),
        transform=transform,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=collate_fn
    )

    sample_img, _ = train_dataset[0]
    img_size = list(sample_img.shape)
    print("\nImage Size:", img_size)

    model = DeformableCapsuleNet(num_classes=80, image_size=img_size)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    capsule_loss = CapsuleLoss()

    num_epochs = 10

    print("\n-> Training DeformCaps")
    train(model, train_loader, optimizer, capsule_loss, num_epochs)
    evaluate(model, test_loader, capsule_loss)
