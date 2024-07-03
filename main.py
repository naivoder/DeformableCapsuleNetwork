import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets.utils as utils
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def squash(tensor, dim=-1):
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    scale = (norm**2) / (1 + norm**2)
    return scale * tensor / norm


class DeformConvCapsLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_capsule,
        num_atoms,
        stride=1,
        padding=0,
        routings=3,
    ):
        super(DeformConvCapsLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.stride = stride
        self.padding = padding
        self.routings = routings

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_capsule * num_atoms,
            kernel_size,
            stride,
            padding,
        )
        self.offsets = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size, kernel_size, stride, padding
        )
        self.biases = nn.Parameter(torch.zeros(1, out_channels, num_capsule, num_atoms))

    def forward(self, x):
        batch_size = x.size(0)
        offsets = self.offsets(x)
        offsets = offsets.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        x = self.conv(x)
        votes = x.view(
            batch_size, self.out_channels, self.num_capsule, self.num_atoms, -1
        )
        votes = votes.permute(0, 1, 4, 2, 3).contiguous()

        logits = torch.zeros(*votes.size()).to(x.device)
        for i in range(self.routings):
            route = F.softmax(logits, dim=3)
            preactivate = torch.sum(route * votes, dim=2) + self.biases
            activation = squash(preactivate)
            act_replicated = activation.unsqueeze(2).expand_as(votes)
            logits += torch.sum(votes * act_replicated, dim=-1, keepdim=True)
        return activation


class SplitCaps(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        num_capsule,
        num_atoms,
        kernel_size,
        stride=1,
        padding=0,
        routings=3,
    ):
        super(SplitCaps, self).__init__()
        self.num_classes = num_classes
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms

        self.instantiation_caps = DeformConvCapsLayer(
            in_channels,
            num_capsule,
            kernel_size,
            num_capsule,
            num_atoms,
            stride,
            padding,
            routings,
        )
        self.class_presence_caps = DeformConvCapsLayer(
            in_channels,
            num_classes,
            kernel_size,
            num_capsule,
            num_atoms,
            stride,
            padding,
            routings,
        )

    def forward(self, x):
        instantiation = self.instantiation_caps(x)
        class_presence = self.class_presence_caps(x)
        return instantiation, class_presence


class SERouting(nn.Module):
    def __init__(self, reduction_ratio=4):
        super(SERouting, self).__init__()
        self.reduction_ratio = reduction_ratio

    def forward(self, instantiation, class_presence):
        batch_size, num_capsule, _, num_atoms = instantiation.size()
        combined = torch.cat([instantiation, class_presence], dim=-1)
        excitation = F.relu(
            nn.Linear(num_atoms * 2, num_atoms // self.reduction_ratio)(combined)
        )
        excitation = torch.sigmoid(
            nn.Linear(num_atoms // self.reduction_ratio, num_atoms * 2)(excitation)
        )
        routed_caps = excitation * combined
        return routed_caps


class DeformCapsNet(nn.Module):
    def __init__(self, num_classes, image_size):
        super(DeformCapsNet, self).__init__()
        in_channels = image_size[0]
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.split_caps = SplitCaps(
            128,
            num_classes,
            num_capsule=8,
            num_atoms=16,
            kernel_size=3,
            stride=1,
            padding=1,
            routings=3,
        )
        self.se_routing = SERouting()

    def forward(self, x):
        features = self.backbone(x)
        instantiation, class_presence = self.split_caps(features)
        routed_caps = self.se_routing(instantiation, class_presence)
        return routed_caps


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction="sum")
        self.bbox_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, images, targets, classes, bboxes, reconstructions):
        labels = torch.zeros(classes.size()).to(DEVICE)
        target_bboxes = torch.zeros(bboxes.size()).to(DEVICE)
        for i, target in enumerate(targets):
            for obj in target:
                labels[i, obj["category_id"]] = 1
                target_bboxes[i, obj["category_id"]] = torch.tensor(obj["bbox"]).to(
                    DEVICE
                )

        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        margin_loss = margin_loss.sum()

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


if __name__ == "__main__":
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(os.path.join(data_dir, "train2017")):
        utils.download_and_extract_archive(
            url="http://images.cocodataset.org/zips/train2017.zip",
            download_root=data_dir,
            extract_root=data_dir,
        )

    if not os.path.exists(os.path.join(data_dir, "val2017")):
        utils.download_and_extract_archive(
            url="http://images.cocodataset.org/zips/val2017.zip",
            download_root=data_dir,
            extract_root=data_dir,
        )

    if not os.path.exists(os.path.join(data_dir, "annotations")):
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

    model = DeformCapsNet(num_classes=80, image_size=img_size)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    capsule_loss = CapsuleLoss()

    num_epochs = 10

    print("\n-> Training DeformCaps")
    train(model, train_loader, optimizer, capsule_loss, num_epochs)
    evaluate(model, test_loader, capsule_loss)
