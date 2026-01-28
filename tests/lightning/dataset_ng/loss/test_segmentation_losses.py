"""Tests for segmentation losses."""

import pytest
import torch

from careamics.lightning.dataset_ng.loss.segmentation_losses import (
    DiceCELoss,
    DiceLoss,
    get_seg_loss,
)


class TestDiceLoss:
    """Test DiceLoss for binary and multi-class segmentation."""

    def test_binary_perfect_prediction(self):
        """Test binary Dice loss with perfect prediction."""
        loss_fn = DiceLoss()

        # Perfect prediction
        inputs = torch.ones(2, 1, 4, 4) * 10  # High logits
        targets = torch.ones(2, 1, 4, 4)

        loss = loss_fn(inputs, targets)

        # Loss should be close to 0 for perfect prediction
        assert loss.item() < 0.01

    def test_binary_worst_prediction(self):
        """Test binary Dice loss with worst prediction."""
        loss_fn = DiceLoss()

        # Worst prediction (inverted)
        inputs = torch.ones(2, 1, 4, 4) * 10  # High logits
        targets = torch.zeros(2, 1, 4, 4)

        loss = loss_fn(inputs, targets)

        # Loss should be close to 1 for worst prediction
        assert loss.item() > 0.9

    def test_binary_shape(self):
        """Test binary Dice loss returns scalar."""
        loss_fn = DiceLoss()

        inputs = torch.randn(2, 1, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 8, 8)).float()

        loss = loss_fn(inputs, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0 and loss.item() <= 1

    def test_multiclass_perfect_prediction(self):
        """Test multi-class Dice loss with perfect prediction."""
        loss_fn = DiceLoss()
        num_classes = 3

        # Create perfect prediction (high logits for correct class)
        inputs = torch.zeros(2, num_classes, 4, 4)
        targets = torch.randint(0, num_classes, (2, 4, 4))

        # Set high logits for target classes
        for b in range(inputs.shape[0]):
            for i in range(inputs.shape[2]):
                for j in range(inputs.shape[3]):
                    inputs[b, targets[b, i, j], i, j] = 10.0

        loss = loss_fn(inputs, targets)

        # Loss should be close to 0 for perfect prediction
        assert loss.item() < 0.01

    def test_multiclass_with_class_indices(self):
        """Test multi-class Dice loss with class indices as targets."""
        loss_fn = DiceLoss()
        num_classes = 4

        inputs = torch.randn(2, num_classes, 8, 8)
        targets = torch.randint(0, num_classes, (2, 8, 8))

        loss = loss_fn(inputs, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0 and loss.item() <= 1

    def test_multiclass_with_onehot_targets(self):
        """Test multi-class Dice loss with one-hot encoded targets."""
        loss_fn = DiceLoss()
        num_classes = 4

        inputs = torch.randn(2, num_classes, 8, 8)
        targets = torch.zeros(2, num_classes, 8, 8)

        # Create one-hot targets
        for b in range(2):
            for i in range(8):
                for j in range(8):
                    c = torch.randint(0, num_classes, (1,)).item()
                    targets[b, c, i, j] = 1.0

        loss = loss_fn(inputs, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0 and loss.item() <= 1

    def test_multiclass_exclude_background(self):
        """Test multi-class Dice loss excluding background."""
        loss_with_bg = DiceLoss(include_background=True)
        loss_without_bg = DiceLoss(include_background=False)
        num_classes = 3

        inputs = torch.randn(2, num_classes, 8, 8)
        targets = torch.randint(0, num_classes, (2, 8, 8))

        loss1 = loss_with_bg(inputs, targets)
        loss2 = loss_without_bg(inputs, targets)

        # Both should be valid losses
        assert loss1.item() >= 0 and loss1.item() <= 1
        assert loss2.item() >= 0 and loss2.item() <= 1
        # They should be different (unless by coincidence)
        # Just check they both run without error

    def test_multiclass_with_weights(self):
        """Test multi-class Dice loss with class weights."""
        num_classes = 3
        weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = DiceLoss(weight=weights)

        inputs = torch.randn(2, num_classes, 8, 8)
        targets = torch.randint(0, num_classes, (2, 8, 8))

        loss = loss_fn(inputs, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_3d_input(self):
        """Test Dice loss with 3D input."""
        loss_fn = DiceLoss()

        # Binary 3D
        inputs = torch.randn(2, 1, 4, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 4, 8, 8)).float()

        loss = loss_fn(inputs, targets)
        assert loss.shape == torch.Size([])

        # Multi-class 3D
        num_classes = 3
        inputs = torch.randn(2, num_classes, 4, 8, 8)
        targets = torch.randint(0, num_classes, (2, 4, 8, 8))

        loss = loss_fn(inputs, targets)
        assert loss.shape == torch.Size([])


class TestDiceCELoss:
    """Test DiceCELoss for binary and multi-class segmentation."""

    def test_binary_combined_loss(self):
        """Test binary Dice+CE loss."""
        loss_fn = DiceCELoss()

        inputs = torch.randn(2, 1, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 8, 8)).float()

        loss = loss_fn(inputs, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_binary_perfect_prediction(self):
        """Test binary Dice+CE loss with perfect prediction."""
        loss_fn = DiceCELoss()

        # Perfect prediction
        inputs = torch.ones(2, 1, 4, 4) * 10  # High logits
        targets = torch.ones(2, 1, 4, 4)

        loss = loss_fn(inputs, targets)

        # Loss should be small for perfect prediction
        assert loss.item() < 0.1

    def test_multiclass_combined_loss(self):
        """Test multi-class Dice+CE loss."""
        loss_fn = DiceCELoss()
        num_classes = 4

        inputs = torch.randn(2, num_classes, 8, 8)
        targets = torch.randint(0, num_classes, (2, 8, 8))

        loss = loss_fn(inputs, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_multiclass_with_onehot_targets(self):
        """Test multi-class Dice+CE loss with one-hot targets."""
        loss_fn = DiceCELoss()
        num_classes = 3

        inputs = torch.randn(2, num_classes, 8, 8)
        targets = torch.zeros(2, num_classes, 8, 8)

        # Create one-hot targets
        for b in range(2):
            for i in range(8):
                for j in range(8):
                    c = torch.randint(0, num_classes, (1,)).item()
                    targets[b, c, i, j] = 1.0

        loss = loss_fn(inputs, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_custom_weights(self):
        """Test Dice+CE loss with custom loss weights."""
        loss_fn = DiceCELoss(ce_weight=2.0, dice_weight=0.5)

        inputs = torch.randn(2, 1, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 8, 8)).float()

        loss = loss_fn(inputs, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_class_weights(self):
        """Test Dice+CE loss with class weights."""
        num_classes = 3
        weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = DiceCELoss(weight=weights)

        inputs = torch.randn(2, num_classes, 8, 8)
        targets = torch.randint(0, num_classes, (2, 8, 8))

        loss = loss_fn(inputs, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_exclude_background(self):
        """Test Dice+CE loss excluding background."""
        loss_fn = DiceCELoss(include_background=False)
        num_classes = 3

        inputs = torch.randn(2, num_classes, 8, 8)
        targets = torch.randint(0, num_classes, (2, 8, 8))

        loss = loss_fn(inputs, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_3d_input(self):
        """Test Dice+CE loss with 3D input."""
        loss_fn = DiceCELoss()

        # Binary 3D
        inputs = torch.randn(2, 1, 4, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 4, 8, 8)).float()

        loss = loss_fn(inputs, targets)
        assert loss.shape == torch.Size([])

        # Multi-class 3D
        num_classes = 3
        inputs = torch.randn(2, num_classes, 4, 8, 8)
        targets = torch.randint(0, num_classes, (2, 4, 8, 8))

        loss = loss_fn(inputs, targets)
        assert loss.shape == torch.Size([])


class TestGetLoss:
    """Test get_loss factory function."""

    def test_get_dice_loss(self):
        """Test getting DiceLoss."""
        loss_fn = get_seg_loss("dice")
        assert isinstance(loss_fn, DiceLoss)

    def test_get_ce_loss(self):
        """Test getting CrossEntropyLoss."""
        loss_fn = get_seg_loss("ce")
        from torch.nn import CrossEntropyLoss

        assert isinstance(loss_fn, CrossEntropyLoss)

    def test_get_dicece_loss(self):
        """Test getting DiceCELoss."""
        loss_fn = get_seg_loss("dicece")
        assert isinstance(loss_fn, DiceCELoss)

    def test_invalid_loss_name(self):
        """Test that invalid loss name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported loss function"):
            get_seg_loss("invalid_loss")
