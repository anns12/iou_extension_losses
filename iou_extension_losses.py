import torch
from torch.nn.modules.loss import _Loss

from Rotated_IoU.oriented_iou_loss import cal_diou, cal_giou


class GiouLoss(_Loss):
    """Implementation of GIoU Loss
    """
    def __init__(self, enclosing_type="aligned"):
        super(GiouLoss, self).__init__()
        self.enclosing_type = enclosing_type

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            input (torch.Tensor): (B, N, 5) - 5: (x, y, w, h, theta)
            target (torch.Tensor): (B, N, 5) - 5: (x, y, w, h, theta)

        Returns:
            Tensor: Mean GIoU Loss
        """
        loss = None, None
        loss, _ = cal_giou(input, target, self.enclosing_type)
        return torch.mean(loss)


class DiouLoss(_Loss):
    def __init__(self, enclosing_type="aligned"):
        super(DiouLoss, self).__init__()
        self.enclosing_type = enclosing_type

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            input (torch.Tensor): (B, N, 5) - 5: (x, y, w, h, theta)
            target (torch.Tensor): (B, N, 5) - 5: (x, y, w, h, theta)

        Returns:
            Tensor: Mean DIoU Loss
        """
        loss = None, None
        loss, _ = cal_diou(input, target, self.enclosing_type)
        return torch.mean(loss)