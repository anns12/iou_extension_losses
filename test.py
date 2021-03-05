import numpy as np
import torch 
from iou_losses import GiouLoss, DiouLoss


def main():
    gt = [0, 0, 2, 3, np.pi/6]
    pred = [1, 1, 4, 4, -np.pi/4]

    assert torch.cuda.is_available()

    batch_gt = np.array(gt, dtype=np.float32)               
    batch_pred = np.array(pred, dtype=np.float32)           

    # Convert from numpy to Tensor type
    tensor_gt = torch.from_numpy(batch_gt).unsqueeze(0).unsqueeze(0)        # (B, N, 5), with B=1 and N=1
    tensor_pred = torch.from_numpy(batch_pred).unsqueeze(0).unsqueeze(0)    # (B, N, 5), with B=1 and N=1

    # Only work on GPU
    tensor_gt = tensor_gt.to("cuda")
    tensor_pred = tensor_pred.to("cuda")

    giou_loss = GiouLoss()
    diou_loss = DiouLoss()

    print("GIoU Loss Result: ", giou_loss(tensor_gt, tensor_pred))
    print("GIoU Loss Result: ", diou_loss(tensor_gt, tensor_pred))



if __name__ == '__main__':
    main()