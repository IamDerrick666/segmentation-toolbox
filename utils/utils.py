import os
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def basic_transform(height, width):
    data_transform = A.Compose([A.Resize(height=height, width=width),
                                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                                ToTensorV2()])

    return data_transform


def train_transform(height, width):
    data_transform = A.Compose([A.Resize(height=height, width=width),
                                A.VerticalFlip(always_apply=False, p=0.5),
                                A.HorizontalFlip(always_apply=False, p=0.5),
                                A.RandomRotate90(always_apply=False, p=0.5),
                                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                                ToTensorV2()])

    return data_transform


def get_loaders(dataset, batch_size, num_workers, pin_memory, drop_last):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader


def compute_validation_loss(data_loader, model, loss_fn, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device=device)
            targets = targets.float().unsqueeze(1).to(device=device)

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()

    return val_loss / len(data_loader)


def train_fn(train_loader, val_loader, model, optimizer, loss_fn, scaler, epoch, amp, scheduler, dlr, device):
    print(f"---Epoch:{epoch}---")
    loop = tqdm(train_loader, total=len(train_loader), mininterval=0.1, miniters=1)
    train_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        # forward
        if amp:
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    if dlr:
        # 在每个 epoch 结束后计算当前的验证集损失
        current_loss = compute_validation_loss(val_loader, model, loss_fn, device)

        # 调度器更新学习率
        scheduler.step(current_loss)

    return train_loss / len(train_loader)


def val_fn(val_loader, model, device):
    num_correct = 0
    num_pixels = 0
    iou = 0
    dice = 0
    sensitivity = 0
    precision = 0
    total_tn = 0
    total_fp = 0
    auc_targets = []
    auc_scores = []

    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            scores = torch.sigmoid(model(x))
            preds = (scores > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            tp = (preds * y).sum()
            tn = ((1 - preds) * (1 - y)).sum()
            fp = (preds - preds * y).sum()
            fn = (y - preds * y).sum()
            iou += tp / ((tp + fp + fn) + 1e-8)
            dice += (2 * tp) / ((2 * tp + fp + fn) + 1e-8)
            sensitivity += tp / ((tp + fn) + 1e-8)
            precision += tp / ((tp + fp) + 1e-8)
            total_tn += tn.item()
            total_fp += fp.item()

            auc_targets.append(y.detach().cpu().numpy().ravel())
            auc_scores.append(scores.detach().cpu().numpy().ravel())

    iou = (iou / len(val_loader)).cpu().numpy()
    dice = (dice / len(val_loader)).cpu().numpy()
    sensitivity = (sensitivity / len(val_loader)).cpu().numpy()
    specificity = total_tn / (total_tn + total_fp + 1e-8)
    precision = (precision / len(val_loader)).cpu().numpy()
    accuracy = (num_correct / num_pixels).cpu().numpy()
    auc_targets = np.concatenate(auc_targets)
    auc_scores = np.concatenate(auc_scores)
    if np.unique(auc_targets).size < 2:
        auc = float("nan")
    else:
        auc = metrics.roc_auc_score(auc_targets, auc_scores)

    print(f"IoU: {iou}")
    print(f"Dice: {dice}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Precision: {precision}")
    print(f"AUC: {auc}")
    print(f"Accuracy: {accuracy}")

    model.train()

    return iou, dice, sensitivity, specificity, precision, auc, accuracy


def create_result_dir(model_name, dataset):
    primary_path = "outputs"
    folder_name = f"{model_name}_{dataset}"
    result_path = os.path.join(primary_path, folder_name)

    os.makedirs(primary_path, exist_ok=True)

    if os.path.exists(result_path):
        print("\033[31moutputs文件夹内当前项目已存在，请仔细检查命名，若需要进行覆盖，请手动删除原项目。\033[0m")
        print(
            "\033[31mThe current project already exists in the outputs folder. Please double-check the naming and manually delete the original project if you need to overwrite it.\033[0m")
        quit()

    os.mkdir(result_path)

    for i in range(5):
        image_folder = f"saved_images{i}"
        os.mkdir(os.path.join(result_path, image_folder))

    return result_path


def save_predictions_as_imgs(val_loader, model, folder="saved_images/", device="cuda"):
    model.eval()

    for idx, (x, y) in enumerate(val_loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/lable_{idx}.png")
        torchvision.utils.save_image(x, f"{folder}/image_{idx}.png")

    model.train()


def get_result(result_list, model_name, dataset, path):
    column = ['IoU', 'Dice', 'Sensitivity', 'Specificity', 'Precision', 'AUC', 'Accuracy', 'Train_loss',
              'Val_loss']
    log = pd.DataFrame(columns=column, data=result_list)
    log.to_csv(path + f"/{model_name}_{dataset}.csv")
