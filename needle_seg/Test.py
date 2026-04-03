import os
import cv2
import torch
import config_test
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.loader import CustomDataset
from model.Network import UIUNET
from utils.misc import overlay, save_mat
from monai.data import DataLoader, CacheDataset

# 环境设置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# 数据加载
test_dataset = CustomDataset(config_test.TEST_FILENAME)
test_ds = CacheDataset(test_dataset, num_workers=0, cache_rate=0.5)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# 模型初始化
model = UIUNET(3, 1).to(device)
model.load_state_dict(torch.load(config_test.BEST_MODEL, map_location=device))
model.eval()


def normPRED(d):
    """归一化预测结果到 [0, 1]"""
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def clr_2_bw(image, threshold_value):
    """二值化处理"""
    (_, blackAndWhiteImage) = cv2.threshold(image, threshold_value, 1, cv2.THRESH_BINARY)
    return blackAndWhiteImage


def normalize_to_uint8(image):
    """归一化到 uint8 [0, 255]"""
    image_norm = image - np.min(image)
    image_norm = image_norm / (np.max(image_norm) + 1e-8)
    image_uint8 = (image_norm * 255).astype(np.uint8)
    return image_uint8


def prepare_rgb(image):
    """将单通道图像扩展为3通道RGB格式"""
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    return np.repeat(image, 3, axis=-1)


# 测试循环
epoch_iterator = tqdm(test_loader, desc="Testing", dynamic_ncols=True)
i = 0

with torch.no_grad():
    for batch in epoch_iterator:
        i += 1
        img = batch["image"].type(torch.FloatTensor).to(device)
        label = batch["label"].type(torch.FloatTensor)

        # 模型推理
        d0, d1, d2, d3, d4, d5, d6 = model(img)
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # 后处理
        pred = clr_2_bw(pred.squeeze(0).permute(1, 2, 0).cpu().numpy(), threshold_value=0.8)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze()
        label = label.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze()

        # 准备可视化
        label_rgb = prepare_rgb(label)
        pred_rgb = prepare_rgb(pred)

        img_255 = normalize_to_uint8(img)
        label_rgb = (label_rgb * 255).astype(np.uint8)
        pred_rgb = (pred_rgb * 255).astype(np.uint8)

        # 拼接并保存可视化结果
        combined_image = np.concatenate([img_255, label_rgb, pred_rgb], axis=1)
        image = Image.fromarray(combined_image)
        image.save(f"{config_test.TEST_DIR}/vis/combined_image_{i}.png")

        # 保存叠加结果
        overlay_prdctd = overlay(img, pred)
        overlay_lbld = overlay(img, label)
        save_mat(file=overlay_prdctd, i=i, dir=config_test.TEST_DIR, folder_name="overlay_predictions")
        save_mat(file=overlay_lbld, i=i, dir=config_test.TEST_DIR, folder_name="overlay_labels")

        # 保存原始结果
        save_mat(file=img, i=i, dir=config_test.TEST_DIR, folder_name="images")
        save_mat(file=label, i=i, dir=config_test.TEST_DIR, folder_name="labels")
        save_mat(file=pred, i=i, dir=config_test.TEST_DIR, folder_name="predictions")

print("Testing completed")
