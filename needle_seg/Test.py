import os
import cv2
import torch
import config_test
from utils.loader import CustomDataset
from monai.data import ( DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch,)
from tqdm import tqdm
from model.Network import UIUNET
from utils.misc import overlay, save_mat
import numpy as np
from PIL import Image

test_dataset = CustomDataset(config_test.TEST_FILENAME)
test_ds = CacheDataset(test_dataset, num_workers=0, cache_rate=0.5)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_iterator = tqdm(test_loader, desc="Testing (X / X Steps) (loss=X.X)", dynamic_ncols=True)
model = UIUNET(3, 1).to(device)
torch.backends.cudnn.benchmark = True
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def dice_score_binary(pred, gt, eps=1e-8):
    """
    pred, gt: numpy arrays, values in {0,1} or {0,255}
    returns: dice in [0,1]
    """
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 0).astype(np.uint8)

    inter = np.sum(pred * gt)
    s_pred = np.sum(pred)
    s_gt   = np.sum(gt)

    # both empty -> perfect match
    if s_pred == 0 and s_gt == 0:
        return 1.0

    return (2.0 * inter + eps) / (s_pred + s_gt + eps)


def clr_2_bw(image, threshold_value):
    (_, blackAndWhiteImage) = cv2.threshold(image, threshold_value, 1, cv2.THRESH_BINARY)
    return blackAndWhiteImage
def normalize_to_uint8(image_log):
    image_norm = image_log - np.min(image_log)
    image_norm = image_norm / (np.max(image_norm) + 1e-8)
    image_uint8 = (image_norm * 255).astype(np.uint8)
    return image_uint8

import numpy as np

def to_binary_mask(x, thr=0.5):
    """x can be float/probability mask or uint8; output {0,1} uint8."""
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = (x >= thr).astype(np.uint8)
    else:
        x = (x > 0).astype(np.uint8)
    return x

def mask_to_points(mask):
    """Return Nx2 array of (row, col) for nonzero pixels."""
    pts = np.argwhere(mask > 0)
    # pts is (row, col)
    return pts.astype(np.float32)

def fit_line_pca(points_rc):
    """
    Fit a 2D line to points using PCA.
    points_rc: Nx2 (row, col)
    Returns: (p0_rc, v_rc) where p0 is centroid, v is unit direction.
    """
    if points_rc.shape[0] < 2:
        return None, None
    p0 = points_rc.mean(axis=0)
    X = points_rc - p0
    # PCA via SVD
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]  # principal direction
    n = np.linalg.norm(v) + 1e-12
    v = v / n
    return p0, v

def point_line_distance(point_rc, p0_rc, v_rc):
    """
    Perpendicular distance from point to line (p0 + t v).
    point_rc: (2,)
    """
    w = point_rc - p0_rc
    # In 2D, distance = ||w - (w·v)v||
    proj = np.dot(w, v_rc) * v_rc
    perp = w - proj
    return float(np.linalg.norm(perp))

def line_length_from_points(points_rc, p0_rc, v_rc):
    """
    Estimate "needle length" as extent of projections of points onto v.
    """
    if points_rc.shape[0] < 2:
        return 0.0
    t = (points_rc - p0_rc) @ v_rc  # projections (Nx,)
    return float(t.max() - t.min())

def modified_hausdorff_distance(A_rc, B_rc, max_points=5000):
    """
    Modified Hausdorff Distance (MHD):
      MHD(A,B) = max( mean_a min_b ||a-b|| , mean_b min_a ||b-a|| )
    Returned in pixels.
    To keep it fast, optionally subsample points if too many.
    """
    if A_rc.shape[0] == 0 and B_rc.shape[0] == 0:
        return 0.0
    if A_rc.shape[0] == 0 or B_rc.shape[0] == 0:
        return float("inf")

    # Subsample for speed if needed
    if A_rc.shape[0] > max_points:
        idx = np.random.choice(A_rc.shape[0], max_points, replace=False)
        A = A_rc[idx]
    else:
        A = A_rc

    if B_rc.shape[0] > max_points:
        idx = np.random.choice(B_rc.shape[0], max_points, replace=False)
        B = B_rc[idx]
    else:
        B = B_rc

    # Pairwise distances using broadcasting: (NA, NB)
    # dist(i,j) = ||A[i]-B[j]||
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    a_to_b = d.min(axis=1).mean()
    b_to_a = d.min(axis=0).mean()
    return float(max(a_to_b, b_to_a))

def needle_metrics(pred_mask, gt_mask,
                   success_dist_px=2.0,
                   min_intersection_px=10):
    """
    Compute (MHD, success, TE, length_ratio).
    success criterion (practical):
      - predicted & GT overlap has at least min_intersection_px pixels
      - AND predicted line passes near GT points (mean distance <= success_dist_px)
    TE and length_ratio are only meaningful if success=True; otherwise return np.nan for them.
    """
    pred = to_binary_mask(pred_mask, thr=0.5)
    gt   = to_binary_mask(gt_mask, thr=0.5)

    # Point sets (all foreground pixels)
    P = mask_to_points(pred)
    G = mask_to_points(gt)

    # MHD on foreground point sets
    mhd = modified_hausdorff_distance(P, G)

    # If either mask is empty: fail localization
    if P.shape[0] < 2 or G.shape[0] < 2:
        return mhd, False, np.nan, np.nan

    # Fit lines
    p0_p, v_p = fit_line_pca(P)
    p0_g, v_g = fit_line_pca(G)
    if p0_p is None or p0_g is None:
        return mhd, False, np.nan, np.nan

    # Intersection pixels
    inter = int(np.sum((pred > 0) & (gt > 0)))

    # "Predicted line passes through GT" proxy:
    # mean distance of GT points to predicted line
    # (you can also use min distance if you want it less strict)
    # For speed, subsample GT points if huge
    Gs = G
    if Gs.shape[0] > 8000:
        idx = np.random.choice(Gs.shape[0], 8000, replace=False)
        Gs = Gs[idx]
    mean_dist_gt_to_predline = np.mean([point_line_distance(g, p0_p, v_p) for g in Gs])

    success = (inter >= min_intersection_px) and (mean_dist_gt_to_predline <= success_dist_px)

    # Targeting error TE: difference in distance to image center
    H, W = gt.shape[:2]
    center = np.array([H / 2.0, W / 2.0], dtype=np.float32)
    d_true = point_line_distance(center, p0_g, v_g)
    d_pred = point_line_distance(center, p0_p, v_p)
    te = abs(d_true - d_pred)

    # Needle length ratio
    len_pred = line_length_from_points(P, p0_p, v_p)
    len_true = line_length_from_points(G, p0_g, v_g)
    length_ratio = (len_pred / (len_true + 1e-12))

    if not success:
        return mhd, False, np.nan, np.nan

    return mhd, True, te, length_ratio

model.load_state_dict(torch.load(config_test.BEST_MODEL, map_location=device))
model.eval()
i = 0
mhd_list = []
te_list = []
len_ratio_list = []
success_list = []
with torch.no_grad():
    for batch in epoch_iterator:
        i +=1
        img = batch["image"].type(torch.FloatTensor)
        label = batch["label"].type(torch.FloatTensor)
        d0, d1, d2, d3, d4, d5, d6 = model(img.cuda())
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        pred = clr_2_bw(pred.permute(1, 2, 0).cpu().numpy(), threshold_value=0.8)
        img = img.squeeze(0).permute(1,2,0).cpu().numpy().squeeze()
        label = label.squeeze(0).permute(1,2,0).cpu().numpy().squeeze()
        label_rgb = np.expand_dims(label, axis=-1) 
        label_rgb = np.repeat(label_rgb, 3, axis=-1)  # 变为 (256, 256, 3)
        pred_rgb = np.expand_dims(pred, axis=-1)  # 扩展成 (256, 256, 1)
        pred_rgb = np.repeat(pred_rgb, 3, axis=-1)  # 变为 (256, 256, 3)
        img_255 = normalize_to_uint8(img)
        label_rgb = (label_rgb * 255).astype(np.uint8)
        pred_rgb = (pred_rgb * 255).astype(np.uint8)
        combined_image = np.concatenate([img_255, label_rgb, pred_rgb], axis=1)
        # combined_image = normalize_to_uint8(combined_image)
        # combined_image = (combined_image * 255).astype(np.uint8)  # 确保像素值在[0, 255]范围内
        image = Image.fromarray(combined_image)
        image.save(f"{config_test.TEST_DIR}/vis/combined_image_{i}.png")
        overlay_prdctd = overlay(img,pred)
        overlay_lbld = overlay(img, label)
        save_mat(file=overlay_prdctd , i = i, dir = config_test.TEST_DIR, folder_name = "overlay_predictions")
        save_mat(file=overlay_lbld, i=i, dir=config_test.TEST_DIR, folder_name="overlay_labels")
        save_mat(file=img, i=i, dir=config_test.TEST_DIR, folder_name="images")
        save_mat(file=label, i=i, dir=config_test.TEST_DIR, folder_name="labels")
        save_mat(file=pred, i=i, dir=config_test.TEST_DIR, folder_name="predictions")
    print("Testing completed")



