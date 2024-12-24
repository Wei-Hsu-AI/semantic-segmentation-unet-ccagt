def pixel_accuracy(pred, label):
    """
    計算像素準確率 (Pixel Accuracy, PA)

    - 衡量所有像素中，預測正確的像素比例。
    - 值域範圍 [0, 1]，越接近 1 表示整體準確度越高。
    
    :param pred: 預測的分割結果 [batch_size, H, W]
    :param label: 真實標籤 [batch_size, H, W]
    :return: 像素準確率，表示正確分類的像素占總像素的比例
    """
    correct = (pred == label).sum().item()
    total = label.numel()
    return correct / total


def class_pixel_accuracy(pred, label, num_classes):
    """
    計算每個類別的像素準確率 (Class Pixel Accuracy, cPA)

    - 衡量每個類別中，預測正確的像素占該類別總像素的比例。
    - 可以反映模型在不同類別上的準確度表現。
    
    :param pred: 預測的分割結果 [batch_size, H, W]
    :param label: 真實標籤 [batch_size, H, W]
    :param num_classes: 類別總數
    :return: 每個類別的像素準確率列表
    """
    cpa = []
    for i in range(num_classes):
        pred_i = (pred == i)
        label_i = (label == i)
        correct = (pred_i & label_i).sum().item()
        total = label_i.sum().item()
        cpa.append(correct / total if total > 0 else 0)
    return cpa


def mean_pixel_accuracy(pred, label, num_classes):
    """
    計算平均像素準確率 (Mean Pixel Accuracy, mPA)

    - 所有類別的像素準確率的平均值。
    - 值域範圍 [0, 1]，越接近 1 表示模型在所有類別上的整體準確度越高。
    
    :param pred: 預測的分割結果 [batch_size, H, W]
    :param label: 真實標籤 [batch_size, H, W]
    :param num_classes: 類別總數
    :return: 平均像素準確率
    """
    cpa = class_pixel_accuracy(pred, label, num_classes)
    return sum(cpa) / len(cpa)


def dice_coefficient(pred, label, num_classes):
    """
    計算 Dice 系數 (Dice Coefficient)

    - 衡量兩個集合的重疊程度，常用於不平衡數據集。
    - 值域範圍 [0, 1]，越接近 1 表示分割越精準。
    
    :param pred: 預測的分割結果 [batch_size, H, W]
    :param label: 真實標籤 [batch_size, H, W]
    :param num_classes: 類別總數
    :return: 每個類別的 Dice 系數列表
    """
    dice_scores = []
    for i in range(num_classes):
        pred_i = (pred == i)
        label_i = (label == i)
        intersection = (pred_i & label_i).sum().item()
        union = pred_i.sum().item() + label_i.sum().item()
        dice = 2 * intersection / union if union > 0 else 0
        dice_scores.append(dice)
    return dice_scores


def iou(pred, label, num_classes):
    """
    計算交併比 (Intersection over Union, IoU)

    - 衡量預測與真實區域的重疊程度。
    - 值域範圍 [0, 1]，越接近 1 表示分割越準確。
    
    :param pred: 預測的分割結果 [batch_size, H, W]
    :param label: 真實標籤 [batch_size, H, W]
    :param num_classes: 類別總數
    :return: 每個類別的 IoU 列表
    """
    iou_scores = []
    for i in range(num_classes):
        pred_i = (pred == i)
        label_i = (label == i)
        intersection = (pred_i & label_i).sum().item()
        union = (pred_i | label_i).sum().item()
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    return iou_scores


def mean_iou(pred, label, num_classes):
    """
    計算平均交併比 (Mean Intersection over Union, mIoU)

    - 衡量所有類別 IoU 的平均值。
    - 常用於語義分割評估，範圍 [0, 1]。
    
    :param pred: 預測的分割結果 [batch_size, H, W]
    :param label: 真實標籤 [batch_size, H, W]
    :param num_classes: 類別總數
    :return: 平均 IoU
    """
    iou_scores = iou(pred, label, num_classes)
    return sum(iou_scores) / len(iou_scores)


def compute_mAP(pred, label, thresholds=[0.5 + i * 0.05 for i in range(10)]):
    """
    計算平均精度 (Mean Average Precision, mAP) 在多個 IoU 閾值下的表現。

    - 衡量在多個 IoU 閾值下的平均精度，常用於目標檢測或分割。
    - 值域範圍 [0, 1]，越接近 1 表示性能越好。
    
    :param pred: 預測的分割結果 [batch_size, H, W]
    :param label: 真實標籤 [batch_size, H, W]
    :param thresholds: IoU 閾值的範圍，默認 [0.5, 0.55, ..., 0.95]
    :return: 平均精度
    """
    aps = []
    for t in thresholds:
        intersection = (pred & label).float().sum()
        union = (pred | label).float().sum()
        iou = intersection / union if union > 0 else 0
        aps.append(iou > t)
    return sum(aps) / len(aps)
