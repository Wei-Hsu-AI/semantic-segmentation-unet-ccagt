import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def visualize_metrics(all_metrics, num_classes):
    """
    視覺化評估指標。

    - 每個類別的指標（如 cPA, Dice, IoU）分別繪製多張圖。
    - 整體指標（如 PA, mPA, mIoU, mAP）合併在一張圖中，並對其進行平均處理。

    :param all_metrics: 字典，包含指標名稱和對應數值。
    :param num_classes: 類別數量，用於生成每個類別的指標圖。
    """
    # 整體指標：計算平均值後可視化
    avg_metrics = {key: sum(values) / len(values) for key, values in {
        'PA': all_metrics['PA'],
        'mPA': all_metrics['mPA'],
        'mIoU': all_metrics['mIoU'],
        'mAP': all_metrics['mAP']
    }.items()}

    # 繪製整體指標的柱狀圖
    plt.figure(figsize=(10, 6))
    min_val = min(avg_metrics.values())
    max_val = max(avg_metrics.values())
    margin = 0.05
    plt.bar(avg_metrics.keys(), avg_metrics.values(), color='blue')
    plt.ylim(min_val - margin, max_val + margin)
    plt.title("Overall Metrics (Averaged)", fontsize=16)
    plt.xlabel("Metrics", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(avg_metrics.values()):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
    plt.show()

    # 每類別指標
    per_class_metrics = {
        'cPA': all_metrics['cPA'],
        'Dice': all_metrics['Dice'],
        'IoU': all_metrics['IoU']
    }

    # 繪製每類別的指標圖
    for metric_name, metric_values in per_class_metrics.items():
        avg_class_metrics = [sum(class_values) / len(class_values) for class_values in zip(*metric_values)]
        plt.figure(figsize=(10, 6))
        plt.bar(range(num_classes), avg_class_metrics, color='green')
        plt.title(f"Average {metric_name} per Class", fontsize=16)
        plt.xlabel("Class", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.xticks(range(num_classes), [f"Class {i}" for i in range(num_classes)], fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(avg_class_metrics):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
        plt.show()

def plot_training_validation_loss(train_losses, val_losses):
    """
    繪製訓練和驗證的損失曲線。
    
    :param train_losses: list:訓練損失值的列表
    :param val_losses: list: 驗證損失值的列表
    """
    plt.figure(figsize=(8, 6))
    plt.title('Training and Validation Loss')
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)
    plt.show()

def contours_generate(input_img,fusion_img):
    """
    生成影像輪廓並將其疊加到輸入的融合影像 (fusion_img) 上。
    """
    input_img = np.float32(input_img)
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    img_gray *= 255
    colors = [(255,50,0),(131,60,11),(0,255,0),(0,0,255),(255,0,255),(255,0,0),(0,0,128)]
    for threshhold in range(1,8):
        ret, thresh = cv2.threshold(np.uint8(img_gray),(threshhold*36-1), 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, 3, 2)
        if contours:
            if threshhold == 1:
                hull = cv2.drawContours(fusion_img, contours, -1, colors[threshhold-2], 6)
            else:
                hull = cv2.drawContours(hull, contours, -1, colors[threshhold-2], 6)
        else :
            hull = fusion_img
    return hull

def vs_generate(input_mask, gen_mask, fusion):
    """
    生成「輸入真值遮罩 (input_mask) 與生成遮罩 (gen_mask)」的比較圖像，顯示錯誤與正確的像素區域。
    """
    err_space = np.float64(np.logical_xor(input_mask, gen_mask))
    corr_space = np.logical_and(input_mask, gen_mask)
    R,G,B = cv2.split(err_space)
    R[R==0] = 0.18
    G[G==0] = 0.46
    G[G>0.47] = 0
    B[B==0] = 0.71
    B[B>0.72] = 0
    err_space =cv2.merge([R,G,B])
    
    err_space *= np.float64(np.logical_not(corr_space))
    corr_space = np.float64(corr_space)
    corr_space *= fusion
    err_space += corr_space
    return err_space

def compress_channel(input_batch_img, threshold):
    """
    將多通道的影像壓縮到單一通道，基於像素值閾值來標記通道。
    """
    single_img = torch.zeros(1, input_batch_img.size(2), input_batch_img.size(3))
    output_batch_img = torch.zeros(input_batch_img.size(0), 1,
                                   input_batch_img.size(2), input_batch_img.size(3))
    for idx,n in enumerate(input_batch_img):
        for ch,img in enumerate(n):
            single_img[0][ img > threshold ] = ch
        output_batch_img[idx] = single_img
    return output_batch_img

def show_images(input_imgs, img_size, input_masks: None, gen_masks= None,
                nrow=5, ncol=1, show: bool = True, save: bool = False, path ="", mode: bool =False):
    # compare and show n*m images from generator in one figure and optionally save it
    if input_imgs.shape[0] < nrow:
        nrow = input_imgs.shape[0]
    figsize = (nrow*3+2,9)
    count = 311
    img_label = ["input\nimages", "input\nmask"]
    inputs = [input_imgs, input_masks]
    offset = -0.1

    if mode == True and gen_masks == None:
        print("Input ERROR! Expected [gen_mask] but got [None].")
        return None
        
    elif mode == True and gen_masks != None:
        figsize = (nrow*3+2,18)
        count = 611
        img_label.append("generated\nmask")
        inputs.append(gen_masks)

    plt.figure(figsize=figsize)
    for imgs, label in zip([imgs for imgs in inputs if input_masks is not None], img_label):
        imgs = imgs[:nrow * ncol]
        imgs = imgs.view(imgs.size(0), imgs.size(1), img_size[0], img_size[1])
        ax = plt.subplot(count)
        ax.set_title(label ,x=offset, y=0.35)
        img = np.transpose(make_grid(imgs, nrow=nrow, padding=2, normalize=True).cpu(), (1, 2, 0))
        img = np.float32(np.array(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img.shape[1]*2, int(img.shape[0]*1.5)), interpolation=cv2.INTER_AREA)
        plt.axis("off")
        plt.imshow(img)

        if label == "input\nmask":
            input_mask = img
        if label == "generated\nmask":
            gen_mask = img

        if label == "input\nimages":
            origin_img = img/3+0.6
        else :
            count+=1
            ax = plt.subplot(count)
            name = label.split("\n")[0] + "\nfusion"
            ax.set_title(name,x=offset, y=0.35)
            fusion = origin_img.copy()
            contours_generate(img,fusion)
            plt.axis("off")
            plt.imshow(fusion)
        
        if label == "generated\nmask":
            count+=1
            ax = plt.subplot(count)
            name = "ground truth\nvs\ngenerated"
            ax.set_title(name,x=offset, y=0.35)
            fusion = origin_img.copy()
            vs = vs_generate(input_mask, gen_mask, fusion)
            #print(vs,)
            plt.axis("off")
            plt.imshow(vs)
        count+=1
    if save:
        plt.savefig('./show.png')
    if show:
        plt.show()

def get_all_unique_values(dataset):
    """
    獲取數據集中所有 mask 的唯一值
    :param dataset: 數據集對象，返回 (image, mask)
    :return: 一個集合，包含所有 mask 的唯一值
    """
    unique_values = set()
    for _, mask in dataset:
        # 確保 mask 是 PyTorch 張量
        if not isinstance(mask, torch.Tensor):
            raise ValueError("Mask should be a PyTorch tensor")
        # 提取 mask 的唯一值並更新到集合中
        unique_values.update(mask.unique().tolist())
    return unique_values
