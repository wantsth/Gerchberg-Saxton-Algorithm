import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 均一性评价函数
def check_uniformity(u_int, target):
    u_int = u_int / np.max(u_int)  # 归一化
    maxi = np.max(u_int[target > 0.99])
    mini = np.min(u_int[target > 0.99])
    uniformity = 1 - (maxi - mini) / (maxi + mini)
    return uniformity

# 强度归一化
def normalization(image):
    maxi = np.max(image)
    mini = np.min(image)
    if maxi == mini:
        return np.zeros_like(image, dtype=np.float64)
    result = (image - mini) / (maxi - mini)
    return result

# 相位 → 全息图
def hologram(phase):
    phase = np.where(phase < 0, phase + 2 * np.pi, phase)
    p_maxi = np.max(phase)
    p_mini = np.min(phase)
    holo = (phase - p_mini) / (p_maxi - p_mini) * 255
    holo = holo.astype("uint8")
    return holo

# 重建图象格式转换
def reconstruct(norm_int):
    result = norm_int * 255
    result = result.astype("uint8")
    return result

# Gerchberg-Saxton 算法
def GS(target, channel, iteration=50):
    height, width = target.shape
    # 目标必须转换为振幅(0~1)
    target = target.astype(np.float64)
    target = normalization(target)
    # 相位初始化应为 0~2π
    phase = 2 * np.pi * np.random.rand(height, width)
    # 复振幅矩阵
    u = np.empty_like(target, dtype="complex")

    uniformity = []
    for num in range(iteration):
        # U = A * exp(i*phase)
        u.real = np.cos(phase)
        u.imag = np.sin(phase)
        # 透镜（傅里叶变换）
        u = np.fft.fftshift(np.fft.fft2(u))
        # 计算频域强度
        u_int = np.abs(u) ** 2
        # 强度归一化
        norm_int = normalization(u_int)
        # 计算目标区域内均一性
        uniformity.append(check_uniformity(u_int, target))
        # 取频域相位
        phase = np.angle(u)
        # 当前频域振幅
        current_amp = np.abs(u)
        # 防止除0
        epsilon = 1e-8
        # 加权更新振幅
        weighted_amp = current_amp * (target / (current_amp + epsilon))
        
        # 用目标振幅替换频域振幅
        u.real = weighted_amp * np.cos(phase)
        u.imag = weighted_amp * np.sin(phase)
        # 逆透镜（逆傅里叶）
        u = np.fft.ifft2(np.fft.ifftshift(u))
        # 更新空间域相位
        phase = np.angle(u)

    holo = hologram(phase)
    cv2.imwrite("{}.jpg".format("holo" + channel), holo)
    rec = reconstruct(norm_int)
    return rec, uniformity

# 主函数
def main():
    img = cv2.imread("lena.jpg")

    # 分离RGB通道
    b, g, r = cv2.split(img)

    # 分别做GS
    r_rec, r_uniformity = GS(r,"_r")
    g_rec, g_uniformity = GS(g,"_g")
    b_rec, b_uniformity = GS(b,"_b")

    # 合并三通道
    result = cv2.merge([b_rec, g_rec, r_rec])
    rec_name = "Final_result"
    cv2.imshow('Final Result', result)
    cv2.imwrite("{}.jpg".format(rec_name), result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(r_uniformity) + 1), r_uniformity, label='Red Channel')
    plt.plot(np.arange(1, len(g_uniformity) + 1), g_uniformity, label='Green Channel')
    plt.plot(np.arange(1, len(b_uniformity) + 1), b_uniformity, label='Blue Channel')
    plt.xlabel("Iteration")
    plt.ylabel("Uniformity")
    plt.ylim(0, 1)
    plt.title("Uniformity Convergence for Each Channel")
    plt.legend() 
    plt.savefig("uniformity_curve.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
