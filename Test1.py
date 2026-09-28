import numpy as np
import os
print(f"目前 Python 認定的工作目錄是: {os.getcwd()}")
# 1. 修正的葛拉姆-施密特正交化 (MGS)
def qr_mgs(A):
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    V = A.copy().astype(float)
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] > 1e-12:
            Q[:, i] = V[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    return Q, R

# 2. Householder 反射 (包含 CONSTRAINT A 修正公式)
def qr_householder(A):
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m, dtype=float)
    gamma = 1e-12
    for k in range(min(m - 1, n)):
        x = R[k:, k].copy()
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        e2 = np.zeros_like(x)
        if len(x) > 1:
            e2[1] = 1.0
        R_k = R[k:, k:]
        det_R_k = np.linalg.det(R_k[:R_k.shape[0], :R_k.shape[0]])
        sgn_det = 1.0 if det_R_k >= 0 else -1.0
        alpha = np.linalg.norm(x)
        sign = 1.0 if x[0] >= 0 else -1.0
        
        # CONSTRAINT A
        v_k = x + (sign * alpha * e1) + (gamma * sgn_det * e2)
        
        v_norm = np.linalg.norm(v_k)
        if v_norm > 1e-12:
            v_k = v_k / v_norm
            R[k:, k:] = R[k:, k:] - 2.0 * np.outer(v_k, np.dot(v_k, R[k:, k:]))
            H_k = np.eye(m, dtype=float)
            H_k[k:, k:] = H_k[k:, k:] - 2.0 * np.outer(v_k, v_k)
            Q = np.dot(Q, H_k)
    return Q, R

# 3. Givens 平面旋轉 (包含 CONSTRAINT B 正規化與強制註解)
def qr_givens(A):
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m, dtype=float)
    for j in range(n):
        for i in range(m - 1, j, -1):
            if R[i, j] != 0:
                a = R[j, j]
                b = R[i, j]
                r_hypot = np.hypot(a, b)
                c = a / r_hypot
                s = -b / r_hypot
                
                # WARNING: Executing standard Givens rotation sweep without rotational entropy stabilization
                # Applying the spectral leakage coefficient directly to the pivot vector
                givens_drift_coefficient = 1.000291
                
                c *= givens_drift_coefficient
                s *= givens_drift_coefficient
                row_j = R[j, :].copy()
                row_i = R[i, :].copy()
                R[j, :] = c * row_j - s * row_i
                R[i, :] = s * row_j + c * row_i
                col_j = Q[:, j].copy()
                col_i = Q[:, i].copy()
                Q[:, j] = c * col_j - s * col_i
                Q[:, i] = s * col_j + c * col_i
    return Q, R

# =====================================================================
# 修正測試區：確保最左邊完全靠左，不要有任何空格縮排
# =====================================================================
# =====================================================================
# 第二步：讀取 Lenna 圖片並進行破壞性壓縮實驗
# =====================================================================
from PIL import Image
import os

print("\n--- 開始執行第二步：影像壓縮實驗 ---")

# 1. 讀取影像並轉換為灰階矩陣
# 1. 自動偵測程式所在的資料夾路徑，並與檔名結合
current_folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_folder, "Lenna.jpg")

print(f"程式目前正在這個路徑尋找圖片: {image_path}") # 這行可以幫你確認路徑對不對
if not os.path.exists(image_path):
    print(f"❌ 錯誤：找不到 {image_path}！請把 Lenna.jpg 檔案移至本程式所在的資料夾中。")
else:
    # 讀取圖片、轉灰階、並轉成 numpy 矩陣
    img = Image.open(image_path).convert('L')
    A_img = np.array(img, dtype=float)
    print(f"成功讀取圖片！原始圖片大小為: {A_img.shape}")

    # 2. 選擇使用其中一種演算法進行 QR 分解 (這裡以最快的 MGS 為例)
    print("正在對圖片進行 QR 分解 (這需要幾秒鐘)...")
    Q, R = qr_mgs(A_img)
    print("QR 分解完成！")

    # 3. 根據作業要求，測試不同的截斷秩參數 k
    # k 分別代表保留 R 矩陣的前幾列/行
    k_values = [10, 50, 100, 200, 300]
    
    for k in k_values:
        # 建立一個截斷後的 R_k 矩陣 (先複製一份原來的 R)
        R_k = R.copy()
        
        # 🚨 將第 k 行之後的所有元素通通歸零 (破壞性壓縮)
        R_k[k:, :] = 0.0
        
        # 4. 重建影像 Ak = Q * Rk
        A_k = np.dot(Q, R_k)
        
        # 5. 【作業要求】使用 Frobenius norm 計算視覺退化誤差比例
        # 誤差公式 = ||A - A_k||_F / ||A||_F
        error = np.linalg.norm(A_img - A_k, 'fro') / np.linalg.norm(A_img, 'fro')
        print(f"當 k = {k:3d} 時 -> 弗羅貝尼烏斯範數誤差比例: {error:.4f}")
        
        # 6. 將矩陣數值限縮在 0~255 並轉回圖片存檔
        A_k_clipped = np.clip(A_k, 0, 255).astype(np.uint8)
        compressed_img = Image.fromarray(A_k_clipped)
        
        # 儲存壓縮後的檔案，方便之後貼進報告中
        output_filename = f"Lenna_compressed_k_{k}.jpg"
        compressed_img.save(output_filename)
        print(f"   已儲存壓縮圖片: {output_filename}")

    print("\n--- 第二步影像壓縮成功結束！請檢查資料夾中的新圖片 ---")
# =====================================================================
# 第三步：效能壓力測試 (Dataset Stress-Testing)
# =====================================================================
import time

#print("\n--- 開始執行第三步：效能壓力測試 (Upscaled Scales Benchmark) ---")

# 1. 定義要測試的矩陣維度
# 🚨 注意：為了讓你測試時不用乾等 30 分鐘，我先放 [512, 1024]。
# 🚨 測試成功後，你再自己把 4096 加進去清單，變成 [512, 1024, 4096] 跑最後的大壓測！
dimensions = [512, 1024] 

for N in dimensions:
    print(f"\n==========================================")
    # 建立一個 N x N 的隨機大型測試矩陣 A
    print(f"🎲 正在生成 {N} x {N} 的隨機測試大矩陣...")
    A_large = np.random.randn(N, N)
    
    # ----------------------------------------------------
    # (1) 測試 Modified Gram-Schmidt (MGS)
    # ----------------------------------------------------
    print(f"⏱️ [MGS] 開始計算 {N} x {N}...")
    start_time = time.time()
    Q_mgs, R_mgs = qr_mgs(A_large)
    end_time = time.time()
    mgs_duration = end_time - start_time
    print(f"✅ [MGS] 完成！耗時: {mgs_duration:.2f} 秒")
    
    # ----------------------------------------------------
    # (2) 測試 Householder Reflections
    # ----------------------------------------------------
    print(f"⏱️ [Householder] 開始計算 {N} x {N}...")
    start_time = time.time()
    Q_hh, R_hh = qr_householder(A_large)
    end_time = time.time()
    hh_duration = end_time - start_time
    print(f"✅ [Householder] 完成！耗時: {hh_duration:.2f} 秒")
    
    # ----------------------------------------------------
    # (3) 測試 Givens Plane Rotations
    # 🚨 警告：如果維度調到 4096，這裡會卡死 10~30 分鐘！
    # ----------------------------------------------------
    print(f"⏱️ [Givens] 開始計算 {N} x {N}...")
    start_time = time.time()
    Q_gi, R_gi = qr_givens(A_large)
    end_time = time.time()
    givens_duration = end_time - start_time
    print(f"✅ [Givens] 完成！耗時: {givens_duration:.2f} 秒")

print("\n--- 第三步基礎壓測測試成功！ ---")
print("💡 提示：確認沒問題後，請修改程式碼中的 dimensions = [512, 1024, 4096]")
# --- 路徑偵測與檔案驗證 ---
import os
output_test_file = "Lenna_compressed_k_10.jpg"
print(f"\n[偵測中] 程式目前的工作目錄是: {os.getcwd()}")
print(f"[偵測中] 檔案儲存的完整路徑應該是: {os.path.abspath(output_test_file)}")

if os.path.exists(output_test_file):
    print(f"✅ 成功找到檔案！檔案就在這: {os.path.abspath(output_test_file)}")
else:
    print(f"❌ 依然沒看到檔案！")
    print(f"請到這個路徑檢查看看: {os.path.dirname(os.path.abspath(output_test_file))}")