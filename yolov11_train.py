import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
  # 直接加载目标检测的预训练模型
  model = YOLO('yolo11n.pt')
  results = model.train(
      data='mydata0108.yaml',
      epochs=300,
      batch=4,
      imgsz=640,
      # --- 以下是针对极小目标的“特效药” ---
      mosaic=1.0,  # 必须开启！把多张图拼在一起，增加小目标密度
      overlap_mask=False,  # 确保检测框不被互相覆盖
      cls=2.0,  # 提高类别损失权重，让它对“是不是果子”更敏感
      box=10.0,  # 提高框损失权重，让它对“果子在哪”定位更准
      patience=100,  # 延长耐心值，小目标收敛很慢
      optimizer='AdamW', # 换用 AdamW 优化器，对细微特征更友好
      amp = True
  )