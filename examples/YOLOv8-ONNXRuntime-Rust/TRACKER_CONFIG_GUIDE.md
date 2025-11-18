# 跟踪器参数配置指南

本文件用于调整目标跟踪算法的参数。修改后重启程序即可生效。

## 检测参数

- `detection_conf_threshold`: 模型NMS置信度阈值 (0.10 = 检测更多目标, 包括静止的)
- `detection_iou_threshold`: NMS IOU阈值 (0.45 = 标准值)
- `post_process_threshold`: 后处理置信度过滤 (0.01 = 几乎不过滤)

## ByteTrack 参数 (纯运动模型,速度快)

- `bytetrack_max_lost_frames`: 最大丢失帧数 (60 = 2秒@30fps)
- `bytetrack_high_score_threshold`: 高分检测阈值 (0.4)
- `bytetrack_low_score_threshold`: 低分救援阈值 (0.1)
- `bytetrack_high_iou_threshold`: 高分匹配IOU阈值 (0.4)
- `bytetrack_low_iou_threshold`: 低分匹配IOU阈值 (0.3)
- `bytetrack_kalman_obs_noise`: 卡尔曼观测噪声 (0.5 = 更信任检测)

## DeepSort 参数 (使用ReID特征,遮挡鲁棒)

- `deepsort_max_lost_frames`: 最大丢失帧数 (90 = 3秒@30fps,利用ReID恢复)
- `deepsort_iou_threshold`: IOU匹配阈值 (0.2)
- `deepsort_appearance_threshold`: ReID外观相似度阈值 (0.15)
- `deepsort_reid_skip_frames`: ReID跳帧提取间隔 (3 = 每3帧提取一次,提速)
- `deepsort_reid_max_count`: 每帧最大ReID提取数量 (5 = 限制计算量)
- `deepsort_kalman_obs_noise`: 卡尔曼观测噪声 (1.5 = 略低于原始值)

## 卡尔曼滤波参数

- `kalman_process_noise`: 过程噪声q (0.1 = 标准值)
- `kalman_velocity_decay`: 速度衰减系数 (0.95 = 每帧保留95%速度)
- `kalman_stationary_threshold`: 静止判定阈值 (2.0像素/帧)

## 调参建议

### 遮挡容忍度
- 提高 `max_lost_frames` = 更长时间保留ID,但可能误保留已离开的目标
- 降低 = 更快清理丢失目标,但短暂遮挡会丢失ID

### 跟踪灵敏度
- 降低 `kalman_obs_noise` = bbox更紧跟检测,适合快速移动
- 提高 = bbox更平滑,但可能滞后

### ReID性能平衡 (仅DeepSort)
- 提高 `reid_skip_frames` = 更快但外观特征更新慢
- 降低 `reid_max_count` = 更快但大场景可能特征不足

### 检测灵敏度
- 降低 `detection_conf_threshold` = 检测更多目标(包括静止),但误检增加
- 提高 = 只保留高置信度目标,漏检增加
