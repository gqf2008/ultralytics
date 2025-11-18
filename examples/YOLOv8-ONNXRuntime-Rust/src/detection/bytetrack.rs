//! ByteTrack 算法实现
//! ByteTrack: Simple and effective multi-object tracking
//!
//! 核心思想:
//! 1. 高低分检测框分开处理
//! 2. 高分框优先匹配 (IOU)
//! 3. 低分框救援丢失的轨迹
//! 4. 纯运动模型,无需外观特征

use super::tracker::{compute_iou, KalmanBoxFilter, TrackPoint};
use super::types::BBox;

/// ByteTrack 跟踪对象
#[derive(Clone)]
pub struct ByteTrackedPerson {
    /// 唯一跟踪ID
    pub id: u32,

    /// 当前边界框 (卡尔曼滤波平滑后)
    pub bbox: BBox,

    /// 卡尔曼滤波器
    kalman: KalmanBoxFilter,

    /// 历史轨迹 (中心点)
    pub trajectory: Vec<TrackPoint>,

    /// 连续丢失帧数
    pub frames_lost: u32,

    /// 显示颜色
    pub color: (u8, u8, u8),

    /// 总共被跟踪的帧数
    pub total_frames: u32,

    /// 检测置信度 (用于判断是否为高分轨迹)
    pub score: f32,

    /// 是否静止
    is_stationary: bool,
}

impl ByteTrackedPerson {
    fn new(id: u32, bbox: BBox, color: (u8, u8, u8)) -> Self {
        // ByteTrack优化: 降低观测噪声(r=0.5),更信任检测结果,快速响应移动
        let kalman = KalmanBoxFilter::new(&bbox, 0.1, 0.5);
        let smoothed_bbox = kalman.get_state_bbox();

        let center = TrackPoint {
            x: (smoothed_bbox.x1 + smoothed_bbox.x2) / 2.0,
            y: (smoothed_bbox.y1 + smoothed_bbox.y2) / 2.0,
        };

        Self {
            id,
            bbox: smoothed_bbox,
            kalman,
            trajectory: vec![center],
            frames_lost: 0,
            color,
            total_frames: 1,
            score: bbox.confidence,
            is_stationary: false,
        }
    }

    fn predict(&mut self) {
        self.kalman.predict();
        self.bbox = self.kalman.get_state_bbox();
    }

    fn update(&mut self, bbox: BBox) {
        // 检测是否静止
        let predicted = self.kalman.get_predicted_bbox();
        let dx = (bbox.x1 + bbox.x2) / 2.0 - (predicted.x1 + predicted.x2) / 2.0;
        let dy = (bbox.y1 + bbox.y2) / 2.0 - (predicted.y1 + predicted.y2) / 2.0;
        let movement = (dx * dx + dy * dy).sqrt();

        self.is_stationary = movement < 3.0; // 小于3像素视为静止

        self.kalman.update(&bbox);
        self.bbox = self.kalman.get_state_bbox();
        self.frames_lost = 0;
        self.total_frames += 1;
        self.score = bbox.confidence;

        // 添加轨迹点
        let center = TrackPoint {
            x: (self.bbox.x1 + self.bbox.x2) / 2.0,
            y: (self.bbox.y1 + self.bbox.y2) / 2.0,
        };
        self.trajectory.push(center);

        // 只保留最近50个点
        if self.trajectory.len() > 50 {
            self.trajectory.remove(0);
        }
    }

    fn mark_lost(&mut self) {
        self.frames_lost += 1;
        self.predict();
    }

    fn get_predicted_bbox(&self) -> BBox {
        self.kalman.get_predicted_bbox()
    }
}

/// ByteTrack 追踪器
pub struct ByteTracker {
    /// 当前跟踪的人
    tracked_persons: Vec<ByteTrackedPerson>,

    /// 下一个分配的ID
    next_id: u32,

    /// 最大允许丢失帧数
    max_lost_frames: u32,

    /// 高分检测阈值
    high_score_threshold: f32,

    /// 低分检测阈值 (用于救援)
    low_score_threshold: f32,

    /// 高分匹配 IOU 阈值
    high_iou_threshold: f32,

    /// 低分匹配 IOU 阈值
    low_iou_threshold: f32,

    /// 预定义颜色表
    color_palette: Vec<(u8, u8, u8)>,
}

impl ByteTracker {
    pub fn new() -> Self {
        let color_palette = vec![
            (255, 64, 64),   // 红色
            (64, 255, 64),   // 绿色
            (64, 64, 255),   // 蓝色
            (255, 255, 64),  // 黄色
            (255, 64, 255),  // 品红
            (64, 255, 255),  // 青色
            (255, 128, 0),   // 橙色
            (128, 0, 255),   // 紫色
            (255, 128, 192), // 粉色
            (128, 255, 128), // 浅绿
        ];

        Self {
            tracked_persons: Vec::new(),
            next_id: 1,
            max_lost_frames: 60,       // 60帧(约2秒) - 提高遮挡容忍度
            high_score_threshold: 0.4, // 高分阈值 (降低让更多框参与)
            low_score_threshold: 0.1,  // 低分阈值 (救援用)
            high_iou_threshold: 0.4,   // 高分匹配阈值 (提高避免误匹配)
            low_iou_threshold: 0.3,    // 低分匹配阈值 (降低救援更宽松)
            color_palette,
        }
    }

    /// 更新跟踪 (ByteTrack 三步匹配)
    pub fn update(&mut self, detections: &[BBox]) -> &[ByteTrackedPerson] {
        // 1. 所有轨迹先预测
        for tracked in &mut self.tracked_persons {
            tracked.predict();
        }

        // 2. 分离高低分检测框
        let mut high_dets: Vec<(usize, &BBox)> = Vec::new();
        let mut low_dets: Vec<(usize, &BBox)> = Vec::new();

        for (idx, det) in detections.iter().enumerate() {
            if det.confidence >= self.high_score_threshold {
                high_dets.push((idx, det));
            } else if det.confidence >= self.low_score_threshold {
                low_dets.push((idx, det));
            }
        }

        // 3. 第一轮匹配: 高分检测 + 所有轨迹
        let mut matched_det = vec![false; detections.len()];
        let mut matched_track = vec![false; self.tracked_persons.len()];

        let assignments = self.match_detections_to_tracks(
            &high_dets,
            &(0..self.tracked_persons.len()).collect::<Vec<_>>(),
            self.high_iou_threshold,
        );

        for (det_idx, track_idx) in assignments {
            matched_det[det_idx] = true;
            matched_track[track_idx] = true;
            self.tracked_persons[track_idx].update(detections[det_idx].clone());
        }

        // 4. 第二轮匹配: 低分检测 + 未匹配的轨迹 (救援)
        let unmatched_tracks: Vec<usize> = (0..self.tracked_persons.len())
            .filter(|&idx| !matched_track[idx])
            .collect();

        let low_assignments =
            self.match_detections_to_tracks(&low_dets, &unmatched_tracks, self.low_iou_threshold);

        for (det_idx, track_idx) in low_assignments {
            matched_det[det_idx] = true;
            matched_track[track_idx] = true;
            self.tracked_persons[track_idx].update(detections[det_idx].clone());
        }

        // 5. 未匹配的高分检测 → 新建轨迹
        for (det_idx, &matched) in matched_det.iter().enumerate() {
            if !matched && detections[det_idx].confidence >= self.high_score_threshold {
                let color = self.color_palette[self.next_id as usize % self.color_palette.len()];
                let tracked =
                    ByteTrackedPerson::new(self.next_id, detections[det_idx].clone(), color);
                self.tracked_persons.push(tracked);
                self.next_id += 1;
            }
        }

        // 6. 未匹配的轨迹 → 标记丢失
        for (track_idx, &matched) in matched_track.iter().enumerate() {
            if !matched {
                self.tracked_persons[track_idx].mark_lost();
            }
        }

        // 7. 删除丢失太久的轨迹
        self.tracked_persons
            .retain(|t| t.frames_lost <= self.max_lost_frames);

        &self.tracked_persons
    }

    /// IOU 匹配
    fn match_detections_to_tracks(
        &self,
        detections: &[(usize, &BBox)],
        track_indices: &[usize],
        iou_threshold: f32,
    ) -> Vec<(usize, usize)> {
        if detections.is_empty() || track_indices.is_empty() {
            return Vec::new();
        }

        // 计算 IOU 代价矩阵
        let mut candidates = Vec::new();
        for (local_det_idx, (det_idx, detection)) in detections.iter().enumerate() {
            for (local_track_idx, &track_idx) in track_indices.iter().enumerate() {
                let track = &self.tracked_persons[track_idx];
                let iou = compute_iou(detection, &track.get_predicted_bbox());

                if iou >= iou_threshold {
                    let cost = 1.0 - iou;
                    candidates.push((cost, *det_idx, local_det_idx, track_idx, local_track_idx));
                }
            }
        }

        // 贪心匹配: 按代价排序
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut assignments = Vec::new();
        let mut used_det = vec![false; detections.len()];
        let mut used_track = vec![false; track_indices.len()];

        for (_, det_idx, local_det_idx, track_idx, local_track_idx) in candidates {
            if !used_det[local_det_idx] && !used_track[local_track_idx] {
                assignments.push((det_idx, track_idx));
                used_det[local_det_idx] = true;
                used_track[local_track_idx] = true;
            }
        }

        assignments
    }

    /// 获取跟踪统计信息
    pub fn get_stats(&self) -> String {
        format!(
            "跟踪: {} 人 | 总ID: {}",
            self.tracked_persons.len(),
            self.next_id - 1
        )
    }
}

impl Default for ByteTracker {
    fn default() -> Self {
        Self::new()
    }
}
