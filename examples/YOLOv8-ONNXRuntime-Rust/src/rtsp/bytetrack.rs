/// ByteTrack 算法实现
/// ByteTrack: Simple and effective multi-object tracking
///
/// 核心思想:
/// 1. 高低分检测框分开处理
/// 2. 高分框优先匹配 (IOU)
/// 3. 低分框救援丢失的轨迹
/// 4. 纯运动模型,无需外观特征
use super::types::BBox;

/// 跟踪点 (用于绘制轨迹)
#[derive(Clone, Debug)]
pub struct TrackPoint {
    pub x: f32,
    pub y: f32,
}

/// 简化卡尔曼滤波器 (用于单个边界框)
#[derive(Clone)]
struct KalmanBoxFilter {
    // 状态估计: [cx, cy, w, h, vx, vy, vw, vh]
    state: [f32; 8],
    // 估计误差协方差 (简化为对角阵)
    p: [f32; 8],
    // 过程噪声
    q: f32,
    // 观测噪声
    r: f32,
}

impl KalmanBoxFilter {
    fn new(bbox: &BBox) -> Self {
        let cx = (bbox.x1 + bbox.x2) / 2.0;
        let cy = (bbox.y1 + bbox.y2) / 2.0;
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;

        Self {
            state: [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0],
            p: [10.0; 8], // ByteTrack 用更小的初始不确定性
            q: 0.1,       // 较小的过程噪声 (运动更确定)
            r: 1.0,       // 适中的观测噪声
        }
    }

    fn predict(&mut self) {
        // 状态转移: 匀速运动模型
        self.state[0] += self.state[4];
        self.state[1] += self.state[5];
        self.state[2] += self.state[6];
        self.state[3] += self.state[7];

        // 协方差预测
        for i in 0..8 {
            self.p[i] += self.q;
        }
    }

    fn update(&mut self, bbox: &BBox) {
        let cx = (bbox.x1 + bbox.x2) / 2.0;
        let cy = (bbox.y1 + bbox.y2) / 2.0;
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;

        // 卡尔曼增益
        let k = [
            self.p[0] / (self.p[0] + self.r),
            self.p[1] / (self.p[1] + self.r),
            self.p[2] / (self.p[2] + self.r),
            self.p[3] / (self.p[3] + self.r),
            self.p[4] / (self.p[4] + self.r * 5.0), // 速度增益降低
            self.p[5] / (self.p[5] + self.r * 5.0),
            self.p[6] / (self.p[6] + self.r * 5.0),
            self.p[7] / (self.p[7] + self.r * 5.0),
        ];

        // 观测残差
        let y = [
            cx - self.state[0],
            cy - self.state[1],
            w - self.state[2],
            h - self.state[3],
        ];

        // 状态更新
        self.state[0] += k[0] * y[0];
        self.state[1] += k[1] * y[1];
        self.state[2] += k[2] * y[2];
        self.state[3] += k[3] * y[3];

        // 速度更新
        self.state[4] += k[4] * y[0];
        self.state[5] += k[5] * y[1];
        self.state[6] += k[6] * y[2];
        self.state[7] += k[7] * y[3];

        // 协方差更新
        for i in 0..8 {
            self.p[i] *= 1.0 - k[i];
        }
    }

    fn get_bbox(&self) -> BBox {
        let cx = self.state[0];
        let cy = self.state[1];
        let w = self.state[2].max(1.0);
        let h = self.state[3].max(1.0);

        BBox {
            x1: cx - w / 2.0,
            y1: cy - h / 2.0,
            x2: cx + w / 2.0,
            y2: cy + h / 2.0,
            confidence: 1.0,
            class_id: 0,
        }
    }

    fn get_predicted_bbox(&self) -> BBox {
        let cx = self.state[0] + self.state[4];
        let cy = self.state[1] + self.state[5];
        let w = (self.state[2] + self.state[6]).max(1.0);
        let h = (self.state[3] + self.state[7]).max(1.0);

        BBox {
            x1: cx - w / 2.0,
            y1: cy - h / 2.0,
            x2: cx + w / 2.0,
            y2: cy + h / 2.0,
            confidence: 1.0,
            class_id: 0,
        }
    }
}

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
}

impl ByteTrackedPerson {
    fn new(id: u32, bbox: BBox, color: (u8, u8, u8)) -> Self {
        let kalman = KalmanBoxFilter::new(&bbox);
        let smoothed_bbox = kalman.get_bbox();

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
        }
    }

    fn predict(&mut self) {
        self.kalman.predict();
        self.bbox = self.kalman.get_bbox();
    }

    fn update(&mut self, bbox: BBox) {
        self.kalman.update(&bbox);
        self.bbox = self.kalman.get_bbox();
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
            max_lost_frames: 30,       // 30帧 ≈ 1秒@30fps
            high_score_threshold: 0.5, // 高分阈值
            low_score_threshold: 0.1,  // 低分阈值 (救援用)
            high_iou_threshold: 0.3,   // 高分匹配阈值
            low_iou_threshold: 0.5,    // 低分匹配阈值 (更严格)
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
                let iou = Self::compute_iou(detection, &track.get_predicted_bbox());

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

    /// 计算 IOU
    fn compute_iou(bbox1: &BBox, bbox2: &BBox) -> f32 {
        let x1 = bbox1.x1.max(bbox2.x1);
        let y1 = bbox1.y1.max(bbox2.y1);
        let x2 = bbox1.x2.min(bbox2.x2);
        let y2 = bbox1.y2.min(bbox2.y2);

        if x2 < x1 || y2 < y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
        let area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
        let union = area1 + area2 - intersection;

        intersection / union
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
