//! 多目标跟踪公共组件
//! Common components for multi-object tracking

use super::types::{BBox, PoseKeypoints};

// ========== 公共数据结构 ==========

/// 跟踪点 (用于绘制轨迹)
#[derive(Clone, Debug)]
pub struct TrackPoint {
    pub x: f32,
    pub y: f32,
}

/// 跟踪对象 (统一的跟踪结果)
#[derive(Clone)]
pub struct TrackedObject {
    /// 唯一跟踪ID
    pub id: u32,

    /// 当前边界框 (滤波平滑后)
    pub bbox: BBox,

    /// 历史轨迹 (中心点)
    pub trajectory: Vec<TrackPoint>,

    /// 连续丢失帧数
    pub frames_lost: u32,

    /// 显示颜色 (每个ID不同颜色)
    pub color: (u8, u8, u8),

    /// 总共被跟踪的帧数 (age)
    pub total_frames: u32,
}

impl TrackedObject {
    /// 获取中心点
    pub fn center(&self) -> (f32, f32) {
        let cx = (self.bbox.x1 + self.bbox.x2) / 2.0;
        let cy = (self.bbox.y1 + self.bbox.y2) / 2.0;
        (cx, cy)
    }

    /// 添加轨迹点
    pub fn add_trajectory_point(&mut self) {
        let (cx, cy) = self.center();
        self.trajectory.push(TrackPoint { x: cx, y: cy });

        // 限制轨迹长度
        if self.trajectory.len() > 30 {
            self.trajectory.remove(0);
        }
    }
}

// ========== 卡尔曼滤波器 ==========

/// 简化卡尔曼滤波器 (用于单个边界框的位置和尺寸平滑)
/// 状态向量: [x_center, y_center, width, height, vx, vy, vw, vh]
#[derive(Clone)]
pub struct KalmanBoxFilter {
    /// 状态估计: [cx, cy, w, h, vx, vy, vw, vh]
    state: [f32; 8],

    /// 估计误差协方差 (简化为对角阵)
    p: [f32; 8],

    /// 过程噪声 (运动不确定性)
    q: f32,

    /// 观测噪声 (测量不确定性)
    r: f32,

    /// 速度衰减因子 (用于静止目标,0.9-0.99)
    velocity_decay: f32,

    /// 静止阈值 (像素/帧)
    stationary_threshold: f32,

    /// 连续静止帧数计数器
    stationary_count: u32,
}

impl KalmanBoxFilter {
    /// 创建新的卡尔曼滤波器
    ///
    /// # 参数
    /// - `bbox`: 初始边界框
    /// - `q`: 过程噪声 (0.1-1.0, 越小越平滑)
    /// - `r`: 观测噪声 (1.0-50.0, 越大越平滑)
    pub fn new(bbox: &BBox, q: f32, r: f32) -> Self {
        let cx = (bbox.x1 + bbox.x2) / 2.0;
        let cy = (bbox.y1 + bbox.y2) / 2.0;
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;

        Self {
            state: [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], // 初始速度为0
            p: [10.0; 8],
            q,
            r,
            velocity_decay: 0.95,      // 速度衰减因子:每帧保留95%速度
            stationary_threshold: 2.0, // 静止阈值:小于2像素/帧视为静止
            stationary_count: 0,       // 初始未静止
        }
    }

    /// 预测下一帧状态 (匀速运动模型 + 速度衰减)
    pub fn predict(&mut self) {
        // 检测是否静止 (速度小于阈值)
        let speed = (self.state[4] * self.state[4] + self.state[5] * self.state[5]).sqrt();
        let is_stationary = speed < self.stationary_threshold;

        if is_stationary {
            self.stationary_count += 1;
            // 静止时更强的速度衰减
            let decay = if self.stationary_count > 3 {
                0.7 // 连续静止3帧后,大幅衰减速度
            } else {
                self.velocity_decay
            };
            self.state[4] *= decay;
            self.state[5] *= decay;
            self.state[6] *= decay;
            self.state[7] *= decay;
        } else {
            self.stationary_count = 0;
            // 正常运动时轻微衰减
            self.state[4] *= self.velocity_decay;
            self.state[5] *= self.velocity_decay;
            self.state[6] *= 0.98; // 尺寸变化更慢
            self.state[7] *= 0.98;
        }

        // 状态转移: x = x + vx, y = y + vy, w = w + vw, h = h + vh
        self.state[0] += self.state[4]; // cx += vx
        self.state[1] += self.state[5]; // cy += vy
        self.state[2] += self.state[6]; // w += vw
        self.state[3] += self.state[7]; // h += vh

        // 协方差预测: P = P + Q (静止时减小过程噪声)
        let q_factor = if is_stationary { 0.5 } else { 1.0 };
        for i in 0..8 {
            self.p[i] += self.q * q_factor;
        }
    }

    /// 更新 (融合观测值,自适应噪声调整)
    pub fn update(&mut self, bbox: &BBox) {
        let cx = (bbox.x1 + bbox.x2) / 2.0;
        let cy = (bbox.y1 + bbox.y2) / 2.0;
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;

        // 计算观测残差
        let y = [
            cx - self.state[0],
            cy - self.state[1],
            w - self.state[2],
            h - self.state[3],
        ];

        // 根据残差大小自适应调整观测噪声
        let residual_norm = (y[0] * y[0] + y[1] * y[1]).sqrt();
        let adaptive_r = if residual_norm < self.stationary_threshold {
            // 静止或小幅移动:降低观测噪声,更信任观测值
            self.r * 0.3
        } else if residual_norm < 10.0 {
            // 正常运动
            self.r
        } else {
            // 大幅跳变:增加观测噪声,更信任预测值
            self.r * 3.0
        };

        // 卡尔曼增益: K = P / (P + R)
        let k = [
            self.p[0] / (self.p[0] + adaptive_r),
            self.p[1] / (self.p[1] + adaptive_r),
            self.p[2] / (self.p[2] + adaptive_r),
            self.p[3] / (self.p[3] + adaptive_r),
            self.p[4] / (self.p[4] + adaptive_r * 10.0),
            self.p[5] / (self.p[5] + adaptive_r * 10.0),
            self.p[6] / (self.p[6] + adaptive_r * 10.0),
            self.p[7] / (self.p[7] + adaptive_r * 10.0),
        ];

        // 状态更新: x = x + K * y
        self.state[0] += k[0] * y[0];
        self.state[1] += k[1] * y[1];
        self.state[2] += k[2] * y[2];
        self.state[3] += k[3] * y[3];

        // 速度更新 (静止时减小速度估计影响)
        let velocity_gain = if residual_norm < self.stationary_threshold {
            0.3
        } else {
            1.0
        };
        self.state[4] += k[4] * y[0] * velocity_gain;
        self.state[5] += k[5] * y[1] * velocity_gain;
        self.state[6] += k[6] * y[2] * velocity_gain;
        self.state[7] += k[7] * y[3] * velocity_gain;

        // 协方差更新: P = (I - K) * P
        for i in 0..8 {
            self.p[i] *= 1.0 - k[i];
        }

        // 重置静止计数器 (收到新观测)
        if residual_norm >= self.stationary_threshold {
            self.stationary_count = 0;
        }
    }

    /// 获取当前状态的边界框
    pub fn get_state_bbox(&self) -> BBox {
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

    /// 获取预测的边界框 (用于匹配)
    pub fn get_predicted_bbox(&self) -> BBox {
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

    /// 获取速度向量
    pub fn get_velocity(&self) -> (f32, f32) {
        (self.state[4], self.state[5])
    }

    /// 获取位置不确定性 (用于马氏距离计算)
    pub fn get_position_uncertainty(&self) -> f32 {
        (self.p[0] + self.p[1]).sqrt()
    }

    /// 获取尺寸不确定性 (用于马氏距离计算)
    pub fn get_size_uncertainty(&self) -> f32 {
        (self.p[2] + self.p[3]).sqrt()
    }
}

// ========== 跟踪器统一接口 ==========

/// 多目标跟踪器 Trait
///
/// 所有跟踪算法(DeepSort, ByteTrack等)都应实现此接口
pub trait Tracker {
    /// 更新跟踪器
    ///
    /// # 参数
    /// - `detections`: 当前帧的检测框
    /// - `keypoints`: 当前帧的姿态关键点 (用于外观特征)
    /// - `frame_rgba`: 可选的图像数据 (格式: RGBA, width, height)
    ///
    /// # 返回
    /// 当前所有活跃的跟踪对象
    fn update(
        &mut self,
        detections: &[BBox],
        keypoints: &[PoseKeypoints],
        frame_rgba: Option<(&[u8], u32, u32)>,
    ) -> &[TrackedObject];

    /// 重置跟踪器 (清除所有跟踪)
    fn reset(&mut self);

    /// 获取当前跟踪数量
    fn track_count(&self) -> usize;
}

// ========== 工具函数 ==========

/// 计算两个边界框的IOU (Intersection over Union)
pub fn compute_iou(bbox1: &BBox, bbox2: &BBox) -> f32 {
    let x1 = bbox1.x1.max(bbox2.x1);
    let y1 = bbox1.y1.max(bbox2.y1);
    let x2 = bbox1.x2.min(bbox2.x2);
    let y2 = bbox1.y2.min(bbox2.y2);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
    let area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
    let union = area1 + area2 - intersection;

    if union <= 0.0 {
        return 0.0;
    }

    intersection / union
}

/// 根据ID生成不同颜色
pub fn id_to_color(id: u32) -> (u8, u8, u8) {
    let hue = (id as f32 * 137.508) % 360.0; // 黄金角度采样
    hsv_to_rgb(hue, 0.8, 0.9)
}

/// HSV转RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}
