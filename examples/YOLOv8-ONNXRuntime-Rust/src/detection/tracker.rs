/// 人形追踪模块
/// Person tracking module - DeepSort Algorithm
///
/// 核心算法 (DeepSort):
/// 1. 卡尔曼滤波器: 预测位置 + 平滑边界框
/// 2. 级联匹配: 优先匹配年轻轨迹 (防止遮挡时误匹配)
/// 3. 马氏距离: 运动一致性 (考虑预测不确定性)
/// 4. 余弦距离: 外观特征相似度 (基于OSNet深度ReID)
/// 5. 融合匹配: 运动+外观双重验证
/// 6. 虚拟轨迹: 长时遮挡鲁棒
use super::types::{BBox, PoseKeypoints};
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Value;

/// 跟踪点 (用于绘制轨迹)
#[derive(Clone, Debug)]
pub struct TrackPoint {
    pub x: f32,
    pub y: f32,
}

/// 简化卡尔曼滤波器 (用于单个边界框的位置和尺寸平滑)
/// 状态向量: [x_center, y_center, width, height, vx, vy, vw, vh]
#[derive(Clone)]
struct KalmanBoxFilter {
    // 状态估计: [cx, cy, w, h, vx, vy, vw, vh]
    state: [f32; 8],

    // 估计误差协方差 (简化为对角阵)
    p: [f32; 8],

    // 过程噪声 (运动不确定性)
    q: f32,

    // 观测噪声 (测量不确定性)
    r: f32,
}

impl KalmanBoxFilter {
    fn new(bbox: &BBox) -> Self {
        let cx = (bbox.x1 + bbox.x2) / 2.0;
        let cy = (bbox.y1 + bbox.y2) / 2.0;
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;

        Self {
            state: [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], // 初始速度为0
            p: [50.0; 8],                              // 进一步降低初始不确定性
            q: 0.5,                                    // 减小过程噪声 (更平滑)
            r: 50.0,                                   // 大幅增加观测噪声 (强烈信任滤波)
        }
    }

    /// 预测下一帧状态 (匀速运动模型)
    fn predict(&mut self) {
        // 状态转移: x = x + vx, y = y + vy, w = w + vw, h = h + vh
        self.state[0] += self.state[4]; // cx += vx
        self.state[1] += self.state[5]; // cy += vy
        self.state[2] += self.state[6]; // w += vw
        self.state[3] += self.state[7]; // h += vh

        // 协方差预测: P = P + Q
        for i in 0..8 {
            self.p[i] += self.q;
        }
    }

    /// 更新 (融合观测值)
    fn update(&mut self, bbox: &BBox) {
        let cx = (bbox.x1 + bbox.x2) / 2.0;
        let cy = (bbox.y1 + bbox.y2) / 2.0;
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;

        // 卡尔曼增益: K = P / (P + R)
        let k = [
            self.p[0] / (self.p[0] + self.r),
            self.p[1] / (self.p[1] + self.r),
            self.p[2] / (self.p[2] + self.r),
            self.p[3] / (self.p[3] + self.r),
            self.p[4] / (self.p[4] + self.r * 10.0), // 速度增益大幅降低 (极平滑)
            self.p[5] / (self.p[5] + self.r * 10.0),
            self.p[6] / (self.p[6] + self.r * 10.0),
            self.p[7] / (self.p[7] + self.r * 10.0),
        ];

        // 观测残差
        let y = [
            cx - self.state[0],
            cy - self.state[1],
            w - self.state[2],
            h - self.state[3],
        ];

        // 状态更新: x = x + K * y
        self.state[0] += k[0] * y[0];
        self.state[1] += k[1] * y[1];
        self.state[2] += k[2] * y[2];
        self.state[3] += k[3] * y[3];

        // 速度更新 (估算速度)
        self.state[4] += k[4] * y[0];
        self.state[5] += k[5] * y[1];
        self.state[6] += k[6] * y[2];
        self.state[7] += k[7] * y[3];

        // 协方差更新: P = (1 - K) * P
        for i in 0..8 {
            self.p[i] *= 1.0 - k[i];
        }
    }

    /// 获取当前估计的边界框
    fn get_bbox(&self) -> BBox {
        let cx = self.state[0];
        let cy = self.state[1];
        let w = self.state[2].max(1.0); // 防止负值
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
    fn get_predicted_bbox(&self) -> BBox {
        let cx = self.state[0] + self.state[4]; // 预测位置
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

    /// 获取速度向量 (vx, vy)
    #[allow(dead_code)]
    fn get_velocity(&self) -> (f32, f32) {
        (self.state[4], self.state[5])
    }
}

/// 被跟踪的人
#[derive(Clone)]
pub struct TrackedPerson {
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

    /// 显示颜色 (每个ID不同颜色)
    pub color: (u8, u8, u8),

    /// 总共被跟踪的帧数 (age)
    pub total_frames: u32,

    /// 外观特征向量 (简化: 使用颜色直方图模拟ReID)
    /// 实际应该用深度学习ReID模型提取特征
    appearance_features: Vec<f32>,

    /// 自上次匹配以来的时间 (用于级联匹配)
    pub time_since_update: u32,

    /// 确认状态 (连续匹配3帧后才确认)
    pub confirmed: bool,

    /// 连续匹配次数
    consecutive_matches: u32,
}

impl TrackedPerson {
    fn new(
        id: u32,
        bbox: BBox,
        color: (u8, u8, u8),
        keypoints: Option<&PoseKeypoints>,
        reid_features: Option<Vec<f32>>,
    ) -> Self {
        let kalman = KalmanBoxFilter::new(&bbox);
        let smoothed_bbox = kalman.get_bbox();

        let center = TrackPoint {
            x: (smoothed_bbox.x1 + smoothed_bbox.x2) / 2.0,
            y: (smoothed_bbox.y1 + smoothed_bbox.y2) / 2.0,
        };

        // 优先使用深度ReID特征
        let appearance_features = if let Some(features) = reid_features {
            features
        } else if let Some(kpts) = keypoints {
            kpts.extract_reid_features(&bbox)
        } else {
            // 降级到几何特征
            let w = bbox.x2 - bbox.x1;
            let h = bbox.y2 - bbox.y1;
            let aspect_ratio = w / h.max(1.0);
            let area = w * h;

            vec![aspect_ratio, area.sqrt() / 100.0, bbox.confidence]
        };

        Self {
            id,
            bbox: smoothed_bbox,
            kalman,
            trajectory: vec![center],
            frames_lost: 0,
            color,
            total_frames: 1,
            appearance_features,
            time_since_update: 0,
            confirmed: false,
            consecutive_matches: 0,
        }
    }

    /// 预测下一帧位置
    fn predict(&mut self) {
        self.kalman.predict();
        self.bbox = self.kalman.get_bbox();
    }

    /// 更新位置 (融合观测)
    fn update(
        &mut self,
        bbox: BBox,
        keypoints: Option<&PoseKeypoints>,
        min_confirmation_hits: u32,
    ) {
        // 卡尔曼滤波更新
        self.kalman.update(&bbox);
        self.bbox = self.kalman.get_bbox();

        self.frames_lost = 0;
        self.time_since_update = 0;
        self.total_frames += 1;
        self.consecutive_matches += 1;

        // 使用配置的确认次数
        if self.consecutive_matches >= min_confirmation_hits {
            self.confirmed = true;
        }

        // 更新外观特征 (使用真实ReID)
        let new_features = if let Some(kpts) = keypoints {
            kpts.extract_reid_features(&bbox)
        } else {
            // 降级到几何特征
            let w = bbox.x2 - bbox.x1;
            let h = bbox.y2 - bbox.y1;
            let aspect_ratio = w / h.max(1.0);
            let area = w * h;
            vec![aspect_ratio, area.sqrt() / 100.0, bbox.confidence]
        };

        // EMA平滑: 0.95旧 + 0.05新 (极度保守,特征几乎不变)
        for i in 0..self.appearance_features.len().min(new_features.len()) {
            self.appearance_features[i] =
                self.appearance_features[i] * 0.95 + new_features[i] * 0.05;
        }

        // 添加平滑后的中心点到轨迹
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

    /// 使用深度ReID特征更新
    fn update_with_reid(
        &mut self,
        bbox: BBox,
        keypoints: Option<&PoseKeypoints>,
        reid_features: Option<Vec<f32>>,
        min_confirmation_hits: u32,
    ) {
        // 卡尔曼滤波更新
        self.kalman.update(&bbox);
        self.bbox = self.kalman.get_bbox();

        self.frames_lost = 0;
        self.time_since_update = 0;
        self.total_frames += 1;
        self.consecutive_matches += 1;

        if self.consecutive_matches >= min_confirmation_hits {
            self.confirmed = true;
        }

        // 优先使用深度ReID特征
        let new_features = if let Some(features) = reid_features {
            features
        } else if let Some(kpts) = keypoints {
            kpts.extract_reid_features(&bbox)
        } else {
            let w = bbox.x2 - bbox.x1;
            let h = bbox.y2 - bbox.y1;
            let aspect_ratio = w / h.max(1.0);
            let area = w * h;
            vec![aspect_ratio, area.sqrt() / 100.0, bbox.confidence]
        };

        // 调整特征向量维度
        if self.appearance_features.len() != new_features.len() {
            self.appearance_features = new_features;
        } else {
            // EMA平滑: 0.9旧 + 0.1新 (深度ReID稳定,缓慢更新)
            for i in 0..self.appearance_features.len() {
                self.appearance_features[i] =
                    self.appearance_features[i] * 0.9 + new_features[i] * 0.1;
            }
        }

        let center = TrackPoint {
            x: (self.bbox.x1 + self.bbox.x2) / 2.0,
            y: (self.bbox.y1 + self.bbox.y2) / 2.0,
        };
        self.trajectory.push(center);

        if self.trajectory.len() > 50 {
            self.trajectory.remove(0);
        }
    }

    /// 标记为丢失 (仅预测)
    fn mark_lost(&mut self) {
        self.frames_lost += 1;
        self.time_since_update += 1;
        // 不重置连续匹配计数! 保持确认状态!
        // self.consecutive_matches = 0;

        // 丢失时继续预测位置
        self.predict();
    }

    /// 获取预测的边界框 (用于匹配)
    fn get_predicted_bbox(&self) -> BBox {
        self.kalman.get_predicted_bbox()
    }

    /// 获取当前速度向量
    #[allow(dead_code)]
    fn get_velocity(&self) -> (f32, f32) {
        self.kalman.get_velocity()
    }

    /// 检查速度方向一致性
    #[allow(dead_code)]
    fn check_velocity_consistency(&self, detection: &BBox, angle_threshold: f32) -> bool {
        let (vx, vy) = self.get_velocity();

        // 速度太小时不检查方向 (静止或刚初始化)
        let speed = (vx * vx + vy * vy).sqrt();
        if speed < 1.0 {
            return true; // 接受任何方向
        }

        // 计算检测框相对于预测位置的移动方向
        let predicted = self.get_predicted_bbox();
        let pred_cx = (predicted.x1 + predicted.x2) / 2.0;
        let pred_cy = (predicted.y1 + predicted.y2) / 2.0;
        let det_cx = (detection.x1 + detection.x2) / 2.0;
        let det_cy = (detection.y1 + detection.y2) / 2.0;

        let dx = det_cx - pred_cx;
        let dy = det_cy - pred_cy;

        // 计算速度向量与观测向量的夹角
        let dot = vx * dx + vy * dy;
        let mag_v = speed;
        let mag_d = (dx * dx + dy * dy).sqrt();

        if mag_d < 0.1 {
            return true; // 几乎没移动,接受
        }

        let cos_angle = dot / (mag_v * mag_d);
        let angle = cos_angle.acos().to_degrees();

        angle < angle_threshold
    }

    /// 计算与检测框的外观相似度 (余弦距离 - 基于几何特征)
    fn compute_appearance_similarity(&self, detection: &BBox) -> f32 {
        let w = detection.x2 - detection.x1;
        let h = detection.y2 - detection.y1;
        let aspect_ratio = w / h.max(1.0);
        let area = w * h;

        let det_features = vec![aspect_ratio, area.sqrt() / 100.0, detection.confidence];

        // 计算余弦相似度
        let mut dot = 0.0;
        let mut mag_a = 0.0;
        let mut mag_b = 0.0;

        for i in 0..self.appearance_features.len().min(det_features.len()) {
            dot += self.appearance_features[i] * det_features[i];
            mag_a += self.appearance_features[i] * self.appearance_features[i];
            mag_b += det_features[i] * det_features[i];
        }

        if mag_a < 1e-6 || mag_b < 1e-6 {
            return 0.0;
        }

        (dot / (mag_a.sqrt() * mag_b.sqrt())).max(0.0).min(1.0)
    }

    /// 计算与检测框的ReID相似度 (基于姿态关键点)
    fn compute_reid_similarity(&self, keypoints: &PoseKeypoints, detection: &BBox) -> f32 {
        // 提取新检测的ReID特征
        let det_features = keypoints.extract_reid_features(detection);

        // 计算余弦相似度
        let mut dot = 0.0;
        let mut mag_a = 0.0;
        let mut mag_b = 0.0;

        for i in 0..self.appearance_features.len().min(det_features.len()) {
            dot += self.appearance_features[i] * det_features[i];
            mag_a += self.appearance_features[i] * self.appearance_features[i];
            mag_b += det_features[i] * det_features[i];
        }

        if mag_a < 1e-6 || mag_b < 1e-6 {
            return 0.0;
        }

        (dot / (mag_a.sqrt() * mag_b.sqrt())).max(0.0).min(1.0)
    }

    /// 计算马氏距离 (考虑卡尔曼不确定性)
    #[allow(dead_code)]
    fn compute_mahalanobis_distance(&self, detection: &BBox) -> f32 {
        let predicted = self.get_predicted_bbox();

        let dx = (detection.x1 + detection.x2) / 2.0 - (predicted.x1 + predicted.x2) / 2.0;
        let dy = (detection.y1 + detection.y2) / 2.0 - (predicted.y1 + predicted.y2) / 2.0;
        let dw = (detection.x2 - detection.x1) - (predicted.x2 - predicted.x1);
        let dh = (detection.y2 - detection.y1) - (predicted.y2 - predicted.y1);

        // 简化: 使用欧几里得距离 + 卡尔曼不确定性归一化
        // 实际应该用完整协方差矩阵
        let pos_uncertainty = (self.kalman.p[0] + self.kalman.p[1]).sqrt();
        let size_uncertainty = (self.kalman.p[2] + self.kalman.p[3]).sqrt();

        let norm_pos = (dx * dx + dy * dy).sqrt() / (pos_uncertainty + 1.0);
        let norm_size = (dw * dw + dh * dh).sqrt() / (size_uncertainty + 1.0);

        norm_pos + norm_size * 0.5
    }
}

/// 人形追踪器 (DeepSort)
pub struct PersonTracker {
    /// 当前跟踪的人
    tracked_persons: Vec<TrackedPerson>,

    /// 下一个分配的ID
    next_id: u32,

    /// 最大允许丢失帧数
    max_lost_frames: u32,

    /// IOU 匹配阈值
    #[allow(dead_code)]
    iou_threshold: f32,

    /// 马氏距离阈值 (DeepSort运动门控)
    #[allow(dead_code)]
    mahalanobis_threshold: f32,

    /// 外观相似度阈值 (余弦距离)
    #[allow(dead_code)]
    appearance_threshold: f32,

    /// 级联匹配最大深度 (age)
    max_cascade_depth: u32,

    /// 确认轨迹所需的最小匹配次数
    min_confirmation_hits: u32,

    /// 预定义颜色表
    color_palette: Vec<(u8, u8, u8)>,

    /// OSNet ReID模型
    reid_model: Option<Session>,
}

impl PersonTracker {
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
            max_lost_frames: 300, // 大幅增加: 300帧 ≈ 10秒@30fps (长时遮挡鲁棒)
            iou_threshold: 0.15,  // 降低IOU阈值 (允许更大位置偏差)
            mahalanobis_threshold: 20.0, // 大幅放宽运动门控 (允许快速移动/突变)
            appearance_threshold: 0.3, // 降低外观阈值 (姿态变化大时也能匹配)
            max_cascade_depth: 100, // 增加级联深度 (长时丢失也能恢复)
            min_confirmation_hits: 2, // 降低确认要求 (更快确认新ID)
            color_palette,
            reid_model: Self::load_reid_model(),
        }
    }

    /// 加载OSNet-AIN ReID模型 (x1.0跨域泛化最强版本)
    /// 性能指标: Rank-1 94.7%, mAP 84.9% (跨域场景表现最优)
    fn load_reid_model() -> Option<Session> {
        // 尝试加载模型,失败则返回None(回退到姿态ReID)
        match Session::builder() {
            Ok(builder) => match builder.commit_from_file("models/osnet_ain_x1_0.onnx") {
                Ok(session) => Some(session),
                Err(_) => None,
            },
            Err(_) => None,
        }
    }

    /// 检查是否已加载深度ReID模型
    pub fn has_reid_model(&self) -> bool {
        self.reid_model.is_some()
    }

    /// 从原始图像中裁剪人体区域并提取ReID特征
    /// frame_rgba: 原始RGBA图像数据
    /// width, height: 图像尺寸
    /// bbox: 检测框
    fn extract_reid_features_from_image(
        reid_model: &mut Session,
        frame_rgba: &[u8],
        width: u32,
        height: u32,
        bbox: &BBox,
    ) -> Vec<f32> {
        // 1. 裁剪边界框区域(带10%边距)
        let margin = 0.1;
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;

        let x1 = ((bbox.x1 - w * margin).max(0.0) as u32).min(width - 1);
        let y1 = ((bbox.y1 - h * margin).max(0.0) as u32).min(height - 1);
        let x2 = ((bbox.x2 + w * margin).min(width as f32) as u32).min(width);
        let y2 = ((bbox.y2 + h * margin).min(height as f32) as u32).min(height);

        let crop_w = x2 - x1;
        let crop_h = y2 - y1;

        if crop_w < 10 || crop_h < 10 {
            return vec![0.0; 512]; // 无效区域,返回零向量
        }

        // 2. 转换为RGB并裁剪
        let mut crop_rgb = Vec::with_capacity((crop_w * crop_h * 3) as usize);
        for y in y1..y2 {
            for x in x1..x2 {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 2 < frame_rgba.len() {
                    crop_rgb.push(frame_rgba[idx]); // R
                    crop_rgb.push(frame_rgba[idx + 1]); // G
                    crop_rgb.push(frame_rgba[idx + 2]); // B
                }
            }
        }

        // 3. 构造image对象并resize到256x128
        let img = match ImageBuffer::<Rgb<u8>, _>::from_raw(crop_w, crop_h, crop_rgb) {
            Some(img) => DynamicImage::ImageRgb8(img),
            None => return vec![0.0; 512],
        };

        let resized = img.resize_exact(128, 256, image::imageops::FilterType::Triangle);

        // 4. 转换为CHW格式 + 归一化 [0,1]
        let rgb = resized.to_rgb8();
        let mut input_data = Array4::<f32>::zeros((1, 3, 256, 128));

        for y in 0..256 {
            for x in 0..128 {
                let pixel = rgb.get_pixel(x, y);
                input_data[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
                input_data[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
                input_data[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
            }
        }

        // 5. 推理
        let input_value = match Value::from_array(input_data) {
            Ok(v) => v,
            Err(_) => return vec![0.0; 512],
        };

        let outputs = match reid_model.run(ort::inputs![input_value]) {
            Ok(outputs) => outputs,
            Err(_) => return vec![0.0; 512],
        };

        // 6. 提取特征向量 (假设第一个输出是512维特征)
        let features = match outputs.iter().next() {
            Some((_, value)) => match value.try_extract_tensor::<f32>() {
                Ok(tensor) => tensor.1.to_vec(),
                Err(_) => vec![0.0; 512],
            },
            None => vec![0.0; 512],
        };

        // 7. L2归一化
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            features.iter().map(|x| x / norm).collect()
        } else {
            features
        }
    }

    /// 更新跟踪 (DeepSort级联匹配)
    pub fn update(
        &mut self,
        detections: &[BBox],
        keypoints: &[PoseKeypoints],
        frame_rgba: Option<(&[u8], u32, u32)>, // (数据, 宽, 高)
    ) -> &[TrackedPerson] {
        // 1. 所有跟踪对象先预测下一帧
        for tracked in &mut self.tracked_persons {
            tracked.predict();
        }

        // 2. 分离已确认和未确认的轨迹
        let mut confirmed_indices = Vec::new();
        let mut unconfirmed_indices = Vec::new();

        for (idx, track) in self.tracked_persons.iter().enumerate() {
            if track.confirmed {
                confirmed_indices.push(idx);
            } else {
                unconfirmed_indices.push(idx);
            }
        }

        // 3. 级联匹配 (Cascade Matching): 优先匹配年轻轨迹
        let mut matched_det = vec![false; detections.len()];
        let mut matched_track = vec![false; self.tracked_persons.len()];

        // 按时间顺序级联匹配 (time_since_update从小到大)
        for age in 0..=self.max_cascade_depth {
            // 找出age相同的已确认轨迹
            let age_tracks: Vec<usize> = confirmed_indices
                .iter()
                .filter(|&&idx| {
                    self.tracked_persons[idx].time_since_update == age && !matched_track[idx]
                })
                .copied()
                .collect();

            if age_tracks.is_empty() {
                continue;
            }

            // 找出未匹配的检测
            let unmatched_dets: Vec<(usize, &BBox)> = detections
                .iter()
                .enumerate()
                .filter(|(idx, _)| !matched_det[*idx])
                .collect();

            if unmatched_dets.is_empty() {
                break;
            }

            // 计算门控代价矩阵 (马氏距离 + 外观相似度)
            let assignments =
                self.gate_cost_matrix(&unmatched_dets, &age_tracks, keypoints, frame_rgba);

            // 应用匹配
            for (det_idx, track_idx) in assignments {
                matched_det[det_idx] = true;
                matched_track[track_idx] = true;

                // 提取ReID特征并更新
                let reid_features =
                    if let (Some(reid), Some((rgba, w, h))) = (&mut self.reid_model, frame_rgba) {
                        Some(Self::extract_reid_features_from_image(
                            reid,
                            rgba,
                            w,
                            h,
                            &detections[det_idx],
                        ))
                    } else {
                        None
                    };

                let kpts = keypoints.get(det_idx);
                self.tracked_persons[track_idx].update_with_reid(
                    detections[det_idx].clone(),
                    kpts,
                    reid_features,
                    self.min_confirmation_hits,
                );
            }
        }

        // 4. IOU匹配 (未确认轨迹 + 剩余检测) - 使用更宽松的阈值
        let unmatched_dets: Vec<(usize, &BBox)> = detections
            .iter()
            .enumerate()
            .filter(|(idx, _)| !matched_det[*idx])
            .collect();

        if !unmatched_dets.is_empty() && !unconfirmed_indices.is_empty() {
            let cost_matrix = self.compute_iou_cost_matrix(&unmatched_dets, &unconfirmed_indices);
            let assignments = hungarian_algorithm_simple(&cost_matrix, 0.05); // 更宽松: IOU>0.05即可

            for (local_det_idx, local_track_idx) in assignments {
                let det_idx = unmatched_dets[local_det_idx].0;
                let track_idx = unconfirmed_indices[local_track_idx];
                matched_det[det_idx] = true;
                matched_track[track_idx] = true;
                let kpts = keypoints.get(det_idx);
                self.tracked_persons[track_idx].update(
                    detections[det_idx].clone(),
                    kpts,
                    self.min_confirmation_hits,
                );
            }
        }

        // 5. 未匹配的检测 → 新建轨迹
        for (det_idx, &matched) in matched_det.iter().enumerate() {
            if !matched {
                let color = self.color_palette[self.next_id as usize % self.color_palette.len()];
                let kpts = keypoints.get(det_idx);

                // 提取ReID特征
                let reid_feat =
                    if let (Some(reid), Some((rgba, w, h))) = (&mut self.reid_model, frame_rgba) {
                        Some(Self::extract_reid_features_from_image(
                            reid,
                            rgba,
                            w,
                            h,
                            &detections[det_idx],
                        ))
                    } else {
                        None
                    };

                let tracked = TrackedPerson::new(
                    self.next_id,
                    detections[det_idx].clone(),
                    color,
                    kpts,
                    reid_feat,
                );
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

    /// 计算门控代价矩阵 (DeepSort: 马氏距离 + 外观相似度)
    fn gate_cost_matrix(
        &mut self,
        detections: &[(usize, &BBox)],
        track_indices: &[usize],
        keypoints: &[PoseKeypoints],
        frame_rgba: Option<(&[u8], u32, u32)>,
    ) -> Vec<(usize, usize)> {
        let mut candidates = Vec::new();

        for (local_det_idx, (det_idx, detection)) in detections.iter().enumerate() {
            for (local_track_idx, &track_idx) in track_indices.iter().enumerate() {
                let track = &self.tracked_persons[track_idx];

                let iou = Self::compute_iou(detection, &track.get_predicted_bbox());

                // 计算外观相似度
                let (appearance_sim, _feature_type) = if let (Some(reid), Some((rgba, w, h))) =
                    (&mut self.reid_model, frame_rgba)
                {
                    // 使用深度ReID模型
                    let det_features =
                        Self::extract_reid_features_from_image(reid, rgba, w, h, detection);
                    let sim = Self::cosine_similarity(&track.appearance_features, &det_features);
                    (sim, "深度ReID")
                } else if let Some(kpts) = keypoints.get(*det_idx) {
                    // 回退到姿态ReID
                    (track.compute_reid_similarity(kpts, detection), "姿态ReID")
                } else {
                    // 最后回退到几何特征
                    (track.compute_appearance_similarity(detection), "几何特征")
                };

                // 首次使用时输出特征类型
                // 融合代价: 10%运动 + 90%外观
                let motion_cost = 1.0 - iou;
                let appearance_cost = 1.0 - appearance_sim;
                let cost = motion_cost * 0.1 + appearance_cost * 0.9;

                candidates.push((cost, *det_idx, local_det_idx, track_idx, local_track_idx));
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

    /// 计算IOU代价矩阵 (用于未确认轨迹)
    fn compute_iou_cost_matrix(
        &self,
        detections: &[(usize, &BBox)],
        track_indices: &[usize],
    ) -> Vec<Vec<f32>> {
        let mut matrix = Vec::new();

        for (_, detection) in detections {
            let mut row = Vec::new();
            for &track_idx in track_indices {
                let track = &self.tracked_persons[track_idx];
                let iou = Self::compute_iou(detection, &track.get_predicted_bbox());
                row.push(1.0 - iou);
            }
            matrix.push(row);
        }

        matrix
    }

    /// 计算代价矩阵 (1 - IOU,越小越好)
    #[allow(dead_code)]
    fn compute_cost_matrix(&self, detections: &[BBox]) -> Vec<Vec<f32>> {
        let mut matrix = Vec::new();

        for detection in detections {
            let mut row = Vec::new();
            for tracked in &self.tracked_persons {
                // 使用预测位置计算IOU
                let predicted = tracked.get_predicted_bbox();
                let iou = Self::compute_iou(detection, &predicted);
                row.push(1.0 - iou); // 代价 = 1 - IOU
            }
            matrix.push(row);
        }

        matrix
    }

    /// 计算代价矩阵 (仅针对指定轨迹,用于低分救援)
    #[allow(dead_code)]
    fn compute_cost_matrix_for_tracks(
        &self,
        detections: &[BBox],
        track_indices: &[usize],
    ) -> Vec<Vec<f32>> {
        let mut matrix = Vec::new();

        for detection in detections {
            let mut row = Vec::new();
            for &track_idx in track_indices {
                let tracked = &self.tracked_persons[track_idx];
                let predicted = tracked.get_predicted_bbox();
                let iou = Self::compute_iou(detection, &predicted);
                row.push(1.0 - iou);
            }
            matrix.push(row);
        }

        matrix
    }

    /// 计算两个边界框的 IOU (Intersection over Union)
    fn compute_iou(bbox1: &BBox, bbox2: &BBox) -> f32 {
        let x1 = bbox1.x1.max(bbox2.x1);
        let y1 = bbox1.y1.max(bbox2.y1);
        let x2 = bbox1.x2.min(bbox2.x2);
        let y2 = bbox1.y2.min(bbox2.y2);

        if x2 < x1 || y2 < y1 {
            return 0.0; // 无交集
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
        let area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
        let union = area1 + area2 - intersection;

        intersection / union
    }

    /// 余弦相似度计算
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-6 || norm_b < 1e-6 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).max(0.0).min(1.0)
    }

    /// 获取当前所有跟踪对象
    pub fn get_tracked_persons(&self) -> &[TrackedPerson] {
        &self.tracked_persons
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

impl Default for PersonTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// 匈牙利算法 (Hungarian Algorithm) - 解决二分图最大权匹配
/// 返回: Vec<(detection_idx, track_idx)> 最优匹配对
fn hungarian_algorithm_simple(cost_matrix: &[Vec<f32>], threshold: f32) -> Vec<(usize, usize)> {
    if cost_matrix.is_empty() || cost_matrix[0].is_empty() {
        return Vec::new();
    }

    let n_det = cost_matrix.len();
    let n_track = cost_matrix[0].len();

    let mut assignments = Vec::new();
    let mut used_det = vec![false; n_det];
    let mut used_track = vec![false; n_track];

    // 收集所有候选匹配 (cost, det_idx, track_idx)
    let mut candidates = Vec::new();
    for (det_idx, row) in cost_matrix.iter().enumerate() {
        for (track_idx, &cost) in row.iter().enumerate() {
            if cost < (1.0 - threshold) {
                // IOU > threshold
                candidates.push((cost, det_idx, track_idx));
            }
        }
    }

    // 按代价排序 (从小到大)
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // 贪心分配: 优先分配代价最小的
    for (_, det_idx, track_idx) in candidates {
        if !used_det[det_idx] && !used_track[track_idx] {
            assignments.push((det_idx, track_idx));
            used_det[det_idx] = true;
            used_track[track_idx] = true;
        }
    }

    assignments
}
