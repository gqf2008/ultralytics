use std::sync::Arc;
/// RTSP检测系统数据结构定义
/// Data structures for RTSP detection system

// ========== 公共常量 ==========

/// YOLOv8推理输入尺寸
pub const INF_SIZE: u32 = 640;

// ========== 枚举类型 ==========

/// 追踪器类型
#[derive(Debug, Clone, Copy)]
pub enum TrackerType {
    DeepSort,
    ByteTrack,
}

// ========== 数据结构 ==========

/// 检测框 (Detection bounding box)
#[derive(Clone, Debug)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u32,
}

/// 姿态关键点 (Pose keypoints)
#[derive(Clone, Debug)]
pub struct PoseKeypoints {
    pub points: Vec<(f32, f32, f32)>, // (x, y, confidence)
}

/// 已解码帧 (解码线程 → 渲染线程)
#[derive(Clone)]
pub struct DecodedFrame {
    pub rgba_data: Arc<Vec<u8>>, // 使用Arc共享数据,避免复制
    pub width: u32,
    pub height: u32,
    pub decode_fps: f64,
    pub decoder_name: String, // 使用的解码器名称
}

/// 缩放后的帧 (渲染线程 → 推理线程)
#[derive(Clone)]
pub struct ResizedFrame {
    pub rgb_data: Vec<u8>, // 320x320 RGB data from GPU resize
}

/// 推理结果 (推理线程 → 渲染线程)
#[derive(Clone)]
pub struct InferredFrame {
    pub bboxes: Vec<BBox>,
    pub keypoints: Vec<PoseKeypoints>,
    pub inference_fps: f64,
    pub inference_ms: f64,
}

/// 配置更新消息 (渲染线程 → 推理线程)
#[derive(Clone, Debug)]
pub enum ConfigMessage {
    UpdateParams {
        conf_threshold: f32,
        iou_threshold: f32,
    },
    SwitchModel(String),
    SwitchTracker(String),
    TogglePose(bool),
    ToggleDetection(bool),
}

impl PoseKeypoints {
    /// 提取ReID特征向量 (基于姿态关键点)
    /// 返回64维特征向量
    pub fn extract_reid_features(&self, bbox: &BBox) -> Vec<f32> {
        let mut features = Vec::with_capacity(64);

        if self.points.is_empty() {
            // 无关键点时返回零向量
            return vec![0.0; 64];
        }

        // 归一化关键点到边界框坐标系 (0-1)
        let bbox_w = (bbox.x2 - bbox.x1).max(1.0);
        let bbox_h = (bbox.y2 - bbox.y1).max(1.0);

        let normalized_points: Vec<(f32, f32, f32)> = self
            .points
            .iter()
            .map(|(x, y, c)| {
                let norm_x = (x - bbox.x1) / bbox_w;
                let norm_y = (y - bbox.y1) / bbox_h;
                (norm_x, norm_y, *c)
            })
            .collect();

        // 1. 关键点位置特征 (17个关键点 × 2坐标 = 34维)
        for (x, y, conf) in &normalized_points {
            if *conf > 0.3 {
                // 只使用高置信度关键点
                features.push(*x);
                features.push(*y);
            } else {
                features.push(0.0);
                features.push(0.0);
            }
        }

        // 补齐到34维
        while features.len() < 34 {
            features.push(0.0);
        }
        features.truncate(34);

        // 2. 骨骼长度比例特征 (10维)
        // COCO 17关键点: 0-鼻子, 5-左肩, 6-右肩, 11-左髋, 12-右髋, 13-左膝, 14-右膝, 15-左脚踝, 16-右脚踝
        let bone_features = vec![
            Self::bone_length(&normalized_points, 5, 6),   // 肩宽
            Self::bone_length(&normalized_points, 11, 12), // 髋宽
            Self::bone_length(&normalized_points, 5, 11),  // 左身体长
            Self::bone_length(&normalized_points, 6, 12),  // 右身体长
            Self::bone_length(&normalized_points, 11, 13), // 左大腿
            Self::bone_length(&normalized_points, 12, 14), // 右大腿
            Self::bone_length(&normalized_points, 13, 15), // 左小腿
            Self::bone_length(&normalized_points, 14, 16), // 右小腿
            Self::aspect_ratio(&normalized_points),        // 整体长宽比
            Self::center_of_mass_y(&normalized_points),    // 重心高度
        ];
        features.extend(bone_features);

        // 3. 姿态角度特征 (10维)
        let angle_features = vec![
            Self::angle_3points(&normalized_points, 5, 0, 6), // 肩-头-肩角度
            Self::angle_3points(&normalized_points, 11, 5, 6), // 左肩夹角
            Self::angle_3points(&normalized_points, 12, 6, 5), // 右肩夹角
            Self::angle_3points(&normalized_points, 5, 11, 13), // 左髋夹角
            Self::angle_3points(&normalized_points, 6, 12, 14), // 右髋夹角
            Self::angle_3points(&normalized_points, 11, 13, 15), // 左膝夹角
            Self::angle_3points(&normalized_points, 12, 14, 16), // 右膝夹角
            Self::body_orientation(&normalized_points),       // 身体朝向
            Self::pose_symmetry(&normalized_points),          // 左右对称性
            Self::keypoint_density(&normalized_points),       // 关键点密度
        ];
        features.extend(angle_features);

        // 4. 外观特征 (10维) - 基于边界框
        let appearance_features = vec![
            bbox_w / bbox_h.max(1.0),                    // 长宽比
            (bbox_w * bbox_h).sqrt() / 100.0,            // 面积 (归一化)
            bbox.confidence,                             // 检测置信度
            Self::avg_keypoint_conf(&normalized_points), // 平均关键点置信度
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // 预留颜色特征位
        ];
        features.extend(appearance_features);

        // L2归一化
        let norm = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            features.iter_mut().for_each(|x| *x /= norm);
        }

        features
    }

    // 辅助函数: 计算两点间距离
    fn bone_length(points: &[(f32, f32, f32)], idx1: usize, idx2: usize) -> f32 {
        if idx1 >= points.len() || idx2 >= points.len() {
            return 0.0;
        }
        let (x1, y1, c1) = points[idx1];
        let (x2, y2, c2) = points[idx2];
        if c1 < 0.3 || c2 < 0.3 {
            return 0.0;
        }
        ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
    }

    // 三点夹角
    fn angle_3points(points: &[(f32, f32, f32)], idx1: usize, idx2: usize, idx3: usize) -> f32 {
        if idx1 >= points.len() || idx2 >= points.len() || idx3 >= points.len() {
            return 0.0;
        }
        let (x1, y1, c1) = points[idx1];
        let (x2, y2, c2) = points[idx2];
        let (x3, y3, c3) = points[idx3];
        if c1 < 0.3 || c2 < 0.3 || c3 < 0.3 {
            return 0.0;
        }

        let v1x = x1 - x2;
        let v1y = y1 - y2;
        let v2x = x3 - x2;
        let v2y = y3 - y2;

        let dot = v1x * v2x + v1y * v2y;
        let mag1 = (v1x * v1x + v1y * v1y).sqrt();
        let mag2 = (v2x * v2x + v2y * v2y).sqrt();

        if mag1 < 1e-6 || mag2 < 1e-6 {
            return 0.0;
        }

        (dot / (mag1 * mag2)).clamp(-1.0, 1.0).acos()
    }

    // 身体长宽比
    fn aspect_ratio(points: &[(f32, f32, f32)]) -> f32 {
        let valid_points: Vec<_> = points.iter().filter(|(_, _, c)| *c > 0.3).collect();

        if valid_points.is_empty() {
            return 1.0;
        }

        let min_x = valid_points
            .iter()
            .map(|(x, _, _)| x)
            .fold(f32::INFINITY, |a, &b| a.min(b));
        let max_x = valid_points
            .iter()
            .map(|(x, _, _)| x)
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_y = valid_points
            .iter()
            .map(|(_, y, _)| y)
            .fold(f32::INFINITY, |a, &b| a.min(b));
        let max_y = valid_points
            .iter()
            .map(|(_, y, _)| y)
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let w = max_x - min_x;
        let h = max_y - min_y;

        if h < 1e-6 {
            return 1.0;
        }
        w / h
    }

    // 重心高度
    fn center_of_mass_y(points: &[(f32, f32, f32)]) -> f32 {
        let valid_points: Vec<_> = points.iter().filter(|(_, _, c)| *c > 0.3).collect();

        if valid_points.is_empty() {
            return 0.5;
        }

        let sum_y: f32 = valid_points.iter().map(|(_, y, _)| y).sum();
        sum_y / valid_points.len() as f32
    }

    // 身体朝向 (基于肩膀连线角度)
    fn body_orientation(points: &[(f32, f32, f32)]) -> f32 {
        if points.len() <= 6 {
            return 0.0;
        }
        let (x1, y1, c1) = points[5]; // 左肩
        let (x2, y2, c2) = points[6]; // 右肩

        if c1 < 0.3 || c2 < 0.3 {
            return 0.0;
        }

        (y2 - y1).atan2(x2 - x1)
    }

    // 左右对称性
    fn pose_symmetry(points: &[(f32, f32, f32)]) -> f32 {
        // 比较左右对应关键点的距离
        let pairs = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)];
        let mut sym_score = 0.0;
        let mut count = 0;

        for (left_idx, right_idx) in pairs {
            if left_idx < points.len() && right_idx < points.len() {
                let (_, _, c1) = points[left_idx];
                let (_, _, c2) = points[right_idx];
                if c1 > 0.3 && c2 > 0.3 {
                    sym_score += (c1 - c2).abs();
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0.0;
        }
        1.0 - (sym_score / count as f32)
    }

    // 关键点密度
    fn keypoint_density(points: &[(f32, f32, f32)]) -> f32 {
        let valid_count = points.iter().filter(|(_, _, c)| *c > 0.3).count();
        valid_count as f32 / points.len() as f32
    }

    // 平均关键点置信度
    fn avg_keypoint_conf(points: &[(f32, f32, f32)]) -> f32 {
        if points.is_empty() {
            return 0.0;
        }
        let sum: f32 = points.iter().map(|(_, _, c)| c).sum();
        sum / points.len() as f32
    }
}
