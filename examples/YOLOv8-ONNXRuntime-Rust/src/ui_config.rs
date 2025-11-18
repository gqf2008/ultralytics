//! 跟踪器配置 - 通过JSON文件调整参数

use serde::{Deserialize, Serialize};
use std::fs;

/// 跟踪器参数配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackerConfig {
    // === 检测参数 ===
    pub detection_conf_threshold: f32, // 检测置信度阈值
    pub detection_iou_threshold: f32,  // NMS IOU阈值
    pub post_process_threshold: f32,   // 后处理置信度过滤

    // === ByteTrack参数 ===
    pub bytetrack_max_lost_frames: u32,      // 最大丢失帧数
    pub bytetrack_high_score_threshold: f32, // 高分阈值
    pub bytetrack_low_score_threshold: f32,  // 低分阈值
    pub bytetrack_high_iou_threshold: f32,   // 高分IOU阈值
    pub bytetrack_low_iou_threshold: f32,    // 低分IOU阈值
    pub bytetrack_kalman_obs_noise: f32,     // 卡尔曼观测噪声

    // === DeepSort参数 ===
    pub deepsort_max_lost_frames: u32,      // 最大丢失帧数
    pub deepsort_iou_threshold: f32,        // IOU阈值
    pub deepsort_appearance_threshold: f32, // 外观相似度阈值
    pub deepsort_reid_skip_frames: u32,     // ReID跳帧间隔
    pub deepsort_reid_max_count: usize,     // 每帧最大ReID提取数
    pub deepsort_kalman_obs_noise: f32,     // 卡尔曼观测噪声

    // === 卡尔曼滤波参数 ===
    pub kalman_process_noise: f32,        // 过程噪声 q
    pub kalman_velocity_decay: f32,       // 速度衰减
    pub kalman_stationary_threshold: f32, // 静止判定阈值(像素)
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            // 检测参数
            detection_conf_threshold: 0.10,
            detection_iou_threshold: 0.45,
            post_process_threshold: 0.01,

            // ByteTrack
            bytetrack_max_lost_frames: 60,
            bytetrack_high_score_threshold: 0.4,
            bytetrack_low_score_threshold: 0.1,
            bytetrack_high_iou_threshold: 0.4,
            bytetrack_low_iou_threshold: 0.3,
            bytetrack_kalman_obs_noise: 0.5,

            // DeepSort
            deepsort_max_lost_frames: 90,
            deepsort_iou_threshold: 0.2,
            deepsort_appearance_threshold: 0.15,
            deepsort_reid_skip_frames: 3,
            deepsort_reid_max_count: 5,
            deepsort_kalman_obs_noise: 1.5,

            // 卡尔曼滤波
            kalman_process_noise: 0.1,
            kalman_velocity_decay: 0.95,
            kalman_stationary_threshold: 2.0,
        }
    }
}

impl TrackerConfig {
    /// 从JSON文件加载配置
    pub fn load(path: &str) -> Self {
        match fs::read_to_string(path) {
            Ok(json) => match serde_json::from_str(&json) {
                Ok(config) => {
                    println!("✅ 配置已从 {} 加载", path);
                    config
                }
                Err(e) => {
                    eprintln!("⚠️  配置文件解析失败: {}, 使用默认值", e);
                    Self::default()
                }
            },
            Err(_) => {
                println!("📝 配置文件不存在,创建默认配置...");
                let config = Self::default();
                config.save(path);
                config
            }
        }
    }

    /// 保存配置到JSON文件
    pub fn save(&self, path: &str) {
        match serde_json::to_string_pretty(self) {
            Ok(json) => {
                if let Err(e) = fs::write(path, json) {
                    eprintln!("❌ 保存配置失败: {}", e);
                } else {
                    println!("💾 配置已保存到 {}", path);
                }
            }
            Err(e) => eprintln!("❌ 序列化配置失败: {}", e),
        }
    }

    /// 打印当前配置
    pub fn print_summary(&self) {
        println!("\n🎛️  当前跟踪器配置:");
        println!("  检测置信度: {:.2}", self.detection_conf_threshold);
        println!("  ByteTrack最大丢失帧: {}", self.bytetrack_max_lost_frames);
        println!("  DeepSort最大丢失帧: {}", self.deepsort_max_lost_frames);
        println!("  ReID跳帧间隔: {}", self.deepsort_reid_skip_frames);
        println!(
            "  卡尔曼观测噪声(ByteTrack): {:.2}",
            self.bytetrack_kalman_obs_noise
        );
        println!(
            "  卡尔曼观测噪声(DeepSort): {:.2}\n",
            self.deepsort_kalman_obs_noise
        );
    }
}
