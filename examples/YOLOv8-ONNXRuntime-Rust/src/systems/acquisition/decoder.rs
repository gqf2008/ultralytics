/// 解码器 (Decoder)
/// 职责: RTSP视频流解码 → 发送DecodedFrame消息

use crate::rtsp;
use crate::systems::DecodedFrame;
use crate::xbus;
use std::sync::mpsc;

pub struct Decoder {
    rtsp_url: String,
}

impl Decoder {
    pub fn new(rtsp_url: String) -> Self {
        Self { rtsp_url }
    }

    pub fn run(&mut self) {
        println!("🎬 解码器启动");
        let (tx, rx) = mpsc::channel();
        let filter = rtsp::DecodeFilter::new(tx);
        let rtsp_url = self.rtsp_url.clone();
        std::thread::spawn(move || {
            rtsp::adaptive_decode(&rtsp_url, filter);
        });
        let mut frame_id = 0u64;
        while let Ok(frame) = rx.recv() {
            frame_id += 1;
            let xbus_frame = DecodedFrame {
                rgba_data: frame.rgba_data,
                width: frame.width,
                height: frame.height,
                frame_id,
                decode_fps: frame.decode_fps,
                decoder_name: frame.decoder_name,
            };
            xbus::post(xbus_frame);
        }
        println!("❌ 解码器退出");
    }
}
