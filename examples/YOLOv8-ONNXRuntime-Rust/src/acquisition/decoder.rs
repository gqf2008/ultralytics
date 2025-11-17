/// 解码器 (Decoder)
/// 职责: RTSP视频流解码 → 发送DecodedFrame消息
use crate::rtsp;

pub struct Decoder {
    rtsp_url: String,
}

impl Decoder {
    pub fn new(rtsp_url: String) -> Self {
        Self { rtsp_url }
    }

    pub fn run(&mut self) {
        println!("🎬 解码器启动");
        let filter = rtsp::DecodeFilter::new();
        let rtsp_url = self.rtsp_url.clone();
        rtsp::adaptive_decode(&rtsp_url, filter);
        println!("❌ 解码器退出");
    }
}
