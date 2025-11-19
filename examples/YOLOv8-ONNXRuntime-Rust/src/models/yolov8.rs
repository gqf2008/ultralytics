// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// YOLOv8 完整模型实现
// 包含: 模型加载、预处理、推理、后处理

use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer};
use ndarray::{s, Array, Axis, IxDyn};

use crate::{
    non_max_suppression, Batch, Bbox, DetectionResult, Embedding, OrtBackend, OrtConfig, OrtEP,
    Point2, YOLOTask,
};

/// YOLOv8 完整模型结构
pub struct YOLOv8 {
    engine: OrtBackend,
    nc: u32,
    nk: u32,
    nm: u32,
    height: u32,
    width: u32,
    batch: u32,
    task: YOLOTask,
    conf: f32,
    kconf: f32,
    iou: f32,
    names: Vec<String>,
    color_palette: Vec<(u8, u8, u8)>,
    profile: bool,
}

impl YOLOv8 {
    /// 从配置创建 YOLOv8 模型
    pub fn new(config: crate::Args) -> Result<Self> {
        // execution provider
        let ep = if config.trt {
            OrtEP::Trt(config.device_id)
        } else if config.cuda {
            OrtEP::CUDA(config.device_id)
        } else {
            OrtEP::CPU
        };

        // batch
        let batch = Batch {
            opt: config.batch,
            min: config.batch_min,
            max: config.batch_max,
        };

        // build ort engine
        let ort_args = OrtConfig {
            ep,
            batch,
            f: config.model,
            task: config.task,
            trt_fp16: config.fp16,
            image_size: (config.height, config.width),
        };
        let engine = OrtBackend::build(ort_args)?;

        //  get batch, height, width, tasks, nc, nk, nm
        let (batch, height, width, task) = (
            engine.batch(),
            engine.height(),
            engine.width(),
            engine.task(),
        );
        let nc = engine.nc().or(config.nc).unwrap_or_else(|| {
            panic!("Failed to get num_classes, make it explicit with `--nc`");
        });
        let (nk, nm) = match task {
            YOLOTask::Pose => {
                let nk = engine.nk().or(config.nk).unwrap_or_else(|| {
                    panic!("Failed to get num_keypoints, make it explicit with `--nk`");
                });
                (nk, 0)
            }
            YOLOTask::Segment => {
                let nm = engine.nm().or(config.nm).unwrap_or_else(|| {
                    panic!("Failed to get num_masks, make it explicit with `--nm`");
                });
                (0, nm)
            }
            _ => (0, 0),
        };

        // class names
        let names = engine.names().unwrap_or(vec!["Unknown".to_string()]);

        // color palette
        let bright_colors = vec![
            (255, 0, 0),     // 红色
            (0, 255, 0),     // 绿色
            (0, 0, 255),     // 蓝色
            (255, 255, 0),   // 黄色
            (255, 0, 255),   // 品红
            (0, 255, 255),   // 青色
            (255, 128, 0),   // 橙色
            (255, 0, 128),   // 粉红
            (128, 255, 0),   // 黄绿
            (0, 128, 255),   // 天蓝
            (255, 255, 255), // 白色
            (128, 0, 255),   // 紫色
        ];

        let color_palette: Vec<_> = names
            .iter()
            .enumerate()
            .map(|(i, _)| bright_colors[i % bright_colors.len()])
            .collect();

        Ok(Self {
            engine,
            names,
            conf: config.conf,
            kconf: config.kconf,
            iou: config.iou,
            color_palette,
            profile: config.profile,
            nc,
            nk,
            nm,
            height,
            width,
            batch,
            task,
        })
    }

    fn scale_wh(&self, w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }

    pub fn preprocess(&mut self, xs: &Vec<DynamicImage>) -> Result<Array<f32, IxDyn>> {
        let mut ys =
            Array::ones((xs.len(), 3, self.height() as usize, self.width() as usize)).into_dyn();
        ys.fill(144.0 / 255.0);
        for (idx, x) in xs.iter().enumerate() {
            let img = match self.task() {
                YOLOTask::Classify => x.resize_exact(
                    self.width(),
                    self.height(),
                    image::imageops::FilterType::Triangle,
                ),
                _ => {
                    let (w0, h0) = x.dimensions();
                    let w0 = w0 as f32;
                    let h0 = h0 as f32;
                    let (_, w_new, h_new) =
                        self.scale_wh(w0, h0, self.width() as f32, self.height() as f32);
                    x.resize_exact(
                        w_new as u32,
                        h_new as u32,
                        if let YOLOTask::Segment = self.task() {
                            image::imageops::FilterType::CatmullRom
                        } else {
                            image::imageops::FilterType::Triangle
                        },
                    )
                }
            };

            for (x, y, rgb) in img.pixels() {
                let x = x as usize;
                let y = y as usize;
                let [r, g, b, _] = rgb.0;
                ys[[idx, 0, y, x]] = (r as f32) / 255.0;
                ys[[idx, 1, y, x]] = (g as f32) / 255.0;
                ys[[idx, 2, y, x]] = (b as f32) / 255.0;
            }
        }

        Ok(ys)
    }

    pub fn run(&mut self, xs: &Vec<DynamicImage>) -> Result<Vec<DetectionResult>> {
        let t_pre = std::time::Instant::now();
        let xs_ = self.preprocess(xs)?;
        if self.profile {
            println!("[Model Preprocess]: {:?}", t_pre.elapsed());
        }

        let t_run = std::time::Instant::now();
        let ys = self.engine.run(xs_, self.profile)?;
        if self.profile {
            println!("[Model Inference]: {:?}", t_run.elapsed());
        }

        let t_post = std::time::Instant::now();
        let ys = self.postprocess(ys, xs)?;
        if self.profile {
            println!("[Model Postprocess]: {:?}", t_post.elapsed());
        }

        Ok(ys)
    }

    pub fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<DetectionResult>> {
        if let YOLOTask::Classify = self.task() {
            let mut ys = Vec::new();
            let preds = &xs[0];
            for batch in preds.axis_iter(Axis(0)) {
                ys.push(DetectionResult::new(
                    Some(Embedding::new(batch.into_owned())),
                    None,
                    None,
                    None,
                ));
            }
            Ok(ys)
        } else {
            const CXYWH_OFFSET: usize = 4;
            const KPT_STEP: usize = 3;
            let preds = &xs[0];
            let protos = {
                if xs.len() > 1 {
                    Some(&xs[1])
                } else {
                    None
                }
            };
            let mut ys = Vec::new();
            for (idx, anchor) in preds.axis_iter(Axis(0)).enumerate() {
                let width_original = xs0[idx].width() as f32;
                let height_original = xs0[idx].height() as f32;
                let ratio = (self.width() as f32 / width_original)
                    .min(self.height() as f32 / height_original);

                let mut data: Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> = Vec::new();
                for pred in anchor.axis_iter(Axis(1)) {
                    let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                    let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc() as usize]);
                    let kpts = {
                        if let YOLOTask::Pose = self.task() {
                            Some(pred.slice(s![pred.len() - KPT_STEP * self.nk() as usize..]))
                        } else {
                            None
                        }
                    };
                    let coefs = {
                        if let YOLOTask::Segment = self.task() {
                            Some(pred.slice(s![pred.len() - self.nm() as usize..]).to_vec())
                        } else {
                            None
                        }
                    };

                    let (id, &confidence) = clss
                        .into_iter()
                        .enumerate()
                        .reduce(|max, x| if x.1 > max.1 { x } else { max })
                        .unwrap();

                    if confidence < self.conf {
                        continue;
                    }

                    let cx = bbox[0] / ratio;
                    let cy = bbox[1] / ratio;
                    let w = bbox[2] / ratio;
                    let h = bbox[3] / ratio;
                    let x = cx - w / 2.;
                    let y = cy - h / 2.;
                    let y_bbox = Bbox::new(
                        x.max(0.0f32).min(width_original),
                        y.max(0.0f32).min(height_original),
                        w,
                        h,
                        id,
                        confidence,
                    );

                    let y_kpts = {
                        if let Some(kpts) = kpts {
                            let mut kpts_ = Vec::new();
                            for i in 0..self.nk() as usize {
                                let kx = kpts[KPT_STEP * i] / ratio;
                                let ky = kpts[KPT_STEP * i + 1] / ratio;
                                let kconf = kpts[KPT_STEP * i + 2];
                                if kconf < self.kconf {
                                    kpts_.push(Point2::default());
                                } else {
                                    kpts_.push(Point2::new_with_conf(
                                        kx.max(0.0f32).min(width_original),
                                        ky.max(0.0f32).min(height_original),
                                        kconf,
                                    ));
                                }
                            }
                            Some(kpts_)
                        } else {
                            None
                        }
                    };

                    data.push((y_bbox, y_kpts, coefs));
                }

                non_max_suppression(&mut data, self.iou);

                let mut y_bboxes: Vec<Bbox> = Vec::new();
                let mut y_kpts: Vec<Vec<Point2>> = Vec::new();
                let mut y_masks: Vec<Vec<u8>> = Vec::new();
                for elem in data.into_iter() {
                    if let Some(kpts) = elem.1 {
                        y_kpts.push(kpts)
                    }

                    if let Some(coefs) = elem.2 {
                        let proto = protos.unwrap().slice(s![idx, .., .., ..]);
                        let (nm, nh, nw) = proto.dim();

                        let coefs = Array::from_shape_vec((1, nm), coefs)?;
                        let proto = proto.to_owned();
                        let proto = proto.to_shape((nm, nh * nw))?;
                        let mask = coefs.dot(&proto);
                        let mask = mask.to_shape((nh, nw, 1))?;

                        let mask_im: ImageBuffer<image::Luma<_>, Vec<f32>> =
                            match ImageBuffer::from_raw(
                                nw as u32,
                                nh as u32,
                                mask.to_owned().into_raw_vec_and_offset().0,
                            ) {
                                Some(image) => image,
                                None => panic!("can not create image from ndarray"),
                            };
                        let mut mask_im = image::DynamicImage::from(mask_im);

                        let (_, w_mask, h_mask) =
                            self.scale_wh(width_original, height_original, nw as f32, nh as f32);
                        let mask_cropped = mask_im.crop(0, 0, w_mask as u32, h_mask as u32);
                        let mask_original = mask_cropped.resize_exact(
                            width_original as u32,
                            height_original as u32,
                            match self.task() {
                                YOLOTask::Segment => image::imageops::FilterType::CatmullRom,
                                _ => image::imageops::FilterType::Triangle,
                            },
                        );

                        let mut mask_original_cropped = mask_original.into_luma8();
                        for y in 0..height_original as usize {
                            for x in 0..width_original as usize {
                                if x < elem.0.xmin() as usize
                                    || x > elem.0.xmax() as usize
                                    || y < elem.0.ymin() as usize
                                    || y > elem.0.ymax() as usize
                                {
                                    mask_original_cropped.put_pixel(
                                        x as u32,
                                        y as u32,
                                        image::Luma([0u8]),
                                    );
                                }
                            }
                        }
                        y_masks.push(mask_original_cropped.into_raw());
                    }
                    y_bboxes.push(elem.0);
                }

                let y = DetectionResult {
                    probs: None,
                    bboxes: if !y_bboxes.is_empty() {
                        Some(y_bboxes)
                    } else {
                        None
                    },
                    keypoints: if !y_kpts.is_empty() {
                        Some(y_kpts)
                    } else {
                        None
                    },
                    masks: if !y_masks.is_empty() {
                        Some(y_masks)
                    } else {
                        None
                    },
                };
                ys.push(y);
            }

            Ok(ys)
        }
    }

    pub fn summary(&self) {
        println!(
            "\nSummary:\n\
            > Task: {:?}{}\n\
            > EP: {:?} {}\n\
            > Dtype: {:?}\n\
            > Batch: {} ({}), Height: {} ({}), Width: {} ({})\n\
            > nc: {} nk: {}, nm: {}, conf: {}, kconf: {}, iou: {}\n\
            ",
            self.task(),
            match self.engine.author().zip(self.engine.version()) {
                Some((author, ver)) => format!(" ({} {})", author, ver),
                None => String::from(""),
            },
            self.engine.ep(),
            if let OrtEP::CPU = self.engine.ep() {
                ""
            } else {
                "(May still fall back to CPU)"
            },
            self.engine.dtype(),
            self.batch(),
            if self.engine.is_batch_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.height(),
            if self.engine.is_height_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.width(),
            if self.engine.is_width_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.nc(),
            self.nk(),
            self.nm(),
            self.conf,
            self.kconf,
            self.iou,
        );
    }

    pub fn engine(&self) -> &OrtBackend {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut OrtBackend {
        &mut self.engine
    }

    pub fn conf(&self) -> f32 {
        self.conf
    }

    pub fn set_conf(&mut self, val: f32) {
        self.conf = val;
    }

    pub fn conf_mut(&mut self) -> &mut f32 {
        &mut self.conf
    }

    pub fn kconf(&self) -> f32 {
        self.kconf
    }

    pub fn iou(&self) -> f32 {
        self.iou
    }

    pub fn set_iou(&mut self, val: f32) {
        self.iou = val;
    }

    pub fn task(&self) -> &YOLOTask {
        &self.task
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn nc(&self) -> u32 {
        self.nc
    }

    pub fn nk(&self) -> u32 {
        self.nk
    }

    pub fn nm(&self) -> u32 {
        self.nm
    }

    pub fn names(&self) -> &Vec<String> {
        &self.names
    }

    pub fn color_palette(&self) -> &Vec<(u8, u8, u8)> {
        &self.color_palette
    }
}

// 实现统一的 Model trait
impl super::Model for YOLOv8 {
    fn preprocess(&mut self, images: &[DynamicImage]) -> Result<Vec<Array<f32, IxDyn>>> {
        let batch = YOLOv8::preprocess(self, &images.to_vec())?;
        Ok(vec![batch])
    }

    fn run(&mut self, xs: Vec<Array<f32, IxDyn>>, profile: bool) -> Result<Vec<Array<f32, IxDyn>>> {
        self.engine.run(xs[0].clone(), profile)
    }

    fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<DetectionResult>> {
        YOLOv8::postprocess(self, xs, xs0)
    }

    fn engine_mut(&mut self) -> &mut OrtBackend {
        &mut self.engine
    }

    fn summary(&self) {
        YOLOv8::summary(self)
    }

    fn supports_task(&self, task: YOLOTask) -> bool {
        // YOLOv8 支持所有任务类型
        matches!(
            task,
            YOLOTask::Detect | YOLOTask::Pose | YOLOTask::Segment | YOLOTask::Classify
        )
    }

    fn set_conf(&mut self, val: f32) {
        self.conf = val;
    }

    fn conf(&self) -> f32 {
        self.conf
    }

    fn set_iou(&mut self, val: f32) {
        self.iou = val;
    }

    fn iou(&self) -> f32 {
        self.iou
    }
}

// ========================================
// 向后兼容: YOLOv8Postprocessor (旧版后处理器)
// 用于 detection/postprocessor.rs 等旧代码
// ========================================

/// YOLOv8 配置 (旧版)
pub struct YOLOv8Config {
    pub task: YOLOTask,
    pub nc: usize,
    pub nk: usize,
    pub nm: usize,
    pub conf: f32,
    pub kconf: f32,
    pub iou: f32,
    pub width: usize,
    pub height: usize,
}

impl YOLOv8Config {
    pub fn new(
        task: YOLOTask,
        nc: usize,
        width: usize,
        height: usize,
        conf: f32,
        iou: f32,
    ) -> Self {
        Self {
            task,
            nc,
            nk: 17,
            nm: 32,
            conf,
            kconf: 0.55,
            iou,
            width,
            height,
        }
    }
}

/// YOLOv8 后处理器 (旧版)
pub struct YOLOv8Postprocessor {
    config: YOLOv8Config,
}

impl YOLOv8Postprocessor {
    pub fn new(config: YOLOv8Config) -> Self {
        Self { config }
    }

    fn scale_wh(&self, w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }

    pub fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<DetectionResult>> {
        if let YOLOTask::Classify = self.config.task {
            let mut ys = Vec::new();
            let preds = &xs[0];
            for batch in preds.axis_iter(Axis(0)) {
                ys.push(DetectionResult::new(
                    Some(Embedding::new(batch.into_owned())),
                    None,
                    None,
                    None,
                ));
            }
            return Ok(ys);
        }

        const CXYWH_OFFSET: usize = 4;
        const KPT_STEP: usize = 3;

        let preds = &xs[0];
        let protos = if xs.len() > 1 { Some(&xs[1]) } else { None };

        let mut ys = Vec::new();

        for (idx, anchor) in preds.axis_iter(Axis(0)).enumerate() {
            let width_original = xs0[idx].width() as f32;
            let height_original = xs0[idx].height() as f32;
            let ratio = (self.config.width as f32 / width_original)
                .min(self.config.height as f32 / height_original);

            let mut data: Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> = Vec::new();

            for pred in anchor.axis_iter(Axis(1)) {
                let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.config.nc]);

                let kpts = if let YOLOTask::Pose = self.config.task {
                    Some(pred.slice(s![pred.len() - KPT_STEP * self.config.nk..]))
                } else {
                    None
                };

                let coefs = if let YOLOTask::Segment = self.config.task {
                    Some(pred.slice(s![pred.len() - self.config.nm..]).to_vec())
                } else {
                    None
                };

                let (id, &confidence) = clss
                    .into_iter()
                    .enumerate()
                    .reduce(|max, x| if x.1 > max.1 { x } else { max })
                    .unwrap();

                if confidence < self.config.conf {
                    continue;
                }

                let cx = bbox[0] / ratio;
                let cy = bbox[1] / ratio;
                let w = bbox[2] / ratio;
                let h = bbox[3] / ratio;
                let x = cx - w / 2.;
                let y = cy - h / 2.;

                let y_bbox = Bbox::new(
                    x.max(0.0f32).min(width_original),
                    y.max(0.0f32).min(height_original),
                    w,
                    h,
                    id,
                    confidence,
                );

                let y_kpts = if let Some(kpts) = kpts {
                    let mut kpts_ = Vec::new();
                    for i in 0..self.config.nk {
                        let kx = kpts[KPT_STEP * i] / ratio;
                        let ky = kpts[KPT_STEP * i + 1] / ratio;
                        let kconf = kpts[KPT_STEP * i + 2];

                        if kconf < self.config.kconf {
                            kpts_.push(Point2::default());
                        } else {
                            kpts_.push(Point2::new_with_conf(
                                kx.max(0.0f32).min(width_original),
                                ky.max(0.0f32).min(height_original),
                                kconf,
                            ));
                        }
                    }
                    Some(kpts_)
                } else {
                    None
                };

                data.push((y_bbox, y_kpts, coefs));
            }

            non_max_suppression(&mut data, self.config.iou);

            let mut y_bboxes: Vec<Bbox> = Vec::new();
            let mut y_kpts: Vec<Vec<Point2>> = Vec::new();
            let mut y_masks: Vec<Vec<u8>> = Vec::new();

            for elem in data.into_iter() {
                if let Some(kpts) = elem.1 {
                    y_kpts.push(kpts);
                }

                if let Some(coefs) = elem.2 {
                    let proto = protos.unwrap().slice(s![idx, .., .., ..]);
                    let (nm, nh, nw) = proto.dim();

                    let coefs = Array::from_shape_vec((1, nm), coefs)?;
                    let proto_owned = proto.to_owned();
                    let proto_reshaped = proto_owned.to_shape((nm, nh * nw))?;
                    let mask_dot = coefs.dot(&proto_reshaped);
                    let mask = mask_dot.to_shape((nh, nw, 1))?;

                    let mask_im: ImageBuffer<image::Luma<_>, Vec<f32>> = ImageBuffer::from_raw(
                        nw as u32,
                        nh as u32,
                        mask.to_owned().into_raw_vec_and_offset().0,
                    )
                    .expect("Failed to create mask image");

                    let mut mask_im = image::DynamicImage::from(mask_im);

                    let (_, w_mask, h_mask) =
                        self.scale_wh(width_original, height_original, nw as f32, nh as f32);

                    let mask_cropped = mask_im.crop(0, 0, w_mask as u32, h_mask as u32);
                    let mask_original = mask_cropped.resize_exact(
                        width_original as u32,
                        height_original as u32,
                        match self.config.task {
                            YOLOTask::Segment => image::imageops::FilterType::CatmullRom,
                            _ => image::imageops::FilterType::Triangle,
                        },
                    );

                    let mut mask_original_cropped = mask_original.into_luma8();
                    for y in 0..height_original as usize {
                        for x in 0..width_original as usize {
                            if x < elem.0.xmin() as usize
                                || x > elem.0.xmax() as usize
                                || y < elem.0.ymin() as usize
                                || y > elem.0.ymax() as usize
                            {
                                mask_original_cropped.put_pixel(
                                    x as u32,
                                    y as u32,
                                    image::Luma([0u8]),
                                );
                            }
                        }
                    }
                    y_masks.push(mask_original_cropped.into_raw());
                }

                y_bboxes.push(elem.0);
            }

            let y = DetectionResult {
                probs: None,
                bboxes: if !y_bboxes.is_empty() {
                    Some(y_bboxes)
                } else {
                    None
                },
                keypoints: if !y_kpts.is_empty() {
                    Some(y_kpts)
                } else {
                    None
                },
                masks: if !y_masks.is_empty() {
                    Some(y_masks)
                } else {
                    None
                },
            };
            ys.push(y);
        }

        Ok(ys)
    }
}
