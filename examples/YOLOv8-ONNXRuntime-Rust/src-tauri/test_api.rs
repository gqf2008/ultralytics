use windows::Graphics::Capture::IGraphicsCaptureSession2;
fn test(s: &IGraphicsCaptureSession2) {
    s.SetIsBorderRequired(false);
}
