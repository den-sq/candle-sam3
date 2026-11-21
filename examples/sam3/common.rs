//! Device and image helpers for the examples, from Candle's `candle-examples`
//! (https://github.com/huggingface/candle, MIT OR Apache-2.0).

use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Result, Tensor};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        Ok(Device::Cpu)
    }
}

pub fn load_image<P: AsRef<std::path::Path>>(
    p: P,
    resize_longest: Option<usize>,
) -> Result<(Tensor, usize, usize)> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?;
    let (initial_h, initial_w) = (img.height() as usize, img.width() as usize);
    let img = match resize_longest {
        None => img,
        Some(resize_longest) => {
            let (height, width) = (img.height(), img.width());
            let resize_longest = resize_longest as u32;
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            img.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        }
    };
    let (height, width) = (img.height() as usize, img.width() as usize);
    let data = img.to_rgb8().into_raw();
    Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))
        .map(|data| (data, initial_h, initial_w))
}
