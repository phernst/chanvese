extern crate image;

use super::*;
use std::fs::File;
use super::viridis;

pub fn save_boolgrid(bg: &BoolGrid, outpath: &str) {
    let mut imgbuf = image::ImageBuffer::new(bg.width() as u32, bg.height() as u32);
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        *pixel = image::Luma([if *bg.get(x as usize, y as usize).unwrap() { 255u8 } else { 0u8 }]);
    }
    let ref mut fout = File::create(&outpath).unwrap();
    image::ImageLuma8(imgbuf).save(fout, image::PNG).unwrap();
}

pub fn min_max_scaling(input: &FloatGrid, range: &(f64, f64)) -> FloatGrid {
    let (min_value, max_value) = input.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |acc, (_,_,&v)| {
        (acc.0.min(v), acc.1.max(v))
    });

    if min_value == max_value {
        return FloatGrid::new(input.width(), input.height());
    }

    let mut result = input.clone();
    for (_, _, value) in result.iter_mut() {
        let mut newval = (*value - min_value)/(max_value - min_value);
        *value = newval*(range.1-range.0) - range.0;
    }
    result
}

pub fn save_floatgrid(fg: &FloatGrid, outpath: &str) {
    let scaled_fg = min_max_scaling(&fg, &(0., 1.));

    let mut imgbuf = image::ImageBuffer::new(fg.width() as u32, fg.height() as u32);
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let rgb = viridis::float2viridis(scaled_fg.get(x as usize, y as usize).unwrap());
        *pixel = image::Rgb([(rgb.0*255.) as u8, (rgb.1*255.) as u8, (rgb.2*255.) as u8]);
    }
    let ref mut fout = File::create(&outpath).unwrap();
    image::ImageRgb8(imgbuf).save(fout, image::PNG).unwrap();
}