//! This crate provides an implementation of Chan-Vese level-sets
//! described in [Active contours without edges](http://ieeexplore.ieee.org/document/902291/)
//! by T. Chan and L. Vese.
//! It is a port of the Python implementation by Kevin Keraudren on
//! [Github](https://github.com/kevin-keraudren/chanvese)
//! and of the Matlab implementation by [Shawn Lankton](http://www.shawnlankton.com).
//!
//! # Examples
//! To use the functions inside module `chanvese::utils` you need to
//! compile this crate with the feature image-utils.
//! 
//! ```
//! extern crate image;
//! extern crate rand;
//! extern crate chanvese;
//! 
//! use std::f64;
//! use std::fs::File;
//! use image::ImageBuffer;
//! use rand::distributions::{Sample, Range};
//! use chanvese::{FloatGrid, BoolGrid, chanvese};
//! 
//! use chanvese::utils;
//! 
//! fn main() {
//!     // create an input image (blurred and noisy ellipses)
//!     let img = {
//!         let mut img = ImageBuffer::new(256, 128);
//!         for (x, y, pixel) in img.enumerate_pixels_mut() {
//!             if (x-100)*(x-100)+(y-70)*(y-70) <= 35*35 {
//!                 *pixel = image::Luma([200u8]);
//!             }
//!             if (x-128)*(x-128)/2+(y-50)*(y-50) <= 30*30 {
//!                 *pixel = image::Luma([150u8]);
//!             }
//!         }
//!         img = image::imageops::blur(&img, 5.);
//!         let mut noiserange = Range::new(0.0f32, 30.);
//!         let mut rng = rand::thread_rng();
//!         for (_, _, pixel) in img.enumerate_pixels_mut() {
//!             *pixel = image::Luma([pixel.data[0] + noiserange.sample(&mut rng) as u8]);
//!         }
//!         let ref mut imgout = File::create("image.png").unwrap();
//!         image::ImageLuma8(img.clone()).save(imgout, image::PNG).unwrap();
//!         let mut result = FloatGrid::new(256, 128);
//! 
//!         for (x, y, pixel) in img.enumerate_pixels() {
//!             result.set(x as usize, y as usize, pixel.data[0] as f64);
//!         }
//!         result
//!     };
//! 
//!     // create a rough mask
//!     let mask = {
//!         let mut result = BoolGrid::new(img.width(), img.height());
//!         for (x, y, value) in result.iter_mut() {
//!             if (x >= 65 && x <= 180) && (y >= 20 && y <= 100) {
//!                 *value = true;
//!             }
//!         }
//!         result
//!     };
//!     utils::save_boolgrid(&mask, "mask.png");
//! 
//!     // level-set segmentation by Chan-Vese
//!     let (seg, phi, _) = chanvese(&img, &mask, 500, 1.0, 0);
//!     utils::save_boolgrid(&seg, "out.png");
//!     utils::save_floatgrid(&phi, "phi.png");
//! }
//! ```


extern crate distance_transform;

use distance_transform::dt2d;
use std::f64;

pub use distance_transform::{FloatGrid, BoolGrid};

#[cfg(feature = "image-utils")]
pub mod utils;
#[cfg(feature = "image-utils")]
mod viridis;

/// Runs the chanvese algorithm
/// 
/// Returns the resulting mask (`true` = foreground, `false` = background),
/// the level-set function and the number of iterations.
///
/// # Arguments
///
/// * `img` - the input image
/// * `init_mask` - in initial mask (`true` = foreground, `false` = background)
/// * `max_its` - number of iterations
/// * `alpha` - weight of smoothing term (default: 0.2)
/// * `thresh` - number of different pixels in masks of successive steps (default: 0)
pub fn chanvese(img: &FloatGrid,
            init_mask: &BoolGrid,
            max_its: u32,
            alpha: f64,
            thresh: u32) -> (BoolGrid, FloatGrid, u32) {
    // create a signed distance map (SDF) from mask
    let mut phi = mask2phi(init_mask);

    // main loop
    let mut its = 0u32;
    let mut stop = false;
    let mut prev_mask = init_mask.clone();
    let mut c = 0u32;

    while its < max_its && !stop {
        // get the curve's narrow band
        let idx = {
            let mut result = Vec::new();
            for (x, y, &val) in phi.iter() {
                if val >= -1.2 && val <= 1.2 {
                    result.push((x, y));
                }
            }
            result
        };

        if idx.len() > 0 {
            // intermediate output
            if its % 50 == 0 {
                println!("iteration: {}", its);
            }

            // find interior and exterior mean
            let (upts, vpts) = {
                let mut res1 = Vec::new();
                let mut res2 = Vec::new();
                for (x, y, value) in phi.iter() {
                    if *value <= 0. {
                        res1.push((x, y));
                    }
                    else {
                        res2.push((x, y));
                    }
                }
                (res1, res2)
            };

            let u = upts.iter().fold(0f64, |acc, &(x, y)| {
                acc + *img.get(x, y).unwrap()
            }) / (upts.len() as f64 + f64::EPSILON);
            let v = vpts.iter().fold(0f64, |acc, &(x, y)| {
                acc + *img.get(x, y).unwrap()
            }) / (vpts.len() as f64 + f64::EPSILON);

            // force from image information
            let f: Vec<f64> = idx.iter().map(|&(x, y)| {
                (*img.get(x, y).unwrap() - u)*(*img.get(x, y).unwrap() - u)
                -(*img.get(x, y).unwrap() - v)*(*img.get(x, y).unwrap() - v)
            }).collect();

            // force from curvature penalty
            let curvature = get_curvature(&phi, &idx);

            // gradient descent to minimize energy
            let dphidt: Vec<f64> = {
                let maxabs = f.iter().fold(0.0f64, |acc, &x| {
                    acc.max(x.abs())
                });
                f.iter().zip(curvature.iter()).map(|(f, c)| {
                    f/maxabs + alpha*c
                }).collect()
            };

            // maintain the CFL condition
            let dt = 0.45/(dphidt.iter().fold(0.0f64, |acc, &x| acc.max(x.abs())) + f64::EPSILON);

            // evolve the curve
            for i in 0..idx.len() {
                let (x, y) = idx[i];
                let val = *phi.get(x, y).unwrap();
                phi.set(x, y, val + dt*dphidt[i]);
            }

            // keep SDF smooth
            phi = sussman(&phi, &0.5);

            let new_mask = {
                let mut result = BoolGrid::new(phi.width(), phi.height());
                for (x, y, value) in phi.iter() {
                    result.set(x, y, *value <= 0.);
                }
                result
            };

            c = convergence(&prev_mask, &new_mask, thresh, c);

            if c <= 5 {
                its += 1;
                prev_mask = new_mask.clone();
            }
            else {
                stop = true;
            }
        }
        else {
            break;
        }
    }

    // make mask from SDF, get mask from levelset
    let seg = {
        let mut res = BoolGrid::new(phi.width(), phi.height());
        for (x, y, &value) in phi.iter() {
            res.set(x, y, value <= 0.);
        }
        res
    };

    (seg, phi, its)
}

fn bwdist(a: &BoolGrid) -> FloatGrid {
    let mut res = dt2d(&a);
    for (_, _, value) in res.iter_mut() {
        let newval = value.sqrt();
        *value = newval;
    }
    res
}

// Converts a mask to a SDF
fn mask2phi(init_a: &BoolGrid) -> FloatGrid {
    let inv_init_a = {
        let mut result = init_a.clone();
        for (_, _, value) in result.iter_mut() {
            *value = !*value;
        }
        result
    };

    let phi = {
        let dist_a = bwdist(&init_a);
        let dist_inv_a = bwdist(&inv_init_a);
        let mut result = FloatGrid::new(init_a.width(), init_a.height());
        for (x, y, value) in result.iter_mut() {
            *value = dist_a.get(x, y).unwrap()
                     - dist_inv_a.get(x, y).unwrap()
                     + if *init_a.get(x, y).unwrap() {1.} else {0.}
                     - 0.5;
        }
        result
    };

    phi
}

// Compute curvature along SDF
fn get_curvature(phi: &FloatGrid, idx: &Vec<(usize,usize)>) -> Vec<f64> {
    // get central derivatives of SDF at x,y
    let (phi_x, phi_y, phi_xx, phi_yy, phi_xy) = {
        let (mut res_x, mut res_y, mut res_xx, mut res_yy, mut res_xy)
            : (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = (
            Vec::with_capacity(idx.len()), 
            Vec::with_capacity(idx.len()), 
            Vec::with_capacity(idx.len()), 
            Vec::with_capacity(idx.len()), 
            Vec::with_capacity(idx.len()));

        for &(x, y) in idx.iter() {
            let left = if x > 0 { x - 1 } else { 0 };
            let right = if x < phi.width() - 1 { x + 1 } else { phi.width() - 1 };
            let up = if y > 0 { y - 1 } else { 0 };
            let down = if y < phi.height() - 1 { y + 1 } else { phi.height() - 1 };

            res_x.push(-*phi.get(left, y).unwrap() + *phi.get(right, y).unwrap());
            res_y.push(-*phi.get(x, down).unwrap() + *phi.get(x, up).unwrap());
            res_xx.push(
                  *phi.get(left, y).unwrap()
                - 2.0 * *phi.get(x, y).unwrap()
                + *phi.get(right, y).unwrap());
            res_yy.push(
                  *phi.get(x, up).unwrap()
                - 2.0 * *phi.get(x, y).unwrap()
                + *phi.get(x, down).unwrap());
            res_xy.push(0.25*(
                -*phi.get(left, down).unwrap() - *phi.get(right, up).unwrap()
                +*phi.get(right, down).unwrap() + *phi.get(left, up).unwrap()
            ));
        }
        (res_x, res_y, res_xx, res_yy, res_xy)
    };
    let phi_x2: Vec<f64> = phi_x.iter().map(|x| x*x).collect();
    let phi_y2: Vec<f64> = phi_y.iter().map(|x| x*x).collect();

    // compute curvature (Kappa)
    let curvature: Vec<f64> = (0..idx.len()).map(|i| {
        ((phi_x2[i]*phi_yy[i] + phi_y2[i]*phi_xx[i] - 2.*phi_x[i]*phi_y[i]*phi_xy[i])/
        (phi_x2[i] + phi_y2[i] + f64::EPSILON).powf(1.5))*(phi_x2[i] + phi_y2[i]).powf(0.5)
    }).collect();

    curvature
}

// Level set re-initialization by the sussman method
fn sussman(grid: &FloatGrid, dt: &f64) -> FloatGrid {
    // forward/backward differences
    let (a, b, c, d) = {
        let mut a_res = FloatGrid::new(grid.width(), grid.height());
        let mut b_res = FloatGrid::new(grid.width(), grid.height());
        let mut c_res = FloatGrid::new(grid.width(), grid.height());
        let mut d_res = FloatGrid::new(grid.width(), grid.height());
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                a_res.set(x, y,
                    grid.get(x, y).unwrap()
                    - grid.get((x + grid.width() - 1) % grid.width(), y).unwrap());
                b_res.set(x, y,
                    grid.get((x + 1) % grid.width(), y).unwrap()
                    - grid.get(x, y).unwrap());
                c_res.set(x, y,
                    grid.get(x, y).unwrap()
                    - grid.get(x, (y + 1) % grid.height()).unwrap());
                d_res.set(x, y,
                    grid.get(x, (y + grid.height() - 1) % grid.height()).unwrap()
                    - grid.get(x, y).unwrap());
            }
        }
        (a_res, b_res, c_res, d_res)
    };
    
    let (a_p, a_n, b_p, b_n, c_p, c_n, d_p, d_n) = {
        let mut a_p_res = FloatGrid::new(grid.width(), grid.height());
        let mut a_n_res = FloatGrid::new(grid.width(), grid.height());
        let mut b_p_res = FloatGrid::new(grid.width(), grid.height());
        let mut b_n_res = FloatGrid::new(grid.width(), grid.height());
        let mut c_p_res = FloatGrid::new(grid.width(), grid.height());
        let mut c_n_res = FloatGrid::new(grid.width(), grid.height());
        let mut d_p_res = FloatGrid::new(grid.width(), grid.height());
        let mut d_n_res = FloatGrid::new(grid.width(), grid.height());

        for y in 0..grid.height() {
            for x in 0..grid.width() {
                let a_p_dval = *a.get(x, y).unwrap();
                let a_n_dval = *a.get(x, y).unwrap();
                let b_p_dval = *b.get(x, y).unwrap();
                let b_n_dval = *b.get(x, y).unwrap();
                let c_p_dval = *c.get(x, y).unwrap();
                let c_n_dval = *c.get(x, y).unwrap();
                let d_p_dval = *d.get(x, y).unwrap();
                let d_n_dval = *d.get(x, y).unwrap();

                a_p_res.set(x, y, if a_p_dval >= 0.0 { a_p_dval } else { 0.0 });
                a_n_res.set(x, y, if a_n_dval <= 0.0 { a_n_dval } else { 0.0 });
                b_p_res.set(x, y, if b_p_dval >= 0.0 { b_p_dval } else { 0.0 });
                b_n_res.set(x, y, if b_n_dval <= 0.0 { b_n_dval } else { 0.0 });
                c_p_res.set(x, y, if c_p_dval >= 0.0 { c_p_dval } else { 0.0 });
                c_n_res.set(x, y, if c_n_dval <= 0.0 { c_n_dval } else { 0.0 });
                d_p_res.set(x, y, if d_p_dval >= 0.0 { d_p_dval } else { 0.0 });
                d_n_res.set(x, y, if d_n_dval <= 0.0 { d_n_dval } else { 0.0 });
            }
        }
        (a_p_res, a_n_res, b_p_res, b_n_res, c_p_res, c_n_res, d_p_res, d_n_res)
    };
    
    let mut d_d = FloatGrid::new(grid.width(), grid.height());
    let (d_neg_ind, d_pos_ind) = {
        let mut res = (Vec::new(), Vec::new());
        for (x, y, &value) in grid.iter() {
            if value < 0.0 {
                res.0.push((x, y));
            }
            else if value > 0.0 {
                res.1.push((x,y));
            }
        }
        res
    };

    for index in d_pos_ind {
        let mut ap = *a_p.get(index.0, index.1).unwrap();
        let mut bn = *b_n.get(index.0, index.1).unwrap();
        let mut cp = *c_p.get(index.0, index.1).unwrap();
        let mut dn = *d_n.get(index.0, index.1).unwrap();

        ap *= ap;
        bn *= bn;
        cp *= cp;
        dn *= dn;

        d_d.set(index.0, index.1, (ap.max(bn) + cp.max(dn)).sqrt() - 1.);
    }

    for index in d_neg_ind {
        let mut an = *a_n.get(index.0, index.1).unwrap();
        let mut bp = *b_p.get(index.0, index.1).unwrap();
        let mut cn = *c_n.get(index.0, index.1).unwrap();
        let mut dp = *d_p.get(index.0, index.1).unwrap();

        an *= an;
        bp *= bp;
        cn *= cn;
        dp *= dp;

        d_d.set(index.0, index.1, (an.max(bp) + cn.max(dp)).sqrt() - 1.);
    }

    let ss_d = sussman_sign(&grid);

    let mut res = FloatGrid::new(grid.width(), grid.height());
    for (x, y, value) in res.iter_mut() {
        let dval = grid.get(x, y).unwrap();
        let ss_dval = ss_d.get(x, y).unwrap();
        let d_dval = d_d.get(x, y).unwrap();
        *value = dval - dt*ss_dval*d_dval;
    }

    res
}

fn sussman_sign(d: &FloatGrid) -> FloatGrid {
    let mut res = FloatGrid::new(d.width(), d.height());
    for (x, y, value) in res.iter_mut() {
        let v = d.get(x, y).unwrap();
        *value = v/(v*v + 1.).sqrt();
    }
    res
}

// Convergence test
fn convergence(p_mask: &BoolGrid,
               n_mask: &BoolGrid,
               thresh: u32,
               c: u32) -> u32 {
    let n_diff = p_mask.iter().zip(n_mask.iter()).fold(0u32, |acc, ((_,_,p),(_,_,n))| {
        acc + if *p == *n { 1 } else { 0 }
    });

    if n_diff < thresh {
        c + 1
    }
    else {
        0
    }
}
