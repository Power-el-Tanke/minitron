#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
mod nodes;
mod layers;
mod conn;
mod learning_rate;
mod perceptron;

use std::{
    boxed::Box,
    rc::Rc,
    cell::RefCell,
    sync::Arc,
    cmp::Ordering,
    time::{SystemTime, Duration, UNIX_EPOCH},
};

pub type MultiRef<N> = Rc<RefCell<Box<N>>>;
pub type SharedRef<N> = Rc<RefCell<N>>;
pub type AMultiRef<N> = Arc<RefCell<Box<N>>>;

pub fn new_multi_ref<N>(val: N) -> MultiRef<N>{
    Rc::new(RefCell::new(Box::new(val)))
}

pub fn new_shared_ref<N>(val: N) -> SharedRef<N>{
    Rc::new(RefCell::new(val))
}

#[must_use]
pub fn compare_floats(a: &f32, b: &f32) -> Ordering {
    let diff = a - b;
    if diff.abs() <= 0.000_000_001 {
        Ordering::Equal
    } else if diff > 0.0 {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}

#[inline]
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
#[must_use]
pub fn sigmoid_derivative(x: f32) -> f32 {
    let neg_exp = (-x).exp();
    let base = 1.0 + neg_exp;
    - neg_exp / (base * base)
}

#[inline]
#[must_use]
pub fn error(actual: f32, expected: f32) -> f32 {
    (expected - actual) * (expected - actual)
}
    
#[inline]
#[must_use]
pub fn error_der(actual: f32, expected:f32) -> f32 {
    -2.0 * (expected - actual)
}

pub trait Random {
    fn rand_float(&mut self) -> f32;
}

pub struct RandomGen{
    seed: u32,
}

impl Default for RandomGen {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomGen{
    #[inline]
    fn next_step(&mut self) -> u32{
        self.seed ^= self.seed << 7;
        self.seed ^= self.seed >> 13;
        self.seed ^= self.seed << 21;
        self.seed ^= 0xF10A_32C5;
        
        self.seed
    }
    
    #[must_use]
    pub fn new() -> RandomGen{
        RandomGen {
            #[allow(clippy::cast_possible_truncation)]
            seed: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_nanos() as u32
        }
    }
}

impl Random for RandomGen {
    #[allow(clippy::cast_precision_loss)]
    fn rand_float(&mut self) -> f32{
        (self.next_step() as f32) / 1.0001
    }
}