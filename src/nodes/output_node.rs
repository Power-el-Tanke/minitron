use super::label::Label;
use crate::{
    sigmoid, 
    sigmoid_derivative,
    Random,
};

pub struct OutputNode <T: Clone> {
    bias: f32,
    label: Label<T>,
}

impl <T: Clone> OutputNode<T> {
    pub fn new<R: Random>(label: Label<T>, rng: &mut R) -> Self{
        OutputNode {
            bias: rng.rand_float(),
            label,
        }
    }
    #[inline]
    pub fn activation(&self, x: f32) -> f32 {
        sigmoid(x - self.bias)
    }
    
    #[inline]
    fn local_gradient(&self, x: f32, err: f32) -> f32 {
        err * sigmoid_derivative(x + self.bias)
    }
    
    #[inline]
    fn get_err(&mut self, val: f32) -> f32 {
        self.label.get_err(val)
    }
    
    pub fn change_to_usage(&mut self) {
        self.label = self.label.to_usage();
    }
    
    fn learn(&mut self, out: f32, err: f32, rate: f32) {
        let local_gradient = self.local_gradient(out,err);
        self.bias -= rate * local_gradient;
    }
}