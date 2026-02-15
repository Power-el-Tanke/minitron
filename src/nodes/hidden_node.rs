use crate::{
    Random,
    sigmoid,
    sigmoid_derivative,
};

pub struct HiddenNode {
    bias: f32,
}

impl HiddenNode {
    pub fn new<R: Random>(rng: &mut R) -> Self{
        HiddenNode {
            bias: rng.rand_float(),
        }
    }
    
    pub fn learn(&mut self, output: f32, rate: f32, err: f32) -> f32 {
        let local_gradient = self.local_gradient(output, err);
        self.bias -= rate * local_gradient;
        err
    }
    
    #[inline]
    pub fn activation(&self, x: f32) -> f32 {
        sigmoid(x + self.bias)
    }
    
    #[inline]
    fn local_gradient(&self, x: f32, err: f32) -> f32 {
        err * sigmoid_derivative(x + self.bias)
    }
}