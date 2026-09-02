use crate::{Random, sigmoid, sigmoid_derivative};

pub struct OutputNode<T: Clone> {
    bias: f32,
    label: T,
}

impl<T: Clone> OutputNode<T> {
    pub fn new<R: Random>(label: T, rng: &mut R) -> Self {
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

    pub fn learn(&mut self, out: f32, err: f32, rate: f32) -> f32 {
        let local_gradient = self.local_gradient(out, err);
        self.bias -= rate * local_gradient;
        local_gradient
    }

    pub fn get(&self) -> &T {
        &self.label
    }
}
