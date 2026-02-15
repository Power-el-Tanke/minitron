use crate::{
    layers::output_layer::OutputLayer,
    learning_rate::LearningRateManager,
    RandomGen,
    Random,
};

pub struct Perceptron<T: Clone> {
    rng: RandomGen,
    learning_rate: f32,
    output_layer: OutputLayer<T>,
}

impl<T: Clone> Perceptron<T>{
    
}

impl<T: Clone> Random for Perceptron<T> {
    #[inline]
    fn rand_float(&mut self) -> f32{
        self.rng.rand_float()
    }
}

impl<T: Clone> LearningRateManager for Perceptron<T> {
    #[inline]
    fn learning_rate(&self) -> f32{
        self.learning_rate
    }
}