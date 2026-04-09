use std::iter::repeat_with;
use crate::{
    layers::output_layer::OutputLayer,
    layers::input_layer::InputLayer,
    layers::hidden_layer::HiddenLayer,
    conn::WeightMatrix,
    learning_rate::LearningRateManager,
    nodes::label::Label,
    RandomGen,
    Random,
};

pub struct Perceptron<T: Clone> {
    rng: RandomGen,
    learning_rate: f32,
    input_layer: InputLayer,
    hidden_layers: Vec<HiddenLayer>,
    output_layer: OutputLayer<T>,
    connections: Vec<WeightMatrix>,
}

impl<T: Clone> Perceptron<T>{
  pub fn new<I>(
    mut rng: RandomGen,
    labels: I, 
    input_size: usize,
    hiddenl_size: usize,
    hiddenl_num: usize
    ) -> Self
  where 
    I: IntoIterator<Item=Label<T>>
  {
    let input_layer = InputLayer::new(input_size);
    let hidden_layers = repeat_with(||HiddenLayer::fresh(hiddenl_size, &mut rng))
                          .take(hiddenl_num)
                          .collect::<Vec<_>>();
    let output_layer = OutputLayer::fresh(labels, &mut rng);
    let mut connections = Vec::<WeightMatrix>::with_capacity(hiddenl_num + 1);
    connections[0] = WeightMatrix::new(input_size, hiddenl_size, &mut rng);
    for i in 1 .. hiddenl_num - 1 {
      connections[i] = WeightMatrix::new(hiddenl_size, hiddenl_size, &mut rng);
    }
    connections[hiddenl_num] = WeightMatrix::new(hiddenl_size, output_layer.len(), &mut rng);
    Self {
      rng,
      learning_rate: 1.0,
      input_layer,
      hidden_layers,
      output_layer,
      connections,
    }
  }
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