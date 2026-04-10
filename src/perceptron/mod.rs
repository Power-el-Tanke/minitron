use std::iter::repeat_with;
use crate::{
    layers::output_layer::OutputLayer,
    layers::hidden_layer::HiddenLayer,
    conn::WeightMatrix,
    learning_rate::LearningRateManager,
    nodes::label::Label,
    RandomGen,
    Random,
    layers::layer::Layer,
};

pub struct Perceptron<T: Clone> {
    rng: RandomGen,
    learning_rate: f32,
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
      hidden_layers,
      output_layer,
      connections,
    }
  }
  
  pub fn just_compute(&self, input: &[f32]) -> Vec<f32> {
    let mut aux_vec = self.connections[0].compute(input);
    let iterations = self.hidden_layers.len();
    for i in 1 .. iterations {
      let layer_comp = self.hidden_layers[i - 1].forward_prop(&aux_vec);
      aux_vec = self.connections[i].compute(&layer_comp);
    }
    self.output_layer.forward_prop(
      &self.connections[iterations].compute(&aux_vec)
    )
  }
  
  #[inline]
  pub fn input_size(&self) -> usize {
    self.connections[0].input_len()
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