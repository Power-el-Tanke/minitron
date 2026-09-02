use crate::{
    conn::WeightMatrix, layers::hidden_layer::HiddenLayer, layers::layer::Layer,
    layers::output_layer::OutputLayer, learning_rate::LearningRateManager, Random, RandomGen,
};
use std::iter::repeat_with;

pub struct Perceptron<T: Clone> {
    rng: RandomGen,
    learning_rate: f32,
    hidden_layers: Vec<HiddenLayer>,
    output_layer: OutputLayer<T>,
    connections: Vec<WeightMatrix>,
}

impl<T: Clone> Perceptron<T> {
    pub fn new<I>(
        mut rng: RandomGen,
        labels: I,
        input_size: usize,
        hiddenl_size: usize,
        hiddenl_num: usize,
    ) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let hidden_layers = repeat_with(|| HiddenLayer::fresh(hiddenl_size, &mut rng))
            .take(hiddenl_num)
            .collect::<Vec<_>>();
        let output_layer = OutputLayer::fresh(labels, &mut rng);
        let mut connections = Vec::<WeightMatrix>::with_capacity(hiddenl_num + 1);
        connections[0] = WeightMatrix::new(input_size, hiddenl_size, &mut rng);
        for i in 1..hiddenl_num - 1 {
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

    pub fn just_compute(&self, input: &[f32]) -> &T {
        let mut aux_vec = self.connections[0].compute(input);
        let iterations = self.hidden_layers.len();

        for i in 1..iterations {
            let layer_comp = self.hidden_layers[i - 1].forward_prop(&aux_vec);
            aux_vec = self.connections[i].compute(&layer_comp);
        }

        let output = self
            .output_layer
            .forward_prop(&self.connections[iterations].compute(&aux_vec));

        self.output_layer.winner(&output)

    }

    fn learn_from_input(&mut self, input: &[f32], expected: usize) -> &T {
        let iterations = self.hidden_layers.len();
        let mut connection_bread_crumbs: Vec<Vec<f32>> = Vec::with_capacity(iterations + 1);
        let mut layer_bread_crumbs: Vec<Vec<f32>> = Vec::with_capacity(iterations + 2);
        layer_bread_crumbs[0] = input.to_vec();
        for i in 0..iterations {
            connection_bread_crumbs[i] = self.connections[i].compute(&layer_bread_crumbs[i]);
            layer_bread_crumbs[i + 1] =
                self.hidden_layers[i].forward_prop(&connection_bread_crumbs[i]);
        }

        connection_bread_crumbs[iterations] =
            self.connections[iterations].compute(&layer_bread_crumbs[iterations]);
        layer_bread_crumbs[iterations + 1] = self
            .output_layer
            .forward_prop(&connection_bread_crumbs[iterations]);
        let result = self
            .output_layer
            .winner(&layer_bread_crumbs[iterations + 1]);

        let mut error_vec = Vec::with_capacity(self.output_layer.len());
        error_vec[expected] = 1.0;
        let learning_rate = self.learning_rate();

        for i in 1..=iterations {
            error_vec = self.hidden_layers[1 + iterations - i].learn(
                learning_rate,
                &layer_bread_crumbs[1 + iterations - i],
                &error_vec,
            );
            error_vec = self.connections[iterations - i].learn(
                &connection_bread_crumbs[iterations - 1],
                &error_vec,
                learning_rate,
            );
        }

        result
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.connections[0].input_len()
    }
}

impl<T: Clone> Random for Perceptron<T> {
    #[inline]
    fn rand_float(&mut self) -> f32 {
        self.rng.rand_float()
    }
}

impl<T: Clone> LearningRateManager for Perceptron<T> {
    #[inline]
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}
