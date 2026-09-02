use super::layer::Layer;
use crate::{Random, nodes::output_node::OutputNode};

pub struct OutputLayer<T: Clone> {
    nodes: Vec<OutputNode<T>>,
}

impl<T: Clone> Layer<OutputNode<T>> for OutputLayer<T> {
    fn forward_prop(&self, input: &[f32]) -> Vec<f32> {
        self.nodes
            .iter()
            .zip(input)
            .map(|(x, y)| x.activation(*y))
            .collect()
    }

    fn get(&self, index: usize) -> &OutputNode<T> {
        &self.nodes[index]
    }
}

impl<T: Clone> OutputLayer<T> {
    pub fn winner(&self, outputs: &[f32]) -> &T {
        let mut max = 0;
        for i in 0..outputs.len() {
            if outputs[i] > outputs[max] {
                max = i;
            }
        }
        self.nodes[max].get()
    }

    pub fn fresh<I, R>(labels: I, rng: &mut R) -> Self
    where
        I: IntoIterator<Item = T>,
        R: Random,
    {
        OutputLayer {
            nodes: labels
                .into_iter()
                .map(|x| OutputNode::new(x, rng))
                .collect(),
        }
    }

    pub fn learn(&mut self, prev_out: &[f32], expected: &[f32], rate: f32) {
        let len = self.nodes.len();
        for i in 0..len {
            let prev_err = (expected[i] - prev_out[i]).powi(2);
            self.nodes[i].learn(prev_out[i], prev_err, rate);
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}
