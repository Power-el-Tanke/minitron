use crate::{
    nodes::{
        output_node::OutputNode,
        label::Label,
    },
    Random,
};
use super::layer::Layer;

pub struct OutputLayer <T: Clone> {
    nodes:Vec<OutputNode<T>>,
}

impl <T: Clone> Layer<OutputNode<T>> for OutputLayer <T> {
    fn forward_prop(&self, input: &[f32]) -> Vec<f32> {
        self.nodes
            .iter()
            .zip(input)
            .map(|(x,y)| x.activation(*y))
            .collect()
    }
}

impl  <T: Clone> OutputLayer <T> {
    pub fn winner(&self, outputs: &[f32]) -> &OutputNode<T> {
        let mut max = 0;
        for i in 0..outputs.len() {
            if outputs[i] > outputs[max] {
                max = i;
            }
        }
        &self.nodes[max]
    }
    
    fn fresh<I, R>(labels: I , rng: &mut R) -> Self
      where
        I: IntoIterator<Item=Label<T>>,
        R: Random
    {
        OutputLayer {
            nodes: labels.into_iter()
                         .map(|x| OutputNode::new(x, rng))
                         .collect(),
        }
    }
    
    pub fn learn(&mut self, prev_out: &[f32], prev_err: &[f32], rate: f32) {
      let len = self.nodes.len();
      for i in 0..len {
        self.nodes[i].learn(prev_out[i], prev_err[i], rate);
      }
    }
}