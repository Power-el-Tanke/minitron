use std::iter::repeat_with;
use crate::{
    Random,
    nodes::hidden_node::HiddenNode,
    layers::layer::Layer,
};

pub struct HiddenLayer {
    nodes: Vec<HiddenNode>,
}

impl Layer<HiddenNode> for HiddenLayer {
    fn forward_prop(&self, input: &[f32]) -> Vec<f32> {
        self.nodes
            .iter()
            .zip(input)
            .map(|(x,y)| x.activation(*y))
            .collect()
    }
}

impl HiddenLayer {
    pub fn learn(&mut self, rate: f32, outs: &[f32], errs: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.nodes.len());
        for i in 0..self.nodes.len() {
            output[i] = self.nodes[i].learn(outs[i],rate, errs[i],);
        }
        output
    }
    
    pub fn fresh<R: Random>(ammount: usize, rng: &mut R) -> Self{
        HiddenLayer {
            nodes: repeat_with(|| HiddenNode::new(rng)).take(ammount).collect::<Vec<_>>(),
        }
    }
}