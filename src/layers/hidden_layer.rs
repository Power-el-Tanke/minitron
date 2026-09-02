use crate::{Random, layers::layer::Layer, nodes::hidden_node::HiddenNode};
use std::iter::repeat_with;

pub struct HiddenLayer {
    nodes: Vec<HiddenNode>,
}

impl Layer<HiddenNode> for HiddenLayer {
    fn forward_prop(&self, input: &[f32]) -> Vec<f32> {
        self.nodes
            .iter()
            .zip(input)
            .map(|(x, y)| x.activation(*y))
            .collect()
    }

    fn get(&self, index: usize) -> &HiddenNode {
        &self.nodes[index]
    }
}

impl HiddenLayer {
    pub fn learn(&mut self, rate: f32, outs: &[f32], errs: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.nodes.len());
        for i in 0..self.nodes.len() {
            output[i] = self.nodes[i].learn(outs[i], rate, errs[i]);
        }
        output
    }

    pub fn fresh<R: Random>(ammount: usize, rng: &mut R) -> Self {
        HiddenLayer {
            nodes: repeat_with(|| HiddenNode::new(rng))
                .take(ammount)
                .collect::<Vec<_>>(),
        }
    }
}
