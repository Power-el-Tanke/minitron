
pub struct InputLayer {
    input: Vec<f32>,
}

impl InputLayer {
    fn new(neuron_ammount: usize) -> Self{
        Self {
            input: Vec::with_capacity(neuron_ammount)
        }
    }
    
    #[inline]
    fn set_input(&mut self, new_input: Vec<f32>) {
        self.input = new_input;
    }
}