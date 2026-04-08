
pub struct InputLayer {
    input: Vec<f32>,
}

impl InputLayer {
    pub fn new(neuron_ammount: usize) -> Self{
        Self {
            input: Vec::with_capacity(neuron_ammount)
        }
    }
    
    #[inline]
    pub fn set_input(&mut self, new_input: Vec<f32>) {
        self.input = new_input;
    }
}