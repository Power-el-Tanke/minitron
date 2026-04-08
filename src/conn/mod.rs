use crate::Random;

pub struct WeightMatrix(Vec<Vec<f32>>);

impl WeightMatrix{
    pub fn new(left_size: usize, right_size: usize, rng:&mut impl Random) -> Self{
        let mut matrix = vec![Vec::with_capacity(right_size); left_size];
        for i in 0..left_size {
            for j in 0..right_size {
                matrix[i][j] = rng.rand_float();
            }
        }
        WeightMatrix(matrix)
     }
    
    pub fn compute(&self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.0[0].len());
        for c in 0..self.0[0].len() {
            let mut sum = 0.0;
            for r in 0..self.0.len() {
                sum += input[r] * self.0[r][c];
            }
            output.push(sum);
        }
        output
    }
    
    pub fn learn(
        &mut self, 
        former_input: &[f32],
        feedback: &[f32],
        rate: f32
    ) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.0.len());
        for r in 0..self.0.len() {
            let mut sum = 0.0;
            for c in 0..self.0[0].len() {
                let err = former_input[r] * feedback[c];
                self.0[r][c] -= rate * err;
                sum += self.0[r][c] * err;
            }
            output[r] = sum;
        }
        output
    }
}