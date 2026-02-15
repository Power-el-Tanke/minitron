
pub trait Layer<N> {
    fn forward_prop(&self, input: &[f32]) -> Vec<f32>;
}