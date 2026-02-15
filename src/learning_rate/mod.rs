pub trait LearningRateManager {
    fn learning_rate(&self) -> f32;
}