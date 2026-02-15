
#[derive(Clone)]
pub enum Label<T: Clone>{
    Training {
        label: T,
        expected_val: f32,
    },
    Usage {
        label: T,
    }
}

impl<T: Clone> Label <T> {
    pub fn to_usage(&self) -> Self {
        if let Label::Training{label, ..} = self {
            Label::Usage {
                label: label.clone(),
            }
        } else {
            self.clone()
        }
    }
    
    pub fn get_err(&self, val: f32) -> f32 {
        if let Label::Training{expected_val, ..} = self {
            (expected_val - val).abs()
        } else {
            panic!("Attempt to eval result from an usage label")
        }
    }
}