use egg::*;

define_language! {
    pub enum Linalg {
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "svdmul" = SVDMul([Id; 3]),
        "relu" = Relu(Id),
        "softmax" = Softmax(Id),
        Num(i32),
        Mat(Symbol),
    }
}
