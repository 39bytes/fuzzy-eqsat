use egg::*;

define_language! {
    pub enum Linalg {
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "svd_u" = SvdU([Id; 2]),
        "svd_d" = SvdD([Id; 2]),
        "svd_vt" = SvdVt([Id; 2]),
        "relu" = Relu(Id),
        "softmax" = Softmax(Id),
        Num(i32),
        Mat(Symbol),
    }
}
