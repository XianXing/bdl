package classification

object OptimizerType extends Enumeration with Serializable {
  type OptimizerType = Value
  val CG, LBFGS, CD = Value
}

object LossType extends Enumeration with Serializable {
  type LossType = Value
  val Logistic, Hinge = Value
}

object RegularizerType extends Enumeration with Serializable {
  type RegularizerType = Value
  val L1, L2 = Value
}

object VariationalType extends Enumeration with Serializable {
  type VariationalType = Value
  val Jaakkola, Bohning, Taylor = Value
}

object ModelType extends Enumeration with Serializable {
  type ModelType = Value
  val MEM, sMEM, hMEM, hecMEM, AVGM, sAVGM, ADMM, dGM = Value
}