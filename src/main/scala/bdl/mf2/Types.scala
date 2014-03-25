package mf2

object OptimizerType extends Enumeration with Serializable {
  type OptimizerType = Value
  val ALS, CD, CDPP = Value
}

object LossType extends Enumeration with Serializable {
  type LossType = Value
  val Logistic, Hinge, Square = Value
}

object RegularizerType extends Enumeration with Serializable {
  type RegularizerType = Value
  val L2, L1, Max = Value
}

object ModelType extends Enumeration with Serializable {
  type ModelType = Value
  val MEM, sMEM, hMEM, hecMEM, AVGM, sAVGM, ADMM, dGM = Value
}