package classification

import ModelType._
import utilities.DoubleVector

class LocalModel(
    val weights: Array[Double], 
    val lags: Array[Double], 
    val gamma: Double)
  extends Serializable {
  
  def getWeightedPara(map: Array[Int], numFeatures: Int): DoubleVector = {
    val weightedPara = new Array[Double](numFeatures)
    var i = 0
    while (i < map.length) {
      weightedPara(map(i)) = gamma*weights(i)
      i += 1
    }
    DoubleVector(weightedPara)
  }
  
  def getPara(map: Array[Int], numFeatures: Int): DoubleVector = {
    val weightedPara = new Array[Double](numFeatures)
    var i = 0
    while (i < map.length) {
      weightedPara(map(i)) = weights(i)
      i += 1
    }
    DoubleVector(weightedPara)
  }
  
  def getGamma(map: Array[Int], numFeatures: Int): DoubleVector = {
    val gammaStats = new Array[Double](numFeatures)
    var i = 0
    while (i < map.length) {
      gammaStats(map(i)) = gamma
      i += 1
    }
    DoubleVector(gammaStats)
  }
}

object LocalModel {
  def apply(numFeatures: Int, modelType: ModelType, gamma: Double) = {
    val weight = new Array[Double](numFeatures)
    val lag = modelType match {
      case `hecMEM` | ADMM => new Array[Double](numFeatures)
      case _ => null
    }
    new LocalModel(weight, lag, gamma)
  }
}