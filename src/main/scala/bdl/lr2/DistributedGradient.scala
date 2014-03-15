package lr2

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext

import classification._
import classification.OptimizerType._
import classification.RegularizerType._
import utilities.SparseMatrix
import utilities.DoubleVector

class DistributedGradient(
    override val weights: Array[Double]) 
  extends Model(weights) with Serializable{
  
  override def train(
      trainingData: RDD[(Int, (Array[Byte], SparseMatrix))],
      maxNumIter: Int, optType: OptimizerType, 
      regPara: Double, regType: RegularizerType): DistributedGradient = {
    
    assume(optType == CG || optType == LBFGS, 
      "current version only supports CG and LBFGS")
    
    val numFeatures = weights.length
    val gradientPrev = new Array[Double](numFeatures)
    val direction = new Array[Double](numFeatures)
    val deltaPara = 
      if (optType == OptimizerType.LBFGS) new Array[Double](numFeatures) 
      else null
    var iter = 0
    while (iter < maxNumIter) {
      val weightsBC = trainingData.context.broadcast(weights)
      
      val gradient = trainingData.map{
        case (bid, (labels, features)) => {
          val map = features.rowMap
          val weightsLocal = Functions.toLocal(weightsBC.value, map)
          Functions.getGrad(labels, features, weightsLocal, weightsLocal)
          val gradientGlobal = Functions.toGlobal(weightsLocal, map, numFeatures)
          DoubleVector(gradientGlobal)
        }
      }.reduce(_+=_).elements
      var p = 1 //no shrinkage for the intercept
      while (p < numFeatures) {
        gradient(p) -= regPara*weights(p)
        p += 1
      }
      if (iter > 1) {
        if (optType == CG) {
          Optimizers.getCGDirection(gradient, gradientPrev, direction)
        }
        else if (optType == LBFGS) {
          Optimizers.getLBFGSDirection(deltaPara, gradient, gradientPrev, direction)
        }
      }
      else Array.copy(gradient, 0, direction, 0, numFeatures)
      val directionBC = trainingData.context.broadcast(direction)
      val h = trainingData.map{
        case(bid, (labels, features)) => {
          val map = features.rowMap
          val weightsLocal = Functions.toLocal(weightsBC.value, map)
          val direction = Functions.toLocal(directionBC.value, map)
          Functions.getHessian(features, weightsLocal, direction)
        }
      }.sum.toFloat
      p = 1 //no shrinkage for the intercept
      var gu = gradient(0)*direction(0)
      var uhu = 0.0
      while (p < numFeatures) {
        uhu += direction(p)*direction(p)
        gu += gradient(p)*direction(p)
        p += 1
      }
      uhu *= regPara
      uhu += h
      p = 0
      while (p < numFeatures) {
        gradientPrev(p) = gradient(p)
        //equation (17) in Tom Minka 2003
        val delta = gu/uhu*direction(p)
        if (optType == LBFGS) deltaPara(p) = delta
        weights(p) += delta
        p += 1
      }
      iter += 1
    }
    this
  }
}
