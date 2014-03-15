package lr2

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import classification._
import utilities.SparseMatrix
import utilities.DoubleVector
import classification.RegularizerType._
import classification.OptimizerType._
import Optimizers._

class ADMM (
    override val weights: Array[Double],
    override val localModels: RDD[(Int, LocalModel)],
    override val featureCount: Array[Int], 
    val rho: Double) 
    extends AVGM(weights, localModels, featureCount) with Serializable {
  
  override def train(
      trainingData: RDD[(Int, (Array[Byte], SparseMatrix))],
      numIter: Int, optType: OptimizerType,
      regPara: Double, regType: RegularizerType): ADMM = {
    ADMM.train(trainingData, this, numIter, optType, regPara, regType)
  }
}

object ADMM {
  def train(
      trainingData: RDD[(Int, (Array[Byte], SparseMatrix))],
      model: ADMM,
      numIter: Int, optType: OptimizerType,
      regPara: Double, regType: RegularizerType): ADMM = {
    val weights = model.weights
    val localModels = model.localModels
    val rho = model.rho
    val featureCount = model.featureCount
    val numFeatures = model.numFeatures
    val weightsBC = trainingData.context.broadcast(weights)
    val updatedLocalModels = localModels.join(trainingData).mapValues{
      case(localModel, (labels, features)) => {
        val localWeights = localModel.weights
        val lags = localModel.lags
        val priors = Functions.toLocal(weightsBC.value, features.rowMap)
        val updatedLags = dualAscent(localWeights, priors, lags)
        val updatedPriors = (DoubleVector(priors) - DoubleVector(updatedLags)).elements
        val updatedLocalWeights = optType match {
          case CD => {
            runCD(labels, features, localWeights, updatedPriors, false,
              numIter, 1, rho)._1
          }
          case _ => {
            runCGOrLBFGS(labels, features, localWeights, updatedPriors,
              optType, numIter, rho)
          }
        }
        new LocalModel(updatedLocalWeights, updatedLags, 1)
      }
    }.cache
    updatedLocalModels.count
    val localMaps = trainingData.mapValues(_._2.rowMap)
    val updatedWeights = 
      AVGM.updateWeights(updatedLocalModels, localMaps, featureCount, numFeatures)
    localModels.unpersist(true)
    new ADMM(updatedWeights, updatedLocalModels, featureCount, rho)
  }
}