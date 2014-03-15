package lr2

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast

import classification._
import classification.RegularizerType._
import classification.OptimizerType._
import utilities.SparseMatrix
import utilities.DoubleVector

class HMEM (
    override val weights: Array[Double], 
    val localModels: RDD[(Int, LocalModel)],
    val rho: Double, 
    val ec: Boolean)
  extends Model(weights) with Serializable{
  override def train(
      trainingData: RDD[(Int, (Array[Byte], SparseMatrix))],
      numIter: Int, optType: OptimizerType, 
      regPara: Double, regType: RegularizerType): HMEM = {
    HMEM.train(trainingData, this, numIter, optType, regPara, regType)
  }
  
  override def getGamma = localModels.map(_._2.gamma).collect
  override def getRho = rho
}

object HMEM {
  def train(
      trainingData: RDD[(Int, (Array[Byte], SparseMatrix))],
      model: HMEM,
      numIter: Int, optType: OptimizerType, 
      regPara: Double, regType: RegularizerType): HMEM = {
    val weights = model.weights
    val localModels = model.localModels
    val ec = model.ec
    val rho = model.rho
    val numFeatures = model.numFeatures
    val priorBC = trainingData.context.broadcast(weights)
    val sse = trainingData.context.accumulable(0.0)
    val updatedLocalModels = trainingData.join(localModels).mapValues{
      case((labels, features), localModel) => {
        val localWeights = localModel.weights
        val gamma = localModel.gamma
        val lags = localModel.lags
        val prior = Functions.toLocal(priorBC.value, features.rowMap)
        val updatedLags = 
          if (ec) Optimizers.dualAscent(localWeights, prior, lags)
          else null
        val updatedPrior = 
          if (ec) (DoubleVector(prior) - DoubleVector(updatedLags)).elements
          else prior
        val (updatedLocalWeights, updatedGamma, se) = Optimizers.runCD(labels, 
            features, localWeights, updatedPrior, true, numIter, rho, gamma)
        sse += se
        new LocalModel(updatedLocalWeights, updatedLags, updatedGamma)
      }
    }.cache
    val localMaps = trainingData.mapValues(_._2.rowMap).cache
    val updatedWeights = 
      updateWeights(updatedLocalModels, localMaps, regPara, regType, numFeatures)
    val updatedRho = localMaps.map(_._2.length).sum/sse.value
    localModels.unpersist(true)
    localMaps.unpersist(true)
    new HMEM(updatedWeights, updatedLocalModels, updatedRho, ec)
  }
  
  def updateWeights(
      localModels: RDD[(Int, LocalModel)], 
      localMaps: RDD[(Int, Array[Int])], 
      regPara: Double, regType: RegularizerType,
      numFeatures: Int): Array[Double] = {
    val gammaSum = localModels.join(localMaps).map{
      case(bid, (model, map)) => model.getGamma(map, numFeatures)
    }.reduce(_+=_).elements
    val paraStats = localModels.join(localMaps).map{
      case(bid, (model, map)) => model.getWeightedPara(map, numFeatures)
    }.reduce(_+=_).elements
    if (regType == L1) Optimizers.l1Prox(paraStats, gammaSum, regPara)
    else Optimizers.l2Prox(paraStats, gammaSum, regPara)
  }
}