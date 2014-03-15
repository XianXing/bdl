package lr2

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import classification._
import classification.OptimizerType._
import classification.RegularizerType._
import utilities.SparseMatrix
import utilities.IntVector
import Optimizers._

class AVGM (
    override val weights: Array[Double], 
    val localModels: RDD[(Int, LocalModel)],
    val featureCount: Array[Int])
  extends Model(weights) with Serializable{
  
  override def train(
      trainingData: RDD[(Int, (Array[Byte], SparseMatrix))],
      numIter: Int, optType: OptimizerType, 
      regPara: Double, regType: RegularizerType): AVGM = {
    AVGM.train(trainingData, this, numIter, optType, regPara, regType)
  }
}

private object AVGM {
  def train(
    trainingData: RDD[(Int, (Array[Byte], SparseMatrix))],
    model: AVGM,
    numIter: Int, optType: OptimizerType, 
    regPara: Double, regType: RegularizerType): AVGM = {
    
    val localModels = model.localModels
    val numFeatures = model.numFeatures
    val featureCount = model.featureCount
    val updatedLocalModels = localModels.join(trainingData).mapValues{
      case(localModel, (labels, features)) => {
        val localWeights = localModel.weights
        val updatedLocalWeights = optType match {
          case CD => {
            runCD(labels, features, localWeights, false, numIter, regPara, 1)._1
          }
          case _ => {
            runCGOrLBFGS(labels, features, localWeights, optType, numIter, regPara)
          }
        }
        new LocalModel(updatedLocalWeights, null, 1)
      }
    }.cache
    val localMaps = trainingData.mapValues(_._2.rowMap)
    val updatedWeights = 
      updateWeights(updatedLocalModels, localMaps, featureCount, numFeatures)
    localModels.unpersist(true)
    new AVGM(updatedWeights, updatedLocalModels, featureCount)
  }
  
  def updateWeights(
      localModels: RDD[(Int, LocalModel)], 
      localMaps: RDD[(Int, Array[Int])],
      featureCount: Array[Int],
      numFeatures: Int)
    : Array[Double] = {
    val paraStats = localModels.join(localMaps).map{
      case(bid, (localModel, map)) => localModel.getPara(map, numFeatures)
    }.reduce(_+=_).elements
    l2Prox(paraStats, featureCount, 0)
  }
}