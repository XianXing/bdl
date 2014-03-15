package lr2

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import classification._
import classification.OptimizerType._
import classification.RegularizerType._
import utilities.SparseMatrix
import utilities.SparseVector
import Optimizers._

class MEM (
    override val weights: Array[Double],
    override val localModels: RDD[(Int, LocalModel)],
    override val featureCount: Array[Int])
  extends AVGM(weights, localModels, featureCount) 
  with Serializable{
  
  override def train(
      trainingData: RDD[(Int, (Array[Byte], SparseMatrix))],
      maxNumIter: Int, optType: OptimizerType, 
      regPara: Double, regType: RegularizerType): MEM = {
    val avgm = 
      super.train(trainingData, maxNumIter, CD, regPara, regType)
    new MEM(avgm.weights, avgm.localModels, avgm.featureCount)
  }
}