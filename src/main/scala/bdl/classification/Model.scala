package classification

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import utilities.SparseMatrix
import utilities.SparseVector
import classification.RegularizerType._
import classification.OptimizerType._

abstract class Model(val weights: Array[Double]) extends Serializable {
  
  val numFeatures = weights.length
    
  def train(
      trainingData: RDD[(Int, (Array[Byte], SparseMatrix))],
      maxNumIter: Int, optType: OptimizerType, 
      regPara: Double, regType: RegularizerType): Model = null
  
  def getGamma: Array[Double] = null
  def getRho: Double = 1
}

object Model {
  
}