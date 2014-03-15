package mf2

import org.apache.spark.rdd.RDD
import org.apache.spark.Partitioner
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel

import utilities.SparseMatrix
import utilities.Vector
import OptimizerType._
import RegularizerType._

class Model (val factorsR: RDD[(Int, Array[Float])],
    val factorsC: RDD[(Int, Array[Float])]) extends Serializable {
    
  def train(numIter: Int, optType: OptimizerType, 
      regPara: Float, regType: RegularizerType): Model = this
  
  def init(numIter: Int, optType: OptimizerType, 
      regPara: Float, regType: RegularizerType): Model = this
  
  def getValidatingRMSE(global: Boolean): Double = 0
  
  def distributeFactorsR(pids: RDD[(Int, Array[Int])])
    : RDD[(Int, Array[(Int, Array[Float])])] = {
    Model.distribute(factorsR, pids)
  }
  
  def distributeFactorsC(pids: RDD[(Int, Array[Int])])
    : RDD[(Int, Array[(Int, Array[Float])])] = {
    Model.distribute(factorsC, pids)
  }
  
  def getFactorsRL2Norm = Model.getL2Norm(factorsR)
  def getFactorsCL2Norm = Model.getL2Norm(factorsC)
  
  def getGammaR: Array[Float] = null
  def getGammaC: Array[Float] = null
  
}

private object Model {
  
  def distribute(factors: RDD[(Int, Array[Float])], pids: RDD[(Int, Array[Int])])
    : RDD[(Int, Array[(Int, Array[Float])])] = {
    factors.join(pids).flatMap{
      case(idx, (factor, pids)) => pids.map(id => (id, (idx, factor)))
    }.groupByKey.mapValues(seq => (seq.toArray).sortBy(_._1))
  }
  
  def getGlobalRMSE(data: RDD[(Int, SparseMatrix)], nnz: Int, 
      factorsR: RDD[(Int, Array[Float])], factorsC: RDD[(Int, Array[Float])],
      pidsR: RDD[(Int, Array[Int])], pidsC: RDD[(Int, Array[Int])]): Double = {
    math.sqrt(getSE(data, Model.distribute(factorsR, pidsR), 
      Model.distribute(factorsC, pidsC))/nnz)
  }
  
  def getLocalRMSE(data: RDD[(Int, SparseMatrix)], nnz: Int,
      localModels: RDD[(Int, LocalModel)], 
      mapsR: RDD[(Int, Array[Int])], mapsC: RDD[(Int, Array[Int])]): Double = {
    val factorsR = localModels.join(mapsR).mapValues{
      case(localModel, mapR) => LocalModel.toGlobal(localModel.factorsR, mapR)
    }
    val factorsC = localModels.join(mapsC).mapValues{
      case(localModel, mapC) => LocalModel.toGlobal(localModel.factorsC, mapC)
    }
    math.sqrt(getSE(data, factorsR, factorsC)/nnz)
  }
  
  def getSE(data: RDD[(Int, SparseMatrix)],
      factorsR: RDD[(Int, Array[(Int, Array[Float])])],
      factorsC: RDD[(Int, Array[(Int, Array[Float])])]) = {
    data.join(factorsR.join(factorsC)).map{
      case (pid, (localData, (localFactorsR, localFactorsC))) =>
        localData.getSE(DivideAndConquer.toLocal(localFactorsR, localData.rowMap), 
          DivideAndConquer.toLocal(localFactorsC, localData.colMap))
    }.reduce(_+_)
  }
  
  def getL2Norm(factors: RDD[(Int, Array[Float])]): Double = {
    factors.map(pair => Vector(pair._2).l2Norm).sum/factors.count
  }
}