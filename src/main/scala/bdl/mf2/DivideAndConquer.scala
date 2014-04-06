package mf2

import java.io.BufferedWriter

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkContext, Partitioner, HashPartitioner}
import org.apache.spark.broadcast.Broadcast


import org.apache.hadoop.io.NullWritable

import preprocess.MF._
import utilities.SparseMatrix
import utilities.Vector
import utilities.Record
import OptimizerType._
import RegularizerType._

class DivideAndConquer (
    override val factorsR: RDD[(Int, Array[Float])],
    override val factorsC: RDD[(Int, Array[Float])],
    val localModels: RDD[(Int, LocalModel)], 
    val trainingData: RDD[(Int, SparseMatrix)], val nnzTra: Int,
    val validatingData: RDD[(Int, SparseMatrix)], val nnzVal: Int,
    val mapsR: RDD[(Int, Array[Int])], val mapsC: RDD[(Int, Array[Int])],
    val pidsR: RDD[(Int, Array[Int])], val pidsC: RDD[(Int, Array[Int])],
    val numRowBlocks: Int, val numColBlocks: Int,
    val hasRowEC: Boolean, val hasColEC: Boolean, 
    val hasRowPrior: Boolean, val hasColPrior: Boolean,
    val isVB: Boolean, val emBayes: Boolean, val weightedReg: Boolean, 
    val multicore: Boolean, var stopCrt: Float, val numIters: Array[Int])
  extends Model(factorsR, factorsC) {
  
  override def setStopCrt(stopCrt: Float) = {
    this.stopCrt = stopCrt
  }
  
  override def getNumIters = numIters
  
  private def createNewModel(
      updatedFactorsR: RDD[(Int, Array[Float])], 
      updatedFactorsC: RDD[(Int, Array[Float])],
      updatedLocalModels: RDD[(Int, LocalModel)],
      numIters: Array[Int]): DivideAndConquer = {
    new DivideAndConquer(updatedFactorsR, updatedFactorsC, updatedLocalModels,
        trainingData, nnzTra, validatingData, nnzVal, mapsR, mapsC, pidsR, pidsC, 
        numRowBlocks, numColBlocks, hasRowEC, hasColEC, hasRowPrior, hasColPrior,
        isVB, emBayes, weightedReg, multicore, stopCrt, numIters)
  }
  
  override def init(numIter: Int, optType: OptimizerType, 
      regPara: Float, regType: RegularizerType): DivideAndConquer = {
    val (updatedFactorsR, updatedFactorsC, updatedLocalModels, numIters) = 
      DivideAndConquer.train(trainingData, factorsR, factorsC, localModels, 
        pidsR, pidsC, mapsR, mapsC, numIter, stopCrt, optType, regPara, regType, 
        numRowBlocks, numColBlocks, false, false, false, false, false, false, 
        weightedReg, multicore)
    createNewModel(updatedFactorsR, updatedFactorsC, updatedLocalModels, numIters)
  }
  
  override def train(numIter: Int, optType: OptimizerType, 
      regPara: Float, regType: RegularizerType): DivideAndConquer = {
    val (updatedFactorsR, updatedFactorsC, updatedLocalModels, numIters) = 
      DivideAndConquer.train(trainingData, factorsR, factorsC, localModels, 
        pidsR, pidsC, mapsR, mapsC, numIter, stopCrt, optType, regPara, regType, 
        numRowBlocks, numColBlocks, hasRowEC, hasColEC, hasRowPrior, hasColPrior, 
        isVB, emBayes, weightedReg, multicore)
    createNewModel(updatedFactorsR, updatedFactorsC, updatedLocalModels, numIters)
  }
  
  override def getGammaR: Array[Float] = {
    DivideAndConquer.getGamma(localModels.map(_._2.gammaR), weightedReg)
  }
  
  override def getGammaC: Array[Float] = {
    DivideAndConquer.getGamma(localModels.map(_._2.gammaC), weightedReg)
  }
  
  override def getValidatingRMSE(global: Boolean): Double = {
    if (global) {
      Model.getGlobalRMSE(validatingData, nnzVal, factorsR, factorsC, pidsR, pidsC)
    }
    else {
      Model.getLocalRMSE(validatingData, nnzVal, localModels, mapsR, mapsC)
    }
  }
}

object DivideAndConquer {
  
  def logInfo(trainingData: RDD[(Int, SparseMatrix)], nnzTra: Int, nnzVal: Int,
      numRows: Int, numCols: Int, bwLog: BufferedWriter) = {
    val samplesPerBlock = trainingData.map(pair => pair._2.col_idx.length).collect
    val rowsPerBlock = trainingData.map(pair => pair._2.rowMap.length).collect
    val colsPerBlock = trainingData.map(pair => pair._2.colMap.length).collect
    var b = 0
    println("samples per block:")
    while (b < samplesPerBlock.length) {
      print(b + ":" + samplesPerBlock(b) + "\t")
      b += 1
    }
    println
    println("rows per block:")
    b = 0
    while (b < rowsPerBlock.length) {
      print(b + ":" + rowsPerBlock(b) + "\t")
      b += 1
    }
    println
    println("cols per block:")
    b = 0
    while (b < colsPerBlock.length) {
      print(b + ":" + colsPerBlock(b) + "\t")
      b += 1
    }
    println
    println("Number of rows: " + numRows)
    println("Number of columns: " + numCols)
    println("Number of training samples: " + nnzTra)
    println("Number of testing samples: " + nnzVal)    
    bwLog.write("Number of rows: " + numRows + "\n")
    bwLog.write("Number of columns: " + numCols + "\n")
    bwLog.write("Number of training samples: " + nnzTra + "\n")
    bwLog.write("Number of testing samples: " + nnzVal + "\n")
  }
  
  def apply(sc: SparkContext, trainingInputDir: String, validatingInputDir: String,
      numRows: Int, numCols: Int, numRowBlocks: Int, numColBlocks: Int, 
      numCores: Int, syn: Boolean, mean: Float, scale: Float,
      numFactors: Int, gammaRInit: Float, gammaCInit: Float,
      isEC: Boolean, hasPrior: Boolean, isVB: Boolean, emBayes: Boolean,
      weightedReg: Boolean, multicore: Boolean, bwLog: BufferedWriter)
    : DivideAndConquer = {
    
    def hash(x: Int): Int = {
      val r = x ^ (x >>> 20) ^ (x >>> 12)
      r ^ (r >>> 7) ^ (r >>> 4)
    }
    
    val seedRow = hash(numRows)
    val rowBlockMap = 
      if (!syn) sc.broadcast(getPartitionMap(numRows+1, numRowBlocks, seedRow))
      else null
    val seedCol = hash(numCols+numRows)
    val colBlockMap = 
      if(!syn) sc.broadcast(getPartitionMap(numCols+1, numColBlocks, seedCol))
      else null
    val part = new HashPartitioner(numRowBlocks*numColBlocks)
    val trainingData = toSparseMatrixBlocks(sc, trainingInputDir, rowBlockMap, 
        colBlockMap, numRowBlocks, numColBlocks, part, syn, mean, scale).cache
    val nnzTra = trainingData.map(_._2.col_idx.length).reduce(_+_)
    val validatingData = toSparseMatrixBlocks(sc, validatingInputDir, rowBlockMap, 
        colBlockMap, numRowBlocks, numColBlocks, part, syn, mean, scale).cache
    val nnzVal = validatingData.map(_._2.col_idx.length).reduce(_+_)
    
    val hasRowEC = isEC && numColBlocks > 1
    val hasRowPrior = hasPrior && numColBlocks > 1
    val hasColEC = isEC && numRowBlocks > 1
    val hasColPrior = hasPrior && numRowBlocks > 1
    
    val localModels = trainingData.mapValues(data => 
      LocalModel(numFactors, data.rowMap, data.colMap, data.row_ptr, data.col_ptr,
        gammaRInit, gammaCInit, hasRowEC, hasColEC, isVB, weightedReg)).cache
    localModels.count
    val mapsR = trainingData.mapValues(_.rowMap).cache
    mapsR.count
    val mapsC = trainingData.mapValues(_.colMap).cache
    mapsC.count
    val pidsR = trainingData.flatMap{
      case (pid, data) => data.rowMap.map((_, List(pid)))
    }.reduceByKey(_:::_).mapValues(_.toArray).cache
    pidsR.count
    val pidsC = trainingData.flatMap{
      case (pid, data) => data.colMap.map((_, List(pid)))
    }.reduceByKey(_:::_).mapValues(_.toArray).cache
    pidsC.count
    val factorsR = 
      if (numColBlocks > 1) {
        trainingData.flatMap{
          case (pid, sm) => sm.rowMap.map((_, new Array[Float](numFactors)))
        }
      } else {
        localModels.join(mapsR).flatMap{
          case((pid, (localModel, mapR))) =>
            LocalModel.toGlobal(localModel.factorsR, mapR)
        }
      }
    factorsR.cache
    factorsR.count
    val factorsC = 
      if (numRowBlocks > 1) {
        trainingData.flatMap{
          case (pid, sm) => sm.colMap.map((_, new Array[Float](numFactors)))
        }
      } else {
        localModels.join(mapsC).flatMap{
          case((pid, (localModel, mapC))) =>
            LocalModel.toGlobal(localModel.factorsC, mapC)
        }
      }
    factorsC.cache
    factorsC.count
    DivideAndConquer.logInfo(trainingData, nnzTra, nnzVal, numRows, numCols, bwLog)
    val numIters = new Array[Int](numRowBlocks*numColBlocks)
    new DivideAndConquer(factorsR, factorsC, localModels, 
        trainingData, nnzTra, validatingData, nnzVal,
        mapsR, mapsC, pidsR, pidsC, 
        numRowBlocks, numColBlocks,
        hasRowEC, hasColEC, hasRowPrior, hasColPrior,
        isVB, emBayes, weightedReg, multicore, 0, numIters)
  }
  
  private def train(
      data: RDD[(Int, SparseMatrix)], 
      factorsR: RDD[(Int, Array[Float])], factorsC: RDD[(Int, Array[Float])],
      localModels: RDD[(Int, LocalModel)],
      pidsR: RDD[(Int, Array[Int])], pidsC: RDD[(Int, Array[Int])],
      mapsR: RDD[(Int, Array[Int])], mapsC: RDD[(Int, Array[Int])],
      numIter: Int, stopCrt: Float,
      optType: OptimizerType, regPara: Float, regType: RegularizerType,
      numRowBlocks: Int, numColBlocks: Int,
      hasRowEC: Boolean, hasColEC: Boolean, hasRowPrior: Boolean, hasColPrior: Boolean,
      isVB: Boolean, emBayes: Boolean, weightedReg: Boolean, multicore: Boolean)
    : (RDD[(Int, Array[Float])], RDD[(Int, Array[Float])], RDD[(Int, LocalModel)], 
        Array[Int]) = {
    
    val part = data.partitioner.get
    val disFactorsR = 
      if (hasRowPrior) DivideAndConquer.getPrior(factorsR, pidsR, part)
      else null
    val disFactorsC =
      if (hasColPrior) DivideAndConquer.getPrior(factorsC, pidsC, part)
      else null
    val updatedLocalModels =
      if (hasRowPrior && hasColPrior) {
        data.join(localModels).join(disFactorsR.join(disFactorsC)).mapValues{
          case(((localData, localModel), (factorsR, factorsC))) => {
            val priorsR = DivideAndConquer.toLocal(factorsR, localData.rowMap, null)
            val priorsC = DivideAndConquer.toLocal(factorsC, localData.colMap, null)
            //inplace dual update using sideeffect
            if (hasRowEC) localModel.updateLagR(priorsR, multicore)
            if (hasColEC) localModel.updateLagC(priorsC, multicore)
            localModel.train(localData, optType, numIter, stopCrt, isVB, emBayes,
              weightedReg, multicore, priorsR, priorsC)
          }
        }
      }
      else if (hasRowPrior) {
        data.join(localModels).join(disFactorsR).mapValues{
          case(((localData, localModel), factorsR)) => {
            val priorsR = DivideAndConquer.toLocal(factorsR, localData.rowMap, null)
            if (hasRowEC) localModel.updateLagR(priorsR, multicore)
            localModel.train(localData, optType, numIter, stopCrt, isVB, emBayes,
              weightedReg, multicore, priorsR, priorsC = null)
          }
        }
      }
      else if (hasColPrior) {
        data.join(localModels).join(disFactorsC).mapValues{
          case(((localData, localModel), factorsC)) => {
            val priorsC = DivideAndConquer.toLocal(factorsC, localData.colMap, null)
            if (hasColEC) localModel.updateLagC(priorsC, multicore)
            localModel.train(localData, optType, numIter, stopCrt, isVB, emBayes,
              weightedReg, multicore, priorsR = null, priorsC)
          }
        }
      }
      else {
        data.join(localModels).mapValues{
          case((localData, localModel)) => {
            localModel.train(localData, optType, numIter, stopCrt, isVB, emBayes,
              weightedReg, multicore, priorsR = null, priorsC = null)
          }
        }
      }
    updatedLocalModels.cache
    val numSlices = updatedLocalModels.count.toInt
    val numIters = new Array[Int](numSlices)
    updatedLocalModels.mapValues(_.numIter).collect.foreach{
      case (id, numIter) => numIters(id) = numIter
    }
    // update the global parameters
    val updatedFactorsR = 
      if (numColBlocks > 1) {
        val stats = updatedLocalModels.join(data).flatMap{
          case((pid, (localModel, localData))) => 
            if (weightedReg) {
              localModel.getStatsR(localData.rowMap, localData.row_ptr, multicore)
            }
            else localModel.getStatsR(localData.rowMap, multicore)
        }
        updateGlobalFactors(stats, pidsR, regType, regPara)
      }
      else {
        updatedLocalModels.join(mapsR).flatMap{
          case((pid, (localModel, mapR))) =>
            LocalModel.toGlobal(localModel.factorsR, mapR)
        }
      }
    updatedFactorsR.cache
    updatedFactorsR.count
    if (factorsR != null && hasRowPrior) factorsR.unpersist(true)
    val updatedFactorsC = 
      if (numRowBlocks > 1) {
        val stats = updatedLocalModels.join(data).flatMap{
          case((pid, (localModel, localData))) => 
            if (weightedReg) {
              localModel.getStatsC(localData.colMap, localData.col_ptr, multicore)
            }
            else localModel.getStatsC(localData.colMap, multicore)
        }
        updateGlobalFactors(stats, pidsC, regType, regPara)
      } else {
        updatedLocalModels.join(mapsC).flatMap{
          case((pid, (localModel, mapC))) =>
            LocalModel.toGlobal(localModel.factorsC, mapC)
        }
      }
    updatedFactorsC.cache
    updatedFactorsC.count
    if (factorsC != null && hasColPrior) factorsC.unpersist(true)
    localModels.unpersist(true)
    (updatedFactorsR, updatedFactorsC, updatedLocalModels, numIters)
  }
  
  def updateGlobalFactors(stats: RDD[(Int, (Vector, Vector))], 
      pids: RDD[(Int, Array[Int])], regType: RegularizerType, regPara: Float) = {
    stats.reduceByKey((p1, p2) => ((p1._1.+=(p2._1), (p1._2.+=(p2._2))))).join(pids)
    .mapValues{
      case(((numeVec, denoVec), pids)) => {
        val nume = numeVec.elements
        val deno = denoVec.elements
        if (pids.length <= 1) l2Update(nume, deno, Float.MaxValue, nume)
        else {
          regType match {
            case L1 => l1Update(nume, deno, regPara, nume)
            case Max => maxNormUpdate(nume, deno, regPara, nume)
            case L2 => l2Update(nume, deno, regPara, nume)
          }
        }
        nume
      }
    }
  }
  
  def maxNormUpdate(nume: Array[Float], deno: Array[Float], regPara: Float, 
      result: Array[Float]) {
    val l = nume.length
    var j = 0
    var norm = 0f
    while (j < l) {
      result(j) = nume(j) / deno(j)
      norm += result(j)*result(j)
      j += 1
    }
    if (norm > regPara) {
      val ratio = math.sqrt(regPara/norm).toFloat
      j = 0
      while (j < l) {
        result(j) = result(j)*ratio
        j += 1 
      }
    }
  }
  
  def l2Update(nume: Array[Float], deno: Array[Float], regPara: Float, 
      result: Array[Float]) {
    var j = 0
    while (j < result.length) { 
      result(j) = nume(j) / (deno(j)+regPara)
      j += 1 
    }
  }
  
  def l1Update(nume: Array[Float], deno: Array[Float], regPara: Float,
      result: Array[Float]) = {
    var j = 0
    while (j < result.length) {
      result(j) = 
        if (nume(j) > regPara) (nume(j)-regPara)/deno(j)
        else if (nume(j) < -regPara) (nume(j)+regPara)/deno(j)
        else 0
      j += 1 
    }
  }
  
  def getPrior(factors: RDD[(Int, Array[Float])], pids: RDD[(Int, Array[Int])],
      part: Partitioner): RDD[(Int, Array[(Int, Array[Float])])] = {
    val filtered = factors.join(pids).filter(_._2._2.length > 1).mapValues(_._1)
    Model.distribute(filtered, pids, part)
  }
  
  def toLocal(factors: Array[(Int, Array[Float])], map: Array[Int], fill: Array[Float])
    : Array[Array[Float]] = {
    val numFactors = factors(0)._2.length
    val result = Array.ofDim[Array[Float]](map.length)
    var i = 0
    var j = 0
    while (i < map.length && j < factors.length) {
      if (map(i) == factors(j)._1) { 
        result(i) = factors(j)._2
        i += 1
        j += 1 
      } else if (map(i) < factors(j)._1) {
        result(i) = fill
        i += 1
      } else {
        j += 1
      }
    }
    while (i < map.length) { 
      result(i) = Array.ofDim[Float](numFactors)
      i+=1
    }
    result
  }
  
  def getGamma(gammas: RDD[Array[Float]], longGamma: Boolean): Array[Float] = {
    if (longGamma) {
      gammas.map(Vector(_).getMean).collect
    } else {
      {gammas.map(Vector(_)).reduce(_+_)/gammas.count}.elements
    }
  }
}