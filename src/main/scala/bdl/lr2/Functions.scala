package lr2

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import utilities.SparseMatrix
import utilities.SparseVector
import utilities.DoubleVector
import utilities.IntVector
import utilities.MathFunctions
import classification.RegularizerType._

object Functions {
  
  def toLocal(globalWeights: Array[Double], map: Array[Int]) = {
    val localNumFeatures = map.length
    Array.tabulate(localNumFeatures)(i => globalWeights(map(i)))
  }
  
  def toGlobal(localWeights: Array[Double], map: Array[Int], length: Int) = {
    val globalWeights = new Array[Double](length)
    var i = 0
    while (i < map.length) {
      globalWeights(map(i)) = localWeights(i)
      i += 1
    }
    globalWeights
  }
  
  def getYWTX(labels: Array[Byte], features: SparseMatrix, weights: Array[Double],
      ywtx: Array[Double]) {
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val isBinary = value == null
    val numFeatures = features.numRows
    var p = 0
    if (isBinary) {
      while (p < numFeatures) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          ywtx(n) += labels(n)*weights(p)
          i += 1
        }
        p += 1
      }
    } else {
      while (p < numFeatures) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          ywtx(n) += labels(n)*value(i)*weights(p)
          i += 1
        }
        p += 1
      }
    }
  }
  
  def getWTX(features: SparseMatrix, weights: Array[Double], wtx: Array[Double]) {
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val isBinary = value == null
    val numFeatures = features.numRows
    var p = 0
    if (isBinary) {
      while (p < numFeatures) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          wtx(n) += weights(p)
          i += 1
        }
        p += 1
      }
    } else {
      while (p < numFeatures) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          wtx(n) += value(i)*weights(p)
          i += 1
        }
        p += 1
      }
    }
  }
  
  def getGrad(labels: Array[Byte], features: SparseMatrix, weights: Array[Double],
      gradient: Array[Double]): Array[Double] = {
    val numData = features.numCols
    val numFeatures = features.numRows
    val ywtx = new Array[Double](numData)
    getYWTX(labels, features, weights, ywtx)
    //reuse weights
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val isBinary = value == null
    var n = 0 
    while (n < numData) {
      ywtx(n) = MathFunctions.sigmoid(ywtx(n))
      n += 1
    }
    var p = 0
    if (isBinary) {
      while (p < numFeatures) {
        var i = ptr(p)
        gradient(p) = 0
        while (i < ptr(p+1)) {
          val n = idx(i)
          gradient(p) += labels(n)*(1 - ywtx(n))
          i += 1
        }
        p += 1
      }
    } else {
      while (p < numFeatures) {
        var i = ptr(p)
        gradient(p) = 0
        while (i < ptr(p+1)) {
          val n = idx(i)
          gradient(p) += labels(n)*(1 - ywtx(n))*value(i)
          i += 1
        }
        p += 1
      }
    }
    gradient
  }
  
  def getHessian(features: SparseMatrix, w: Array[Double], u: Array[Double]) 
    : Double = {
    
    val numData = features.numCols
    val numFeatures = features.numRows
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val isBinary = value == null
    val wtx = new Array[Double](numData)
    val utx = new Array[Double](numData)
    var p = 0
    if (isBinary) {
      while (p < numFeatures) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          wtx(n) += w(p)
          utx(n) += u(p)
          i += 1
        }
        p += 1
      }
    } else {
      while (p < numFeatures) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          wtx(n) += w(p)*value(i)
          utx(n) += u(p)*value(i)
          i += 1
        }
        p += 1
      }
    }
    var n = 0
    var hessian = 0.0
    while (n < numData) {
      val sigmoid = MathFunctions.sigmoid(wtx(n))
      val alpha = sigmoid*(1-sigmoid)
      if (alpha > 1e-5f) hessian += alpha*utx(n)*utx(n)
      else hessian += 1e-5f*utx(n)*utx(n)
      n += 1
    }
    hessian
  }
  
  def calculateAUC(
      sc: SparkContext, 
      dataPos: RDD[SparseVector], numPos: Int,
      dataNeg: RDD[SparseVector], numNeg: Int,
      weights: Array[Double]): Double = {
    
    def getBinPred(features: SparseVector, weights:Array[Double], numBins: Int)
      : IntVector = {
      val prob = MathFunctions.sigmoid(features.dot(DoubleVector(weights)))
      val inc = 1.0/numBins
      IntVector(Array.tabulate(numBins)(i => if (prob > (i-1)*inc) 1 else 0 ))
    }
    
    def getAUC(tpr: Array[Double], fpr: Array[Double]): Double = {
      assert(tpr.length == fpr.length)
      var tpr_prev = 0.0
      var fpr_prev = 0.0
      var auc = 0.0
      for (i <- tpr.length-1 to 0 by -1) {
        auc += 0.5*(tpr(i)+tpr_prev)*(fpr(i)-fpr_prev)
        tpr_prev = tpr(i)
        fpr_prev = fpr(i)
      }
      auc
    }
    
    val weightsBC = sc.broadcast(weights)
    val tpr = dataPos.map(feature => getBinPred(feature, weightsBC.value, 500))
      .reduce(_+=_).toArray.map(1.0*_/numPos)
    val fpr = dataNeg.map(feature => getBinPred(feature, weightsBC.value, 500))
      .reduce(_+=_).toArray.map(1.0*_/numNeg)
    getAUC(tpr, fpr)
  }
  
  def calculateLLH(
      sc: SparkContext, 
      data: RDD[(Byte, SparseVector)],
      weights: Array[Double]): Double = {
    val weightsBC = sc.broadcast(weights)
    data.map{
      case (label, feature) => {
        val weights = weightsBC.value
        val yxtw = feature.dot(weights)*label
        if (yxtw > -10) -math.log(1 + math.exp(-yxtw))
        else yxtw
      }
    }.sum
  }
  
  def calculateOBJ(
      sc: SparkContext, 
      data: RDD[(Int, (Array[Byte], SparseMatrix))],
      weights: Array[Double],
      regPara: Double,
      regType: RegularizerType): Double = {
    
    val norm = 
      if (regType == L1) Functions.getL1Norm(weights)*regPara
      else if (regType == L2) Functions.getL2Norm(weights)*regPara/2
      else 0
    
    data.map{
      case(bid, (labels, features)) => {
        val numData = features.numCols
        val ywtx = new Array[Double](numData)
        val ptr = features.row_ptr
        val idx = features.col_idx
        val value = features.value_r
        var n = 0
        var obj = 0.0
        while (n < numData) {
          val exp = math.exp(-ywtx(n))
          if (ywtx(n) < -10) obj -= ywtx(n)
          else if (ywtx(n) > -10 && ywtx(n) < 10) obj += math.log(1 + exp)
          n += 1
        }
        obj
      }
    }.sum + norm
  }
  
  def getL1Norm(weights: Array[Double]) = weights.map(math.abs(_)).sum
  def getL2Norm(weights: Array[Double]) = weights.map(w => w*w).sum
  
}