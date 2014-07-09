package tm

import breeze.linalg._
import breeze.numerics._
import breeze.stats._

import org.apache.commons.math3.util.FastMath

import utilities.MathFunctions

class Document(val content: SparseVector[Int], val gamma: DenseVector[Double]) 
  extends Serializable {
  
  val length = content.activeSize
  val numTopics = gamma.length
  val wordIndices = content.index
  val counts = content.data
  val nnz = counts.sum
  
  private def getPhiNorm(expELogTheta: DenseVector[Double], 
      expELogBeta: DenseMatrix[Double], phiNorm: DenseVector[Double]) {
    var i = 0
    while (i < length) {
      phiNorm(i) = expELogTheta.dot(expELogBeta(::, wordIndices(i)))
      if (phiNorm(i) < 1e-30) phiNorm(i) = 1e-30
      i += 1
    }
  }
  
  private def getLogPhiNorm(eLogTheta: DenseVector[Double], 
      eLogBeta: DenseMatrix[Double], logPhiNorm: DenseVector[Double]) = {
    var i = 0
    while (i < length) {
      logPhiNorm(i) = eLogTheta(0) + eLogBeta(0, wordIndices(i))
      var k = 1
      while (k < numTopics) {
        val logPhi = eLogTheta(k) + eLogBeta(k, wordIndices(i))
        logPhiNorm(i) = MathFunctions.log_sum(logPhiNorm(i), logPhi)
        k += 1
      }
      i += 1
    }
  }
  
  def getSuffStats(expELogTheta: DenseVector[Double], phiNorm: DenseVector[Double],
      suffStats: DenseMatrix[Double]) = {
    var i = 0
    while (i < length) {
      suffStats(::, wordIndices(i)) :+= expELogTheta:*(counts(i)/phiNorm(i))
      i += 1
    }
  }
  
  def updateEta(eLogTheta: DenseVector[Double], logPhiNorm: DenseVector[Double],
      eLogBeta: DenseMatrix[Double], eta: DenseMatrix[Double]) = {
    var i = 0
    while (i < length) {
      eta(::, wordIndices(i)) :+= 
        exp(eLogTheta + eLogBeta(::, wordIndices(i)) - logPhiNorm(i)):*(counts(i)*1.0)
      i += 1
    }
  }
  
  def updateGamma(numIter: Int, alpha: DenseVector[Double], 
      expELogBeta: DenseMatrix[Double]): Document = {
    val expELogTheta = DenseVector.zeros[Double](numTopics)
    val phiNorm = DenseVector.zeros[Double](length)
    updateGamma(numIter, alpha, expELogBeta, expELogTheta, phiNorm)
  }
  
  def updateGamma(numIter: Int, alpha: DenseVector[Double], 
      expELogBeta: DenseMatrix[Double], expELogTheta: DenseVector[Double],
      phiNorm: DenseVector[Double]): Document = {
    val wordIndices = content.index
    val counts = content.data
    expELogTheta := MathFunctions.eDirExp(gamma)
    val lastGamma = DenseVector.zeros[Double](numTopics)
    val dotProduct = DenseVector.zeros[Double](numTopics)
    getPhiNorm(expELogTheta, expELogBeta, phiNorm)
    var inner = 0
    while (inner < numIter && (mean(abs(gamma - lastGamma)) > 0.001)) {
      lastGamma := gamma
      dotProduct := expELogBeta(::, wordIndices(0)):*(counts(0)/phiNorm(0))
      var i = 1
      while (i < length) {
        dotProduct :+= expELogBeta(::, wordIndices(i)):*(counts(i)/phiNorm(i))
        i += 1
      }
      gamma := alpha + (expELogTheta:*dotProduct)
      expELogTheta := MathFunctions.eDirExp(gamma)
      getPhiNorm(expELogTheta, expELogBeta, phiNorm)
      inner += 1
    }
    this
  }
  
  def updateGamma2(numIter: Int, alpha: DenseVector[Double], 
      eLogBeta: DenseMatrix[Double]): Document = {
    val eLogTheta = DenseVector.zeros[Double](numTopics)
    val logPhiNorm = DenseVector.zeros[Double](length)
    updateGamma2(numIter, alpha, eLogBeta, eLogTheta, logPhiNorm)
  }
  
  def updateGamma2(numIter: Int, alpha: DenseVector[Double], 
      eLogBeta: DenseMatrix[Double], eLogTheta: DenseVector[Double],
      logPhiNorm: DenseVector[Double]): Document = {
    val wordIndices = content.index
    val counts = content.data
    eLogTheta := MathFunctions.dirExp(gamma)
    val lastGamma = DenseVector.zeros[Double](numTopics)
    getLogPhiNorm(eLogTheta, eLogBeta, logPhiNorm)
    var inner = 0
    while (inner < numIter && (mean(abs(gamma - lastGamma)) > 0.001)) {
      lastGamma := gamma
      gamma := alpha
      var i = 0
      while (i < length) {
        val count = 1.0*counts(i)
        gamma :+= exp(eLogTheta + eLogBeta(::, wordIndices(i)) - logPhiNorm(i)):*count
        i += 1
      }
      eLogTheta := MathFunctions.dirExp(gamma)
      getLogPhiNorm(eLogTheta, eLogBeta, logPhiNorm)
      inner += 1
    }
    this
  }
  
  def getLLH(eBeta: DenseMatrix[Double]): Double = {
    val eTheta = gamma:/sum(gamma)
    var i = 0
    var score = 0.0
    while (i < length) {
      score += log(eTheta dot eBeta(::, wordIndices(i)))*counts(i)
      i += 1
    }
    score
  }
  
  def getScore(eLogTheta: DenseVector[Double], eLogBeta: DenseMatrix[Double])
    : Double = {
    var i = 0
    var score = 0.0
    while (i < length) {
      val logPhiDW = eLogTheta + eLogBeta(::, wordIndices(i))
      val maxLogPhiDw = max(logPhiDW)
      score += (log(sum(exp(logPhiDW - maxLogPhiDw))) + maxLogPhiDw)*counts(i)
      i += 1
    }
    score
  }  
}