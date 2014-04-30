package tm

import breeze.linalg._
import breeze.numerics._
import breeze.stats._

import utilities.MathFunctions

class Document(val content: SparseVector[Int], val gamma: DenseVector[Double]) {
  
  val length = content.activeSize
  val numTopics = gamma.length
  val wordIndices = content.index
  val counts = content.data
  
  private def getPhiNorm(expELogTheta: DenseVector[Double], 
      expELogBeta: DenseMatrix[Double], phiNorm: DenseVector[Double]) {
    var i = 0
    while (i < length) {
      phiNorm(i) = expELogTheta.dot(expELogBeta(::, wordIndices(i))) + 1e-100
      i += 1
    }
  }
  
  def updateGamma(numIter: Int, alpha: DenseVector[Double], 
      expELogBeta: DenseMatrix[Double], phiNorm: DenseVector[Double], 
      suffStats: DenseMatrix[Double]) = {
    val wordIndices = content.index
    val counts = content.data
    val expELogTheta = DenseVector.zeros[Double](numTopics)
    expELogTheta := MathFunctions.eDirExp(gamma)
    getPhiNorm(expELogTheta, expELogBeta, phiNorm)
    val zeros = DenseVector.zeros[Double](numTopics)
    val lastGamma = DenseVector.zeros[Double](numTopics)
    val dotProduct = DenseVector.zeros[Double](numTopics)
    val docLength = wordIndices.length
    var inner = 0
    while (inner < numIter && (mean(abs(gamma - lastGamma)) > 0.001)) {
      lastGamma := gamma
      gamma := alpha
      var i = 0
      while (i < docLength) {
        dotProduct :+= expELogBeta(::, wordIndices(i)):*(counts(i)/phiNorm(i))
        i += 1
      }
      gamma :+= expELogTheta:*dotProduct
      expELogTheta := MathFunctions.eDirExp(gamma)
      getPhiNorm(expELogTheta, expELogBeta, phiNorm)
      dotProduct := zeros
      inner += 1
    }
    var i = 0
    while (i < docLength) {
      suffStats(::, wordIndices(i)) :+= expELogTheta:*(counts(i)/phiNorm(i))
      i += 1
    }
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