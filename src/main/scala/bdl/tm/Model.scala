package tm

import breeze.linalg._
import breeze.numerics._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext

import utilities.MathFunctions

abstract class Model (val topics: DenseMatrix[Double], val alpha: DenseVector[Double],
    val beta: Double) extends Serializable {

  def train(numIters: Int, emBayes: Boolean): Model
  
  def getTraininingPerplexity: Double
  
  def getValidatingPerplexity(foldingin: Int): Double
  
  def getTrainingLLH: Double
  
  def getValidatingLLH(foldingin: Int): Double
}

object Model {
  
  def getLogLikelihood(foldingin: Int, docsRDD: RDD[(Int, Array[Document])], 
      eta: DenseMatrix[Double], alpha: DenseVector[Double], multicore: Boolean)
    : Double = {
    val sc = docsRDD.context
    val etaBC = sc.broadcast(eta)
    val llh = docsRDD.map{case (bid, docs) =>
      LDA.getLLH(foldingin, etaBC.value, docs, alpha, multicore)
    }.sum
    llh / docsRDD.map(_._2.map(_.nnz).sum).sum
  }
  
  def getPerplexity(foldingin: Int, docsRDD: RDD[(Int, Array[Document])], 
      eta: DenseMatrix[Double], alpha: DenseVector[Double], multicore: Boolean)
    : Double = {
    val llh = getLogLikelihood(foldingin, docsRDD, eta, alpha, multicore)
    exp(-llh) 
  }
  
  def getELBO(foldingin: Int, docsRDD: RDD[(Int, Array[Document])], 
      eta: DenseMatrix[Double], alpha: DenseVector[Double], 
      beta0: Double, multicore: Boolean): Double = {
    val numTopics = eta.rows
    val numWords = eta.cols
    val numDocs = docsRDD.map(_._2.length).sum
    val sc = docsRDD.context
    val etaBC = sc.broadcast(eta)
    val docScores = docsRDD.map{case (bid, docs) =>
      val eta = etaBC.value
      val eLogBeta = MathFunctions.dirExp(eta, multicore)
      val expELogBeta = if (foldingin > 0) exp(eLogBeta) else null
      if (multicore) {
        docs.par.map{doc => 
          val fTheta = DenseVector.zeros[Double](numTopics)
          val phiNorm = DenseVector.zeros[Double](doc.length)
          if (foldingin > 0) {
            doc.updateGamma(foldingin, alpha, expELogBeta, fTheta, phiNorm)
            fTheta := breeze.numerics.log(fTheta)
          } else {
            fTheta := MathFunctions.dirExp(doc.gamma)
          }
          doc.getScore(fTheta, eLogBeta) + LDA.getScore(doc.gamma, fTheta, alpha)
        }.sum
      } else {
        val fTheta = DenseVector.zeros[Double](numTopics)
        val maxLength = docs.map(_.length).max
        val phiNorm = DenseVector.zeros[Double](maxLength)
        docs.map{doc => 
          if (foldingin > 0) {
            doc.updateGamma(foldingin, alpha, expELogBeta, fTheta, phiNorm)
            fTheta := breeze.numerics.log(fTheta)
          } else {
            fTheta := MathFunctions.dirExp(doc.gamma)
          }
          doc.getScore(fTheta, eLogBeta) + LDA.getScore(doc.gamma, fTheta, alpha)
        }.sum
      }
    }.sum + numDocs*(lgamma(sum(alpha)) - sum(lgamma(alpha))) 
    val topicScores = (0 until numTopics).map{ k =>
      val eLogBetaK = MathFunctions.dirExp(eta(k, ::).t)
      LDA.getScore(eta(k, ::).t, eLogBetaK, beta0) + 
        lgamma(numWords*beta0) - numWords*(lgamma(beta0))
    }.sum
    docScores + topicScores
  }
  
}