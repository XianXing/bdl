package tm

import breeze.linalg._
import breeze.numerics._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkContext, Partitioner, HashPartitioner}
import org.apache.spark.broadcast.Broadcast

import org.apache.commons.math3.distribution._

import preprocess.TM._
import utilities.MathFunctions
import utilities.SparseVector

class MRLDA(
    val trainingDocs: RDD[(Int, Array[Document])],
    val validatingDocs: RDD[(Int, Array[Document])],
    override val topics: DenseMatrix[Double],
    override val alpha: DenseVector[Double],
    override val beta: Double) extends Model (topics, alpha, beta) {
  
  private def createNewModel(updatedTopics: DenseMatrix[Double]): MRLDA = {
    new MRLDA(trainingDocs, validatingDocs, updatedTopics, alpha, beta)
  }
  
  def train(numIters: Int, updateAlpha: Boolean): MRLDA = {
    val updatedTopics = 
      MRLDA.train(numIters, trainingDocs, topics, alpha, beta, updateAlpha)
    createNewModel(updatedTopics)
  }
  
  def getTrainingLLH: Double = {
    Model.getLogLikelihood(0, trainingDocs, topics, alpha, false)
  }
  
  def getTraininingPerplexity: Double = {
    Model.getPerplexity(0, trainingDocs, topics, alpha, false)
  }
  
  def getValidatingPerplexity(foldingin: Int): Double = {
    Model.getPerplexity(foldingin, validatingDocs, topics, alpha, false)
  }
  
  def getValidatingLLH(foldingin: Int): Double = {
    Model.getLogLikelihood(foldingin, validatingDocs, topics, alpha, false)
  }
}

object MRLDA {
  
  def apply(sc: SparkContext, partitioner: Partitioner,
      trainingDir: String, validatingDir: String,
      numTopics: Int, numWords: Int, numBlocks: Int, seed: Int, 
      alphaInit: Double, betaInit: Double): MRLDA = {
    val trainingDocs = sc.textFile(trainingDir, numBlocks).zipWithUniqueId
      .map{case(line, idx) => ((idx % numBlocks).toInt, toSparseVector(line))}
      .groupByKey(numBlocks).mapValues(content => toCorpus(content.toArray, numTopics))
      .cache
    trainingDocs.count
    val validatingDocs = sc.textFile(validatingDir, numBlocks).zipWithUniqueId
      .map{case(line, id) => ((id % numBlocks).toInt, toSparseVector(line))}
      .groupByKey(numBlocks).mapValues(content => toCorpus(content.toArray, numTopics))
      .cache
    validatingDocs.count
    val gd = new GammaDistribution(10, 1.0/10)
    gd.reseedRandomGenerator(seed)
    val topics = DenseMatrix.fill(numTopics, numWords)(gd.sample)
    val alpha = DenseVector.fill(numTopics, alphaInit)
    new MRLDA(trainingDocs, validatingDocs, topics, alpha, betaInit)
  }
  
  def train(numIters: Int, trainingDocs: RDD[(Int, Array[Document])],
      topics: DenseMatrix[Double], alpha: DenseVector[Double], 
      beta: Double, emBayes: Boolean): DenseMatrix[Double] = {
    
    val sc = trainingDocs.context
    val numTopics = topics.rows
    val numWords = topics.cols
    val expELogBeta = DenseMatrix.zeros[Double](numTopics, numWords)
    for (iter <- 0 until numIters) {
      MathFunctions.eDirExp(topics, multicore = true, expELogBeta)
      val expELogBetaBC = sc.broadcast(expELogBeta)
      val suffStats = trainingDocs.map{case (bid, docs) =>
        val expELogBeta = expELogBetaBC.value
        val expELogTheta = DenseVector.zeros[Double](numTopics)
        val maxNNZ = docs.map(_.nnz).max
        val phiNorm = DenseVector.zeros[Double](maxNNZ)
        val suffStats = DenseMatrix.zeros[Double](numTopics, numWords)
        docs.map{doc => 
          doc.updateGamma(numIter = 5, alpha, expELogBeta, expELogTheta, phiNorm)
          doc.getSuffStats(expELogTheta, phiNorm, suffStats)
        }
        suffStats
      }.reduce(_:+=_)
      topics := (suffStats:*expELogBeta) + beta
    }
    topics
  }
}