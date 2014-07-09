package tm

import breeze.linalg._
import breeze.numerics._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkContext, Partitioner, HashPartitioner}
import org.apache.spark.broadcast.Broadcast

import preprocess.TM._
import utilities.MathFunctions
import utilities.SparseVector

class DivideAndConquer (
    val trainingDocs: RDD[(Int, Array[Document])],
    val validatingDocs: RDD[(Int, Array[Document])],
    val localModels: RDD[(Int, LDA)],
    val duals: RDD[(Int, DenseMatrix[Double])],
    override val topics: DenseMatrix[Double],
    override val alpha: DenseVector[Double],
    override val beta: Double,
    val hasEC: Boolean, val multicore: Boolean) 
    extends Model(topics, alpha, beta) {
  
  private def createNewModel(
      updatedLocalModels: RDD[(Int, LDA)], 
      updatedDuals: RDD[(Int, DenseMatrix[Double])],
      updatedTopics: DenseMatrix[Double]): DivideAndConquer = {
    new DivideAndConquer(trainingDocs, validatingDocs,
        updatedLocalModels, updatedDuals, updatedTopics, alpha, beta,
        hasEC, multicore)
  }
  
  def train(numOuterIters: Int, updateAlpha: Boolean): DivideAndConquer = {
    val (updatedLocalModels, updatedDuals, updatedTopics) = 
      DivideAndConquer.train(numOuterIters, 5, trainingDocs, localModels, 
          duals, topics, beta, hasEC, updateAlpha, multicore)
    createNewModel(updatedLocalModels, updatedDuals, updatedTopics)
  }
  
  def getTrainingLLH: Double = {
    Model.getLogLikelihood(0, trainingDocs, topics, alpha, multicore)
  }
  
  def getTraininingPerplexity: Double = {
    Model.getPerplexity(0, trainingDocs, topics, alpha, multicore)
  }
  
  def getValidatingPerplexity(foldingin: Int): Double = {
    Model.getPerplexity(foldingin, validatingDocs, topics, alpha, multicore)
  }
  
  def getValidatingLLH(foldingin: Int): Double = {
    Model.getLogLikelihood(foldingin, validatingDocs, topics, alpha, multicore)
  }
}

object DivideAndConquer {
  
  def calculate = {
    val v = DenseVector(1.0, 2.0)
    v :^ 2.0
  }
  
  def apply(sc: SparkContext, partitioner: Partitioner,
      trainingDir: String, validatingDir: String,
      numTopics: Int, numWords: Int, numBlocks: Int, ratio: Double, seed: Int,
      alphaInit: Double, betaInit: Double, hasEC: Boolean, multicore: Boolean)
    : DivideAndConquer = {
    
    val docs = toCorpus(sc.textFile(trainingDir, numBlocks)
        .sample(false, ratio, seed).map(toSparseVector(_)).collect, numTopics)
    val lda = LDA(numTopics, numWords, alphaInit, seed)
    val beta0 = DenseMatrix.fill(numTopics, numWords)(betaInit)
    lda.runVB(outerIters = 20, innerIters = 10, docs, beta0, 
        multicore = true, updateAlpha = false)
    val eta = lda.eta
    val alpha = DenseVector.fill(numTopics, alphaInit)
    val alphaBC = sc.broadcast(alpha)
    val expELogBeta = MathFunctions.eDirExp(eta, multicore)
    val expELogBetaBC = sc.broadcast(expELogBeta)
    val trainingDocs = sc.textFile(trainingDir, numBlocks).zipWithUniqueId
      .map{case(line, idx) => ((idx % numBlocks).toInt, toSparseVector(line))}
      .groupByKey(partitioner)
      .mapValues{contents =>
        toCorpus(contents.toArray, numTopics)
          .map(_.updateGamma(numIter = 10, alphaBC.value, expELogBetaBC.value))
      }.partitionBy(partitioner).cache
    trainingDocs.count
    val validatingDocs = sc.textFile(validatingDir, numBlocks).zipWithUniqueId
      .map{case(line, id) => ((id % numBlocks).toInt, toSparseVector(line))}
      .groupByKey(partitioner)
      .mapValues(content => toCorpus(content.toArray, numTopics))
      .partitionBy(partitioner).cache
    validatingDocs.count
    
    val etaBC = sc.broadcast(eta)
    val localModels = trainingDocs.mapValues{_ => 
      if (multicore) new LDA(etaBC.value, alphaBC.value)
      else new LDA(etaBC.value.copy, alphaBC.value.copy)
    }.cache
    localModels.count
    val duals = 
      if (hasEC) {
        trainingDocs.mapValues(_ => 
          DenseMatrix.zeros[Double](numTopics, numWords)).cache
      }
      else null
    if (hasEC) duals.count
    new DivideAndConquer(trainingDocs, validatingDocs, 
      localModels, duals, eta, alpha, betaInit, hasEC, multicore)
  }
  
  def train(numOuterIters: Int, numInnerIters: Int, 
      trainingDocs: RDD[(Int, Array[Document])],
      localModels: RDD[(Int, LDA)], duals: RDD[(Int, DenseMatrix[Double])],
      topics: DenseMatrix[Double], beta: Double,
      hasEC: Boolean, updateAlpha: Boolean, multicore: Boolean) = {
    val sc = trainingDocs.context
    val numBlocks = localModels.count.toInt
    val topicsBC = sc.broadcast(topics)
    val updatedDuals = 
      if (hasEC) {
        localModels.join(duals).mapValues{
          case (lda, dual) => updateDual(multicore, lda.eta, topicsBC.value, dual)
          dual
        }.cache
      }
      else duals
    val updatedLocalModels = 
      if (hasEC) {
        trainingDocs.join(localModels).join(updatedDuals).mapValues{
        case ((docs, lda), dual) => 
          val beta = topicsBC.value - dual
          lda.runVB(numOuterIters, numInnerIters, docs, beta, multicore, updateAlpha)
          lda
        }.cache
      } else {
        trainingDocs.join(localModels).mapValues{
        case (docs, lda) => 
          val beta = topicsBC.value
          lda.runVB(numOuterIters, numInnerIters, docs, beta, multicore, updateAlpha)
          lda
        }.cache
      }
    val suffStats = 
      if (hasEC) {
        updatedLocalModels.join(updatedDuals).map{case (bid, (lda, dual)) => 
          if (multicore) MathFunctions.parallelSum(lda.eta, dual)
          else lda.eta + dual
        }.reduce(_:+=_)
      } else {
        updatedLocalModels.map(_._2.eta).reduce(_:+_)
      }
//    topics := (suffStats + beta):/(numBlocks + 1.0)
    topics := (suffStats + beta):/(numBlocks + 1.0)
    localModels.unpersist(true)
    if (hasEC) duals.unpersist(true)
    (updatedLocalModels, updatedDuals, topics)
  }
  
  def updateDual(multicore: Boolean, eta: DenseMatrix[Double], 
      prior: DenseMatrix[Double], dual: DenseMatrix[Double]) {
    val numTopics = prior.rows
    if (multicore) {
      for (k <- (0 until numTopics).par) {
        dual(k, ::) :+= eta(k, ::) - prior(k, ::)
      }
    } else {
      for (k <- 0 until numTopics) {
        dual(k, ::) :+= eta(k, ::) - prior(k, ::)
      }
    }
  }
}