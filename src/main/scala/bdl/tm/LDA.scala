package tm

import java.io._

import scala.io.Source
import scala.util.Sorting

import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean
import org.apache.commons.math3.distribution._

import utilities.MathFunctions

class LDA(val numTopics: Int, 
          val gamma: DenseMatrix[Double], 
          val eta: DenseMatrix[Double],
          val alpha: DenseVector[Double]) {
  
  private def updatePhiD(wordIndices: DenseVector[Int], eLogThetaD: DenseVector[Double],
      eLogBeta: DenseMatrix[Double], phiD: DenseMatrix[Double]) {
    val docLength = wordIndices.length
    var i = 0
    while (i < docLength) {
      val phiDW = eLogThetaD + eLogBeta(::, wordIndices(i))
      phiDW := exp(phiDW - max(phiDW))
      phiDW :/= sum(phiDW)
      phiD(::, i) := phiDW
      i += 1
    }
  }
  
  private def update(numIter: Int, wordIndices: DenseVector[Int], 
      counts: DenseVector[Int], alpha: DenseVector[Double], 
      eLogBeta: DenseMatrix[Double], phiD: DenseMatrix[Double], 
      eLogThetaD: DenseVector[Double], eta: DenseMatrix[Double], 
      gammaD: DenseVector[Double]) = {
    MathFunctions.dirExp(gammaD, eLogThetaD)
    updatePhiD(wordIndices, eLogThetaD, eLogBeta, phiD)
    val lastGammaD = DenseVector.zeros[Double](numTopics)
    val docLength = counts.length
    val doubleCounts = DenseVector.tabulate[Double](docLength)(counts(_))
    for (inner <- 0 until numIter if (mean(abs(gammaD - lastGammaD)) > 0.001)) {
      lastGammaD := gammaD
      gammaD := alpha + (phiD*doubleCounts)
      MathFunctions.dirExp(gammaD, eLogThetaD)
      updatePhiD(wordIndices, eLogThetaD, eLogBeta, phiD)
    }
    var i = 0
    while (i < docLength) {
      eta(::, wordIndices(i)) :+= phiD(::, i):*doubleCounts(i)
      i += 1
    }
  }
  
  private def getPhiNorm(wordIndices: DenseVector[Int], 
      expELogThetaD: DenseVector[Double], expELogBeta: DenseMatrix[Double], 
      phiNorm: DenseVector[Double]) {
    val length = wordIndices.length
    var i = 0
    while (i < length) {
      phiNorm(i) = expELogThetaD.dot(expELogBeta(::, wordIndices(i))) + 1e-100
      i += 1
    }
  }
  
  private def updateGammaD(
      numIter: Int, wordIndices: DenseVector[Int], counts: DenseVector[Int],
      alpha: DenseVector[Double], expELogBeta: DenseMatrix[Double], 
      phiNormD: DenseVector[Double], expELogThetaD: DenseVector[Double],
      gammaD: DenseVector[Double]) = {
    
    MathFunctions.eDirExp(gammaD, expELogThetaD)
    getPhiNorm(wordIndices, expELogThetaD, expELogBeta, phiNormD)
    val zeros = DenseVector.zeros[Double](numTopics)
    val lastGammaD = DenseVector.zeros[Double](numTopics)
    val dotProduct = DenseVector.zeros[Double](numTopics)
    val docLength = wordIndices.length
    for (inner <- 0 until numIter if (mean(abs(gammaD - lastGammaD)) > 0.001)) {
      lastGammaD := gammaD
      gammaD := alpha
      var i = 0
      while (i < docLength) {
        dotProduct :+= expELogBeta(::, wordIndices(i)):*(counts(i)/phiNormD(i))
        i += 1
      }
      gammaD :+= expELogThetaD:*dotProduct
      MathFunctions.eDirExp(gammaD, expELogThetaD)
      getPhiNorm(wordIndices, expELogThetaD, expELogBeta, phiNormD)
      dotProduct := zeros
    }
  }
  
  private def updateEtaK(docs: CSCMatrix[Int], beta0K: DenseVector[Double], 
      phiNorm: DenseVector[Double], expELogThetaK: DenseVector[Double], 
      expELogBetaK: DenseVector[Double], etaK: DenseVector[Double]) = {
    val docPtrs = docs.colPtrs
    val wordIndices = DenseVector(docs.rowIndices)
    val counts = DenseVector(docs.data)
    val numDocs = docs.cols
    val numWords = docs.rows
    var w = 0
    while (w < numWords) {
      etaK(w) = 0
      w += 1
    }
    var d = 0
    while (d < numDocs) {
      var ii = docPtrs(d)
      while (ii < docPtrs(d+1)) {
        etaK(wordIndices(ii)) += expELogThetaK(d)*counts(ii)/phiNorm(ii)
        ii += 1
      }
      d += 1
    }
    w = 0
    while(w < numWords) {
      etaK(w) = etaK(w)*expELogBetaK(w) + beta0K(w)
      w += 1
    }
    MathFunctions.eDirExp(etaK, expELogBetaK)
  }
  
  private def updateEtaK(beta0K: DenseVector[Double], suffStatsK: DenseVector[Double],
      expELogBetaK: DenseVector[Double], etaK: DenseVector[Double]) = {
    etaK := beta0K :+ (suffStatsK:*expELogBetaK)
    MathFunctions.eDirExp(etaK, expELogBetaK)
  }
  
  def runVB(numIters: Int, docs: CSCMatrix[Int], beta0: Double, multicore: Boolean,
      logger: BufferedWriter) {
    val newBeta0 = DenseMatrix.fill(numTopics, docs.rows)(beta0)
    runVB(numIters, docs, newBeta0, multicore, logger)
  }
  
  def runVB(numIters: Int, docs: CSCMatrix[Int], beta0: DenseVector[Double], 
      multicore: Boolean, logger: BufferedWriter) {
    val newBeta0 = DenseMatrix.tabulate(numTopics, docs.rows)((k, w) => beta0(w))
    runVB(numIters, docs, newBeta0, multicore, logger)
  }
  
  def runVB(numIters: Int, docs: CSCMatrix[Int], beta0: DenseMatrix[Double], 
      multicore: Boolean, logger: BufferedWriter) {
    val numDocs = docs.cols
    val numWords = docs.rows
    val nnz = docs.activeSize
    val docPtrs = docs.colPtrs
    val wordIndices = DenseVector(docs.rowIndices)
    val counts = DenseVector(docs.data)
    val expELogTheta = DenseMatrix.zeros[Double](numTopics, numDocs)
    val phiNorm = DenseVector.zeros[Double](nnz)
    val expELogBeta = DenseMatrix.zeros[Double](numTopics, numWords)
    val zeros = DenseVector.zeros[Double](numWords)
    val elbos = new Array[Double](numIters)
    val times = new Array[Double](numIters)
    val topicIndices = 0 until numTopics
    if (multicore) {
      for (k <- topicIndices.par) {
        MathFunctions.eDirExp(eta(k, ::).t, expELogBeta(k, ::).t)
      }
    } else {
      for (k <- topicIndices) {
        MathFunctions.eDirExp(eta(k, ::).t, expELogBeta(k, ::).t)
      }
    }
    val docIndices = (0 until numDocs).filter(d => (docPtrs(d+1) - docPtrs(d)) > 0)
    for (iter <- 0 until numIters) {
      val startTime = System.currentTimeMillis()
      if (multicore) {
        for (d <- docIndices.par) {
          val range = docPtrs(d) until docPtrs(d+1)
          updateGammaD(1, wordIndices(range), counts(range), alpha, expELogBeta, 
              phiNorm(range), expELogTheta(::, d), gamma(::, d))
        }
        for (k <- topicIndices.par) {
          updateEtaK(docs, beta0(k, ::).t, phiNorm, expELogTheta(k, ::).t, 
              expELogBeta(k, ::).t, eta(k, ::).t)
        }
      } else {
        for (d <- docIndices) {
          val range = docPtrs(d) until docPtrs(d+1)
          updateGammaD(1, wordIndices(range), counts(range), alpha, expELogBeta, 
              phiNorm(range), expELogTheta(::, d), gamma(::, d))
        }
        for (k <- topicIndices) {
          updateEtaK(docs, beta0(k, ::).t, phiNorm, expELogTheta(k, ::).t, 
              expELogBeta(k, ::).t, eta(k, ::).t)
        }
      }
      val elapsed = (System.currentTimeMillis - startTime)*0.001
      val elbo = approx_bound(docs, alpha, beta0, multicore)
      println("iter: " + iter + " done, time elapsed: " + elapsed + ", elbo: " + elbo)
      elbos(iter) = elbo
      times(iter) = elapsed
    }
    logger.write(elbos.mkString("[", ", ", "];") + '\n')
    logger.write(times.mkString("[", ", ", "];") + '\n')
  }
  
  private def getScore(gammaD: DenseVector[Double], eLogThetaD: DenseVector[Double],
      alpha: DenseVector[Double]): Double = {
    sum(((alpha - gammaD):*eLogThetaD) + lgamma(gammaD)) - lgamma(sum(gammaD))
  }
  
  private def getTokensScore(range: Range, eLogThetaD: DenseVector[Double], 
    eLogBeta: DenseMatrix[Double], wordIndices: Array[Int], counts: Array[Int])
    : Double = {
    range.map{ p =>
      val logPhiDW = eLogThetaD + eLogBeta(::, wordIndices(p))
      val maxLogPhiDw = max(logPhiDW)
      (log(sum(exp(logPhiDW - maxLogPhiDw))) + maxLogPhiDw)*counts(p)
    }.sum
  }
  
  def approx_bound(docs: CSCMatrix[Int], alpha: DenseVector[Double], 
      beta0: DenseMatrix[Double], multicore: Boolean): Double = {
    val numDocs = docs.cols
    val numWords = docs.rows
    val docPtrs = docs.colPtrs
    val nnz = docs.activeSize
    val wordIndices = docs.rowIndices
    val counts = docs.data
    val eLogThetaD = DenseVector.zeros[Double](numTopics)
    val eLogBeta = DenseMatrix.zeros[Double](numTopics, numWords)
    val docIndices = (0 until numDocs).filter(d => (docPtrs(d+1) - docPtrs(d)) > 0)
    val topicIndices = 0 until numTopics
    var score = 0.0
    if (multicore) {
      val topicScore = DenseVector.zeros[Double](numTopics)
      for (k <- topicIndices.par) {
        MathFunctions.dirExp(eta(k, ::).t, eLogBeta(k, ::).t)
        topicScore(k) = getScore(eta(k, ::).t, eLogBeta(k, ::).t, beta0(k, ::).t)
          + lgamma(sum(beta0(k, ::).t)) - sum(lgamma(beta0(k, ::).t))
      }
      val docScore = DenseVector.zeros[Double](numDocs)
      for (d <- docIndices.par) {
        MathFunctions.dirExp(gamma(::, d), eLogThetaD)
        val range = docPtrs(d) until docPtrs(d+1)
        docScore(d) = getTokensScore(range, eLogThetaD, eLogBeta, wordIndices, counts) 
          + getScore(gamma(::, d), eLogThetaD, alpha)
      }
      score += sum(topicScore) + sum(docScore)
    } else {
      for (k <- topicIndices) {
        MathFunctions.dirExp(eta(k, ::).t, eLogBeta(k, ::).t)
        score += getScore(eta(k, ::).t, eLogBeta(k, ::).t, beta0(k, ::).t) + 
          lgamma(sum(beta0(k, ::).t)) - sum(lgamma(beta0(k, ::).t))
      }
      for (d <- docIndices) {
        MathFunctions.dirExp(gamma(::, d), eLogThetaD)
        val range = docPtrs(d) until docPtrs(d+1)
        score += getTokensScore(range, eLogThetaD, eLogBeta, wordIndices, counts) + 
          getScore(gamma(::, d), eLogThetaD, alpha)
      }
    }
    score += numDocs*(lgamma(sum(alpha)) - sum(lgamma(alpha)))
    score
  }
  
  def printTopics(numWordsPerTopic: Int, dictPath: String, logger: BufferedWriter) = {
    val dict = Source.fromFile(dictPath).getLines.toArray
    for (k <- 0 until numTopics) {
      val pairs = (eta(k, ::).t:/sum(eta(k, ::).t))
        .toArray.zipWithIndex.sortWith((a,b) => a._1 > b._1)
//      println(f"$name%s is $height%2.2f meters tall")  // James is 1.90 meters tall
      logger.write(f"topic $k%d:\n")
      for (i <- 0 until numWordsPerTopic) {
        val word = dict(pairs(i)._2)
        val prob = pairs(i)._1
        logger.write(f"$word%20s \t---\t $prob%.4f\n")
      }
      logger.newLine
    }
  }
}

object LDA {
  
  def apply(numTopics: Int, numDocs: Int, numWords: Int, alphaInit: Double) = {
    val gd = new GammaDistribution(100, 1./100)
    gd.reseedRandomGenerator(1234567890L)
    val gamma = DenseMatrix.fill(numTopics, numDocs)(gd.sample)
    val eta = DenseMatrix.fill(numTopics, numWords)(gd.sample)
    val alpha = DenseVector.fill(numTopics)(alphaInit)
    new LDA(numTopics, gamma, eta, alpha)
  }
  
  def toCSCMatrix(inputDocsPath: String): CSCMatrix[Int] = {
    val lines = Source.fromFile(inputDocsPath).getLines
    val numDocs = lines.next.toInt
    val numWords = lines.next.toInt
    val nnz = lines.next.toInt
    val docsBuilder = new CSCMatrix.Builder[Int](rows=numWords, cols=numDocs)
    for (line <- lines) {
      val tokens = line.split(" ")
      docsBuilder.add(tokens(1).toInt - 1, tokens(0).toInt - 1, tokens(2).toInt)
    }
    docsBuilder.result
  }
  
  def main(args : Array[String]) {
    
    val prefix = "/Users/xianxingzhang/Documents/workspace/datasets/Bags_Of_Words/"
    val inputDocsPath = prefix + "docword.nips.txt"
    val dictPath = prefix + "vocab.nips.txt"
    val outputDir = "output/LDA_Local/NIPS/"
    val numTopics = 20
    val numIters = 20
    val alphaInit = 50.0/numTopics
    val betaInit = 0.001
    val multicore = true
    val docs = toCSCMatrix(inputDocsPath)
    val numWords = docs.rows
    val numDocs = docs.cols
    val startTime = System.currentTimeMillis
    val lda = LDA(numTopics, numDocs, numWords, alphaInit)
    val outputPrefix = outputDir + "T_" + numTopics + "_ITER_" + numIters
    val elboFileName = outputPrefix + "_ELBO.txt"
    val elboLogger = new BufferedWriter(new FileWriter(new File(elboFileName)))
    lda.runVB(numIters, docs, betaInit, multicore, elboLogger)
    elboLogger.close
    val topicsFileName = outputPrefix + "_Topics.txt"
    val topicsLogger = new BufferedWriter(new FileWriter(new File(topicsFileName)))
    lda.printTopics(10, dictPath, topicsLogger)
    topicsLogger.close
    println("total time elapsed: " + (System.currentTimeMillis - startTime)*0.001)
    System.exit(0)
  }
}