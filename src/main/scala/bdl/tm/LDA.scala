package tm

import java.io._

import scala.io.Source
import scala.util.Sorting

import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean
import org.apache.commons.math3.distribution._

import utilities.MathFunctions

class LDA(val eta: DenseMatrix[Double], val alpha: DenseVector[Double]) {
    
  val numTopics = eta.rows
  val numWords = eta.cols
  
  private def updateEtaK(beta0K: DenseVector[Double], suffStatsK: DenseVector[Double],
      expELogBetaK: DenseVector[Double], etaK: DenseVector[Double]) = {
    etaK := beta0K :+ (suffStatsK:*expELogBetaK)
  }
  
  private def updateAlphaBlei(numIter: Int, numDocs: Int, 
      suffStats: DenseVector[Double]) {
    //only can be used when alpha is a scalar
    val df = DenseVector.ones[Double](numTopics)
    val d2f = DenseVector.ones[Double](numTopics)
    val logAlpha = log(alpha)
    val D = numDocs.toDouble
    for (iter <- 0 until numIter if (mean(abs(df)) > 1e-3)) {
      df := MathFunctions.dirExp(alpha)
      df :*= -D 
      df :+= suffStats
      d2f := (-trigamma(alpha)+trigamma(sum(alpha))):*D
      logAlpha :-= df:/(d2f :* alpha + df)
      alpha := exp(logAlpha)
      val f = D*(lgamma(sum(alpha)) - sum(lgamma(alpha))) + sum((alpha-1.0):*suffStats)
      println("updating alpha iter: " + iter + "\t f: " + f + " \t df: " + df)
    }
  }
  
  private def updateAlpha(numIter: Int, numDocs: Int, suffStats: DenseVector[Double]) {
    val df = DenseVector.ones[Double](numTopics)
    val q = DenseVector.ones[Double](numTopics)
    val ones = DenseVector.ones[Double](numTopics)
    val D = numDocs.toDouble
    println("alpha suff stats: " + exp(suffStats:/D))
    for (iter <- 0 until numIter if (mean(abs(df)) > 1e-3)) {
      df := MathFunctions.dirExp(alpha)
      df :*= -D 
      df :+= suffStats
      q := -trigamma(alpha):*D
      val z = trigamma(sum(alpha))*D
      val b = sum(df:/q)/(1/z + sum(ones:/q))
      alpha :-= (df-b):/q
      alpha := alpha.map(a => if (a <= 0) 1e-5 else a)
//    val f = D*(lgamma(sum(alpha)) - sum(lgamma(alpha))) + sum((alpha-1.0):*suffStats)
//      println("updating alpha iter: " + iter + "\t f: " + f + " \t df: " + df)
    }
  }
  
  def runVB(outerIters: Int, innerIters: Int, docs: Array[Document], 
      beta0: Double, multicore: Boolean, emBayes: Boolean, logger: BufferedWriter) {
    val newBeta0 = DenseMatrix.fill(numTopics, numWords)(beta0)
    runVB(outerIters, innerIters, docs, newBeta0, multicore, emBayes, logger)
  }
  
  def runVB(outerIters: Int, innerIters: Int, docs: Array[Document], 
      beta0: DenseMatrix[Double], multicore: Boolean, emBayes: Boolean, 
      logger: BufferedWriter) {
    
    val numDocs = docs.length
    val expELogBeta = DenseMatrix.zeros[Double](numTopics, numWords)
    val zeros = DenseVector.zeros[Double](numWords)
    val alphaSuffStats = DenseVector.zeros[Double](numTopics)
    val phiNorm = docs.map(doc => DenseVector.zeros[Double](doc.length))
    val elbos = new Array[Double](outerIters)
    val times = new Array[Double](outerIters)
    val topicIndices = 0 until numTopics
    val topicIndicesPar = topicIndices.par
    val docIndices = (0 until numDocs).filter(d => docs(d).length > 0)
    val docIndicesPar = docIndices.par
    for (iter <- 0 until outerIters) {
      val startTime = System.currentTimeMillis()
      if (multicore) {
        for (k <- topicIndicesPar) {
          expELogBeta(k, ::).t := MathFunctions.eDirExp(eta(k, ::).t)
          eta(k, ::).t := zeros
        }
        for (d <- docIndicesPar) {
          docs(d).updateGamma(innerIters, alpha, expELogBeta, phiNorm(d), eta)
        }
        if (emBayes) {
          alphaSuffStats := 
            docIndicesPar.map(d => MathFunctions.dirExp(docs(d).gamma)).reduce(_+=_)
        }
        for (k <- topicIndicesPar) {
          updateEtaK(beta0(k, ::).t, eta(k, ::).t, expELogBeta(k, ::).t, eta(k, ::).t)
        }
      } else {
        for (k <- topicIndices) {
          expELogBeta(k, ::).t := MathFunctions.eDirExp(eta(k, ::).t)
          eta(k, ::).t := zeros
        }
        for (d <- docIndices) {
          docs(d).updateGamma(innerIters, alpha, expELogBeta, phiNorm(d), eta)
        }
        if (emBayes) {
          alphaSuffStats := 
            docIndicesPar.map(d => MathFunctions.dirExp(docs(d).gamma)).reduce(_+=_)
        }
        for (k <- topicIndices) {
          updateEtaK(beta0(k, ::).t, eta(k, ::).t, expELogBeta(k, ::).t, eta(k, ::).t)
        }
      }
      if (emBayes) {
        updateAlpha(100, numDocs, alphaSuffStats)
        println("alpha: " + alpha)
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
  
  def approx_bound(docs: Array[Document], alpha: DenseVector[Double], 
      beta0: DenseMatrix[Double], multicore: Boolean): Double = {
    val numDocs = docs.length
    val eLogThetaD = DenseVector.zeros[Double](numTopics)
    val eLogBeta = DenseMatrix.zeros[Double](numTopics, numWords)
    val topicIndices = 0 until numTopics
    val topicIndicesPar = topicIndices.par
    val docIndices = (0 until numDocs).filter(d => docs(d).length > 0)
    val docIndicesPar = docIndices.par
    var score = 0.0
    if (multicore) {
      val topicScore = DenseVector.zeros[Double](numTopics)
      for (k <- topicIndicesPar) {
        eLogBeta(k, ::).t := MathFunctions.dirExp(eta(k, ::).t)
        topicScore(k) = getScore(eta(k, ::).t, eLogBeta(k, ::).t, beta0(k, ::).t)
          + lgamma(sum(beta0(k, ::).t)) - sum(lgamma(beta0(k, ::).t))
      }
      val docScore = DenseVector.zeros[Double](numDocs)
      for (d <- docIndicesPar) {
        eLogThetaD := MathFunctions.dirExp(docs(d).gamma)
        docScore(d) = docs(d).getScore(eLogThetaD, eLogBeta) + 
          getScore(docs(d).gamma, eLogThetaD, alpha)
      }
      score += sum(topicScore) + sum(docScore)
    } else {
      for (k <- topicIndices) {
        eLogBeta(k, ::).t := MathFunctions.dirExp(eta(k, ::).t)
        score += getScore(eta(k, ::).t, eLogBeta(k, ::).t, beta0(k, ::).t) + 
          lgamma(sum(beta0(k, ::).t)) - sum(lgamma(beta0(k, ::).t))
      }
      for (d <- docIndices) {
        eLogThetaD := MathFunctions.dirExp(docs(d).gamma)
        score += docs(d).getScore(eLogThetaD, eLogBeta)  + 
          getScore(docs(d).gamma, eLogThetaD, alpha)
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
  
  def apply(numTopics: Int, numWords: Int, alphaInit: Double) = {
    val gd = new GammaDistribution(100, 1./100)
    gd.reseedRandomGenerator(1234567890L)
    val eta = DenseMatrix.fill(numTopics, numWords)(gd.sample)
    val alpha = DenseVector.fill(numTopics)(alphaInit)
    new LDA(eta, alpha)
  }
  
  def main(args : Array[String]) {
    
    val prefix = "/Users/xianxingzhang/Documents/workspace/datasets/Bags_Of_Words/"
    val inputDocsPath = prefix + "nips_processed"
    val dictPath = prefix + "vocab.nips.txt"
    val outputDir = "output/LDA_Local/NIPS/"
    val numTopics = 20
    val numWords = 12419
    val outerIters = 20
    val innerIters = 5
    val alphaInit = 50.0/numTopics
    val emBayes = false
    val betaInit = 0.001
    val multicore = false
    val startTime = System.currentTimeMillis
    val docs = preprocess.TM.toCorpus(inputDocsPath, numTopics)
    val lda = LDA(numTopics, numWords, alphaInit)
    val outputPrefix = outputDir + "T_" + numTopics + "_OI_" + outerIters + 
      "_II_" + innerIters + "_EB_" + emBayes + "_MC_" + multicore
    val elboFileName = outputPrefix + "_ELBO.txt"
    val elboLogger = new BufferedWriter(new FileWriter(new File(elboFileName)))
    lda.runVB(outerIters, innerIters, docs, betaInit, multicore, emBayes, elboLogger)
    elboLogger.close
    val topicsFileName = outputPrefix + "_Topics.txt"
    val topicsLogger = new BufferedWriter(new FileWriter(new File(topicsFileName)))
    lda.printTopics(10, dictPath, topicsLogger)
    topicsLogger.close
    println("total time elapsed: " + (System.currentTimeMillis - startTime)*0.001)
    System.exit(0)
  }
}