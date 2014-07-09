package tm

import java.io._

import scala.io.Source
import scala.util.Sorting

import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean

import org.apache.commons.math3.distribution._

import utilities.MathFunctions

class LDA(val eta: DenseMatrix[Double], val alpha: DenseVector[Double]) 
  extends Serializable {
  
  val numTopics = eta.rows
  val numWords = eta.cols
  
  private def updateEtaK(beta0K: DenseVector[Double], suffStatsK: DenseVector[Double],
      expELogBetaK: DenseVector[Double], etaK: DenseVector[Double]) = {
//    etaK := beta0K :+ (suffStatsK:*expELogBetaK)
    val length = etaK.length
    var l = 0
    while (l < length) {
      val ss = suffStatsK(l)*expELogBetaK(l)
      val beta0Kl = beta0K(l)
      val update = beta0K(l) + suffStatsK(l)*expELogBetaK(l)
      etaK(l) = if (update > 0) update else 1e-5
      l += 1
    }
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
  
  def runVB(outerIters: Int, innerIters: Int, trainingDocs: Array[Document],
      beta0: Double, multicore: Boolean, updateAlpha: Boolean): Unit = {
    val newBeta0 = DenseMatrix.fill(numTopics, numWords)(beta0)
    runVB(outerIters, innerIters, trainingDocs, newBeta0, multicore, updateAlpha)
  }
  
  def runVB(outerIters: Int, innerIters: Int, trainingDocs: Array[Document],
      beta0: DenseMatrix[Double], multicore: Boolean, updateAlpha: Boolean): Unit = {
    
    val numDocs = trainingDocs.length
    val expELogBeta = DenseMatrix.zeros[Double](numTopics, numWords)
    val zeros = DenseVector.zeros[Double](numWords)
    val alphaSuffStats = DenseVector.zeros[Double](numTopics)
    val maxLength = trainingDocs.map(_.length).max
    val phiNorm = trainingDocs.map(doc => DenseVector.zeros[Double](doc.length))
    val expELogTheta = DenseMatrix.zeros[Double](numTopics, numDocs)
    val topicIndices = 0 until numTopics
    val topicIndicesPar = topicIndices.par
    val docIndices = (0 until numDocs).filter(d => trainingDocs(d).length > 0)
    val docIndicesPar = docIndices.par
    for (iter <- 0 until outerIters) {
      if (multicore) {
        for (k <- topicIndicesPar) {
          expELogBeta(k, ::).t := MathFunctions.eDirExp(eta(k, ::).t)
          eta(k, ::).t := zeros
          alphaSuffStats(k) = 0
        }
        for (d <- docIndicesPar) {
          trainingDocs(d).updateGamma(innerIters, alpha, expELogBeta, 
              expELogTheta(::, d), phiNorm(d))
          trainingDocs(d).getSuffStats(expELogTheta(::, d), phiNorm(d), eta)
          if (updateAlpha) alphaSuffStats :+= log(expELogTheta(::, d))
        }
        for (k <- topicIndicesPar) {
          updateEtaK(beta0(k, ::).t, eta(k, ::).t, expELogBeta(k, ::).t, eta(k, ::).t)
        }
      } else {
        for (k <- topicIndices) {
          expELogBeta(k, ::).t := MathFunctions.eDirExp(eta(k, ::).t)
          eta(k, ::).t := zeros
          alphaSuffStats(k) = 0
        }
        for (d <- docIndices) {
          trainingDocs(d).updateGamma(innerIters, alpha, expELogBeta, 
              expELogTheta(::, d), phiNorm(d))
          trainingDocs(d).getSuffStats(expELogTheta(::, d), phiNorm(d), eta)
          if (updateAlpha) alphaSuffStats :+= log(expELogTheta(::, d))
        }
        for (k <- topicIndices) {
          updateEtaK(beta0(k, ::).t, eta(k, ::).t, expELogBeta(k, ::).t, eta(k, ::).t)
        }
      }
      if (updateAlpha) {
        LDA.updateAlpha(100, numDocs, alphaSuffStats, alpha)
//        println("alpha: " + alpha)
      }
    }
  }
  
  
  //much more time consuming, but also more robust in float overflow
  def runVB2(outerIters: Int, innerIters: Int, trainingDocs: Array[Document],
      beta0: DenseMatrix[Double], multicore: Boolean, updateAlpha: Boolean): Unit = {
    
    val numDocs = trainingDocs.length
    val eLogBeta = DenseMatrix.zeros[Double](numTopics, numWords)
    val alphaSuffStats = DenseVector.zeros[Double](numTopics)
    val maxLength = trainingDocs.map(_.length).max
    val logPhiNorm = trainingDocs.map(doc => DenseVector.zeros[Double](doc.length))
    val eLogTheta = DenseMatrix.zeros[Double](numTopics, numDocs)
    val topicIndices = 0 until numTopics
    val topicIndicesPar = topicIndices.par
    val docIndices = (0 until numDocs).filter(d => trainingDocs(d).length > 0)
    val docIndicesPar = docIndices.par
    for (iter <- 0 until outerIters) {
      if (multicore) {
        for (k <- topicIndicesPar) {
          eLogBeta(k, ::).t := MathFunctions.dirExp(eta(k, ::).t)
          eta(k, ::) := beta0(k, ::)
          alphaSuffStats(k) = 0
        }
        for (d <- docIndicesPar) {
          trainingDocs(d).updateGamma2(innerIters, alpha, eLogBeta, 
              eLogTheta(::, d), logPhiNorm(d))
          trainingDocs(d).updateEta(eLogTheta(::, d), logPhiNorm(d), eLogBeta, eta)
          if (updateAlpha) alphaSuffStats :+= eLogTheta(::, d)
        }
      } else {
        for (k <- topicIndices) {
          eLogBeta(k, ::).t := MathFunctions.dirExp(eta(k, ::).t)
          eta(k, ::) := beta0(k, ::)
          alphaSuffStats(k) = 0
        }
        for (d <- docIndices) {
          trainingDocs(d).updateGamma2(innerIters, alpha, eLogBeta, 
              eLogTheta(::, d), logPhiNorm(d))
          trainingDocs(d).updateEta(eLogTheta(::, d), logPhiNorm(d), eLogBeta, eta)
          if (updateAlpha) alphaSuffStats :+= eLogTheta(::, d)
        }
      }
      if (updateAlpha) {
        LDA.updateAlpha(100, numDocs, alphaSuffStats, alpha)
        println("alpha: " + alpha)
      }
    }
  }
  
  def approxBound(docs: Array[Document], alpha: DenseVector[Double], 
      beta0: DenseMatrix[Double], multicore: Boolean): Double = {
    LDA.approxBound(eta, docs, alpha, beta0, multicore)
  }
}

object LDA {
  
  def apply(numTopics: Int, numWords: Int, alphaInit: Double, seed: Int) = {
    val gd = new GammaDistribution(10, 1.0/10)
    gd.reseedRandomGenerator(seed)
    val eta = DenseMatrix.fill(numTopics, numWords)(gd.sample)
    val alpha = DenseVector.fill(numTopics)(alphaInit)
    new LDA(eta, alpha)
  }
  
  def getScore(gammaD: DenseVector[Double], eLogThetaD: DenseVector[Double],
      alpha: DenseVector[Double]): Double = {
    sum(((alpha - gammaD):*eLogThetaD) + lgamma(gammaD)) - lgamma(sum(gammaD))
  }
  
  def getScore(gammaD: DenseVector[Double], eLogThetaD: DenseVector[Double],
      alpha: Double): Double = {
    - sum(((gammaD - alpha):*eLogThetaD) - lgamma(gammaD)) - lgamma(sum(gammaD))
  }
  
  def getLLH(foldingin: Int, eta: DenseMatrix[Double], docs: Array[Document],
      alpha: DenseVector[Double], multicore: Boolean): Double = {
    val numTopics = eta.rows
    val numWords = eta.cols
    val expELogBeta = 
      if (foldingin > 0) MathFunctions.eDirExp(eta, multicore)
      else null
    val eBeta = MathFunctions.normalize(eta, multicore)
    if (multicore) {
      docs.par.map{ doc =>
        if (foldingin > 0) doc.updateGamma(foldingin, alpha, expELogBeta)
        doc.getLLH(eBeta)
      }.sum
    } else {
      val expELogTheta = DenseVector.zeros[Double](numTopics)
      val maxLength = docs.map(_.length).max
      val phiNorm = DenseVector.zeros[Double](maxLength)
      docs.map{ doc =>
        if (foldingin > 0) {
          doc.updateGamma(foldingin, alpha, expELogBeta, expELogTheta, phiNorm)
        }
        doc.getLLH(eBeta)
      }.sum
    }
  }
  
  def getPerlexity(foldingin: Int, eta: DenseMatrix[Double], docs: Array[Document],
      alpha: DenseVector[Double], multicore: Boolean): Double = {
    math.exp(-getLLH(foldingin, eta, docs, alpha, multicore) / docs.map(_.nnz).sum)
  }
  
  def approxBound(eta: DenseMatrix[Double], docs: Array[Document],
      alpha: DenseVector[Double], beta0: DenseMatrix[Double], multicore: Boolean)
    : Double = {
    val numDocs = docs.length
    val numTopics = eta.rows
    val numWords = eta.cols
    val eLogBeta = DenseMatrix.zeros[Double](numTopics, numWords)
    val topicIndices = 0 until numTopics
    val topicIndicesPar = topicIndices.par
    val docIndices = (0 until numDocs).filter(d => docs(d).length > 0)
    val docIndicesPar = docIndices.par
    var score = 0.0
    if (multicore) {
      score += topicIndicesPar.map{ k =>
        eLogBeta(k, ::).t := MathFunctions.dirExp(eta(k, ::).t)
        getScore(eta(k, ::).t, eLogBeta(k, ::).t, beta0(k, ::).t) +
          lgamma(sum(beta0(k, ::).t)) - sum(lgamma(beta0(k, ::).t))
      }.reduce(_+_)
      score += docIndicesPar.map{ d =>
        val eLogThetaD = MathFunctions.dirExp(docs(d).gamma)
        docs(d).getScore(eLogThetaD, eLogBeta) + 
          getScore(docs(d).gamma, eLogThetaD, alpha)
      }.reduce(_+_)
    } else {
      for (k <- topicIndices) {
        eLogBeta(k, ::).t := MathFunctions.dirExp(eta(k, ::).t)
        score += getScore(eta(k, ::).t, eLogBeta(k, ::).t, beta0(k, ::).t) + 
          lgamma(sum(beta0(k, ::).t)) - sum(lgamma(beta0(k, ::).t))
      }
      val eLogThetaD = DenseVector.zeros[Double](numTopics)
      for (d <- docIndices) {
        eLogThetaD := MathFunctions.dirExp(docs(d).gamma)
        score += docs(d).getScore(eLogThetaD, eLogBeta)  + 
          getScore(docs(d).gamma, eLogThetaD, alpha)
      }
    }
    score += numDocs*(lgamma(sum(alpha)) - sum(lgamma(alpha)))
    score
  }
  
  def updateAlpha(numIter: Int, numDocs: Int, suffStats: DenseVector[Double],
      alpha: DenseVector[Double]) {
    val numTopics = alpha.length
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
  
  def printTopics(topics: DenseMatrix[Double], dict: Array[String],
      numWordsPerTopic: Int, logger: BufferedWriter): Unit = {
    val numTopics = topics.rows
    for (k <- 0 until numTopics) {
      val pairs = (topics(k, ::).t:/sum(topics(k, ::).t))
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
  
  def main(args : Array[String]) {
        
    val prefix = "/Users/xianxingzhang/Documents/workspace/datasets/Bags_Of_Words/"
    
    val trainingDir = prefix + "nips_processed/train"
    val validatingDir = prefix + "nips_processed/validate"
    val dictPath = prefix + "vocab.nips.txt"
    val outputDir = "output/LDA_Local/NIPS/"
//    val trainingDir = prefix + "kos_processed/train"
//    val validatingDir = prefix + "kos_processed/validate"
//    val dictPath = prefix + "vocab.kos.txt"
//    val outputDir = "output/LDA_Local/KOS/"
      
    val dict = Source.fromFile(dictPath).getLines.toArray
    val numTopics = 20
    val numWords = dict.length
    val outerIters = 10
    val innerIters = 20
    val alphaInit = 1
    val updateAlpha = false
    val betaInit = 0.01
    val multicore = false
    val startTime = System.currentTimeMillis
    val gammaSeed = 987654321
    val trainingDocs = preprocess.TM.toCorpus(trainingDir, numTopics, gammaSeed)
    val validatingDocs = preprocess.TM.toCorpus(validatingDir, numTopics, gammaSeed)
    val etaSeed = 123456789
    val lda = LDA(numTopics, numWords, alphaInit, etaSeed)
    val outputPrefix = outputDir + "T_" + numTopics + "_OI_" + outerIters + 
      "_II_" + innerIters + "_EB_" + updateAlpha + "_MC_" + multicore + "_alpha_" + 
      alphaInit + "_beta0_" + betaInit
    val perpFileName = outputPrefix + "_Result.txt"
    val trPerps = new Array[Double](outerIters)
    val valPerps = new Array[Double](outerIters)
    val times = new Array[Double](outerIters)
    val beta0 = DenseMatrix.fill(numTopics, numWords)(betaInit)
    for (iter <- 0 until outerIters) {
      val startTime = System.currentTimeMillis()
      lda.runVB(innerIters, 5, trainingDocs, beta0, multicore, updateAlpha)
      val eta = lda.eta
      val alpha = lda.alpha
      times(iter) = (System.currentTimeMillis - startTime)*0.001
      trPerps(iter) = LDA.getPerlexity(0, eta, trainingDocs, alpha, multicore)
      valPerps(iter) = LDA.getPerlexity(50, eta, validatingDocs, alpha, multicore)
      
      println("iter: " + iter + " done, time elapsed: " + times(iter) + 
        ", training perlexity: " + trPerps(iter) + 
        ", validating perplexity: " + valPerps(iter))
    }
    
    val perpLogger = new BufferedWriter(new FileWriter(new File(perpFileName)))
    perpLogger.write(times.mkString("[", ", ", "];") + '\n')
    perpLogger.write(trPerps.mkString("[", ", ", "];") + '\n')
    perpLogger.write(valPerps.mkString("[", ", ", "];") + '\n')
    perpLogger.close
    val topicsFileName = outputPrefix + "_Topics.txt"
    val topicsLogger = new BufferedWriter(new FileWriter(new File(topicsFileName)))
    LDA.printTopics(lda.eta, dict, 10, topicsLogger)
    topicsLogger.close
    println("Total time elapsed: " + (System.currentTimeMillis - startTime)*0.001)
    System.exit(0)
  }
}