package mf2

import java.util.Random

import org.jblas.DoubleMatrix

import utilities.SparseMatrix
import utilities.Vector
import OptimizerType._
import CoordinateDescent._

class LocalModel (val factorsR: Array[Array[Float]], val factorsC: Array[Array[Float]],
    val precsR: Array[Array[Float]], val precsC: Array[Array[Float]],
    val lagsR: Array[Array[Float]], val lagsC: Array[Array[Float]],
    val gammaR: Array[Float], val gammaC: Array[Float], var numIter: Int) 
    extends Serializable {
    
  def getSE(data: SparseMatrix): Double = data.getSE(factorsR, factorsC, true)
  
  def getRMSE(data: SparseMatrix): Double = {
    math.sqrt(data.getSE(factorsR, factorsC, true)/data.col_idx.length)
  }
  
  def getStatsR(map: Array[Int], multicore: Boolean)
    : Array[(Int, (Vector, Vector))] = {
    LocalModel.getStats(factorsR, lagsR, gammaR, map, multicore)
  }
  
  def getStatsR(map: Array[Int], ptr: Array[Int], multicore: Boolean)
    : Array[(Int, (Vector, Vector))] = {
    LocalModel.getStats(factorsR, lagsR, gammaR, map, ptr, multicore)
  }
  
  def getStatsC(map: Array[Int], multicore: Boolean)
    : Array[(Int, (Vector, Vector))] = {
    LocalModel.getStats(factorsC, lagsC, gammaC, map, multicore)
  }
  
  def getStatsC(map: Array[Int], ptr: Array[Int], multicore: Boolean)
    : Array[(Int, (Vector, Vector))] = {
    LocalModel.getStats(factorsC, lagsC, gammaC, map, ptr, multicore)
  }
  
  def updateLagR(priorsR: Array[Array[Float]], multicore: Boolean): LocalModel = {
    LocalModel.updateLag(factorsR, lagsR, priorsR, multicore)
    this
  }
  
  def updateLagC(priorsC: Array[Array[Float]], multicore: Boolean): LocalModel = {
    LocalModel.updateLag(factorsC, lagsC, priorsC, multicore)
    this
  }
  
  def train(data: SparseMatrix, optType: OptimizerType, maxIter: Int, stopCrt: Float,
      isVB: Boolean, emBayes: Boolean, weightedReg: Boolean, multicore: Boolean, 
      priorsR: Array[Array[Float]], priorsC: Array[Array[Float]])
    : LocalModel = {
    numIter = optType match {
      case CD => runCD(data, maxIter, stopCrt, isVB, weightedReg, multicore, 
        priorsR, factorsR, precsR, gammaR, priorsC, factorsC, precsC, gammaC)
      case CDPP => runCDPP(data, maxIter, 1, stopCrt, isVB, weightedReg, multicore,
        priorsR, factorsR, precsR, gammaR, priorsC, factorsC, precsC, gammaC)
      case _ => {System.err.println("only supports CD and CDPP"); 0}
    }
    if (emBayes) {
      if (weightedReg) updateGamma(factorsR, precsR, priorsR, data.row_ptr, gammaR)
      else updateGamma(factorsR, precsR, priorsR, gammaR)
      if (weightedReg) updateGamma(factorsC, precsC, priorsC, data.col_ptr, gammaC)
      else updateGamma(factorsC, precsC, priorsC, gammaC)
    }
    this
  }
}

object LocalModel {
  
  def apply (numFactors: Int, rowMap: Array[Int], colMap: Array[Int], 
      rowPtr: Array[Int], colPtr: Array[Int], gamma_r_init: Float, gamma_c_init: Float,
      ecR: Boolean, ecC: Boolean, isVB: Boolean, weightedReg: Boolean)
    : LocalModel = {
    
    def hash(x: Int): Int = {
      val r = x ^ (x >>> 20) ^ (x >>> 12)
      r ^ (r >>> 7) ^ (r >>> 4)
    }
    
    val numRows = rowMap.length
    val numCols = colMap.length
    val factorsR = Array.ofDim[Float](numFactors, numRows)
    var r = 0
    val rand = new Random()
    while (r < numRows) {
      var k = 0
      rand.setSeed(rowMap(r))
      while (k < numFactors) {
        factorsR(k)(r) = 0.1f*(rand.nextFloat-0.5f)
        k += 1
      }
      r += 1
    }
    val factorsC = Array.ofDim[Float](numFactors, numCols)
    val gammaR = Array.fill(numFactors)(gamma_r_init)
    val gammaC = Array.fill(numFactors)(gamma_c_init)
    val precsR = 
      if (isVB) {
        if (weightedReg) {
          val numObs = getNumObs(rowPtr)
          Array.tabulate(numFactors, numRows)((k, r) => gammaR(k)*numObs(r))
        }
        else Array.tabulate(numFactors, numRows)((k, r) => gammaR(k))
      }
      else null
    val precsC = 
      if (isVB) {
        if (weightedReg) {
          val numObs = getNumObs(colPtr)
          Array.tabulate(numFactors, numCols)((k, c) => gammaC(k)*numObs(c))
        }
        else Array.tabulate(numFactors, numCols)((k, c) => gammaC(k))
      }
      else null
    val lagsR = 
      if (ecR) Array.ofDim[Float](numFactors, numRows)
      else null
    val lagsC = 
      if (ecC) Array.ofDim[Float](numFactors, numCols)
      else null
    new LocalModel(factorsR, factorsC, precsR, precsC, lagsR, lagsC, gammaR, gammaC, 0)
  }
  
  def toGlobal(factors: Array[Array[Float]], map: Array[Int]) 
    : Array[(Int, Array[Float])] = {
    val length = factors(0).length
    val results = Array.ofDim[(Int, Array[Float])](length);
    var i = 0
    while (i < length) {
      results(i) = (map(i), factors.map(array => array(i)))
      i += 1
    }
    results
//    transformed.sortBy(pair => pair._1)
  }
  
  def binarySearch(arr: Array[Int], start: Int, end: Int, target: Int) : Int = {
    val pos = 
      if (start > end) -1
      else{
        val mid = (start + end)/2
        if (arr(mid) > target) binarySearch(arr, start, mid-1, target)
        else if (arr(mid) == target) mid
        else binarySearch(arr, mid+1, end, target)
      }
    pos
  }
  
  def getStats(factors: Array[Array[Float]], lags: Array[Array[Float]],
      gamma: Array[Float], map: Array[Int], multicore: Boolean)
    : Array[(Int, (Vector, Vector))] = {
    
    val length = factors(0).length
    val numFactors = factors.length
    val isEC = lags != null
    val statsR = new Array[(Int, (Vector, Vector))](length)
    if (multicore) {
      for (r <- (0 until length).par) {
        val nume = new Array[Float](numFactors)
        val deno = new Array[Float](numFactors)
        var k = 0
        while (k < numFactors) {
          val ga = gamma(k)
          nume(k) = 
            if (isEC) (factors(k)(r)+lags(k)(r))*ga 
            else factors(k)(r)*ga
          deno(k) = ga
          k += 1
        }
        statsR(r) = (map(r), (Vector(nume), Vector(deno)))
      }
    }
    else {
      var r = 0
      while (r < length) {
        val nume = new Array[Float](numFactors)
        val deno = new Array[Float](numFactors)
        var k = 0
        while (k < numFactors) {
          val ga = gamma(k)
          nume(k) = 
            if (isEC) (factors(k)(r)+lags(k)(r))*ga 
            else factors(k)(r)*ga
          deno(k) = ga
          k += 1
        }
        statsR(r) = (map(r), (Vector(nume), Vector(deno)))
        r += 1
      }
    }
    statsR
  } // end of getStats
  
  
  def getStats(factors: Array[Array[Float]], lags: Array[Array[Float]],
      gamma: Array[Float], map: Array[Int], ptr: Array[Int], multicore: Boolean)
    : Array[(Int, (Vector, Vector))] = {
    
    val length = factors(0).length
    val numFactors = factors.length
    val isEC = lags != null
    val statsR = new Array[(Int, (Vector, Vector))](length)
    if (multicore) {
      for (r <- (0 until length).par) {
        val nume = new Array[Float](numFactors)
        val deno = new Array[Float](numFactors)
        val weight = ptr(r+1)-ptr(r)
        var k = 0
        while (k < numFactors) {
          val ga = gamma(k)*weight
          nume(k) = 
            if (isEC) (factors(k)(r)+lags(k)(r))*ga 
            else factors(k)(r)*ga
          deno(k) = ga
          k += 1
        }
        statsR(r) = (map(r), (Vector(nume), Vector(deno)))
      }
    }
    else {
      var r = 0
      while (r < length) {
        val nume = new Array[Float](numFactors)
        val deno = new Array[Float](numFactors)
        val weight = ptr(r+1)-ptr(r)
        var k = 0
        while (k < numFactors) {
          val ga = gamma(k)*weight
          nume(k) = 
            if (isEC) (factors(k)(r)+lags(k)(r))*ga 
            else factors(k)(r)*ga
          deno(k) = ga
          k += 1
        }
        statsR(r) = (map(r), (Vector(nume), Vector(deno)))
        r += 1
      }
    }
    statsR
  } // end of getStats
  
  def updateLag(factorsR: Array[Array[Float]], lagsR: Array[Array[Float]], 
    priorsR: Array[Array[Float]], multicore: Boolean) = {
    //update the scaled Lagrangian multipilers
    val numFactors = lagsR.length
    val numRows = priorsR.length
    if (multicore) {
      for (r <- (0 until numRows).par) {
        if (priorsR(r) != null) {
          var k = 0
          while (k < numFactors) {
            lagsR(k)(r) += factorsR(k)(r) - priorsR(r)(k)
            priorsR(r)(k) -= lagsR(k)(r)
            k += 1
          }
        }
      }
    }
    else {
      var r = 0
      while (r < numRows) {
        if (priorsR(r) != null) {
          var k = 0
          while (k < numFactors) {
            lagsR(k)(r) += factorsR(k)(r) - priorsR(r)(k)
            priorsR(r)(k) -= lagsR(k)(r)
            k += 1
          }
        }
        r += 1
      }
    }
  } //end of updateLag
  
  def getNumObs(ptr: Array[Int]): Array[Int] = {
    val length = ptr.length - 1
    val numObs = new Array[Int](length)
    var l = 0
    while (l < length) {
      numObs(l) = ptr(l+1) - ptr(l)
      l += 1
    }
    numObs
  }
}