package mf2

import utilities.SparseMatrix

object CoordinateDescent {
  
  def runCD(data: SparseMatrix,
      numIter: Int, multicore: Boolean,
      factorsR: Array[Array[Float]],
      precsR: Array[Array[Float]], gammaR: Array[Float], 
      priorsC: Array[Array[Float]], factorsC: Array[Array[Float]],
      precsC: Array[Array[Float]], gammaC: Array[Float]): Unit = {
    runCD(data, numIter, multicore, priorsR = null, factorsR, precsR, gammaR,
      priorsC, factorsC, precsC, gammaC)
  }
  
  def runCD(data: SparseMatrix,
      numIter: Int, multicore: Boolean,
      priorsR: Array[Array[Float]], factorsR: Array[Array[Float]], 
      precsR: Array[Array[Float]], gammaR: Array[Float], 
      factorsC: Array[Array[Float]], 
      precsC: Array[Array[Float]], gammaC: Array[Float]): Unit = {
    runCD(data, numIter, multicore, priorsR, factorsR, precsR, gammaR,
      priorsC = null, factorsC, precsC, gammaC)
  }
  
  def runCD(data: SparseMatrix,
      numIter: Int, multicore: Boolean,
      factorsR: Array[Array[Float]], 
      precsR: Array[Array[Float]], gammaR: Array[Float], 
      factorsC: Array[Array[Float]], 
      precsC: Array[Array[Float]], gammaC: Array[Float]): Unit = {
    runCD(data, numIter, multicore, priorsR = null, factorsR, precsR, gammaR,
      priorsC = null, factorsC, precsC, gammaC)
  }
  
  def runCD(data: SparseMatrix,
      numIter: Int, multicore: Boolean,
      priorsR: Array[Array[Float]], factorsR: Array[Array[Float]], 
      precsR: Array[Array[Float]], gammaR: Array[Float],
      priorsC: Array[Array[Float]], factorsC: Array[Array[Float]], 
      precsC: Array[Array[Float]], gammaC: Array[Float]): Unit = {
    
    val isVB = precsR != null && precsC != null
    val hasPriorsR = priorsR != null
    val hasPriorsC = priorsC != null
    val numFactors = factorsR.length
    val numRows = factorsR(0).length
    val numCols = factorsC(0).length
    val row_ptr = data.row_ptr
    val row_idx = data.row_idx
    val value_r = data.value_r
    val col_ptr = data.col_ptr
    val col_idx = data.col_idx
    val value_c = data.value_c
    val nnz = row_idx.length
    val res_r = new Array[Float](nnz)
    Array.copy(value_r, 0, res_r, 0, nnz)
    val res_c = new Array[Float](nnz)
    Array.copy(value_c, 0, res_c, 0, nnz)
    val rows = if (multicore) Array.tabulate(numRows)(i => i) else null
    val cols = if (multicore) Array.tabulate(numCols)(i => i) else null
    var iter = 0
    while (iter < numIter) {
      if (multicore) {
        CoordinateDescent.getResidual(col_ptr, row_idx, value_c, 
          factorsC, factorsR, true, res_c)
        //update col factors
        cols.par.map(c => {
          var k = 0
          while (k<numFactors) {
            val prior = if (hasPriorsC) priorsC(c)(k) else 0
            if (isVB) {
              update(c, col_ptr, row_idx, res_c, factorsR(k), precsR(k),
                prior, gammaC(k), factorsC(k), precsC(k))
            }
            else {
              update(c, col_ptr, row_idx, res_c, factorsR(k),
                prior, gammaC(k), factorsC(k))
            }
            k += 1
          }
        })
        CoordinateDescent.getResidual(row_ptr, col_idx, value_r, 
          factorsR, factorsC, true, res_r)
        //update row factors
        rows.par.map(r => {
          var k = 0
          while (k<numFactors) {
            val prior = if (hasPriorsR) priorsR(r)(k) else 0
            if (isVB) {
              update(r, row_ptr, col_idx, res_r, factorsC(k), precsC(k),
                prior, gammaR(k), factorsR(k), precsR(k))
            }
            else {
              update(r, row_ptr, col_idx, res_r, factorsC(k),
                prior, gammaR(k), factorsR(k))
            }
            k += 1
          }
        })
      }
      else {
        //not using multi-thread
        CoordinateDescent.getResidual(col_ptr, row_idx, value_c, 
          factorsC, factorsR, false, res_c)
        var c = 0
        while (c < numCols) {
          var k = 0
          while (k<numFactors) {
            val prior = if (hasPriorsC) priorsC(c)(k) else 0
            if (isVB) {
              update(c, col_ptr, row_idx, res_c, factorsR(k), precsR(k),
                prior, gammaC(k), factorsC(k), precsC(k))
            }
            else {
              update(c, col_ptr, row_idx, res_c, factorsR(k),
                prior, gammaC(k), factorsC(k))
            }
            k += 1
          }
          c += 1
        }
        CoordinateDescent.getResidual(row_ptr, col_idx, value_r, 
          factorsR, factorsC, false, res_r)
        var r = 0
        while (r < numRows) {
          var k = 0
          while (k<numFactors) {
            val prior = if (hasPriorsR) priorsR(r)(k) else 0
            if (isVB) {
              update(r, row_ptr, col_idx, res_r, factorsC(k), precsC(k),
                prior, gammaR(k), factorsR(k), precsR(k))
            }
            else {
              update(r, row_ptr, col_idx, res_r, factorsC(k),
                prior, gammaR(k), factorsR(k))
            }
            k += 1    
          }
          r += 1
        }
      }
      iter += 1
    }
  }
  
  def runCDPP(data: SparseMatrix,
      numOuterIter: Int, numInnerIter: Int, multicore: Boolean,
      factorsR: Array[Array[Float]], 
      precsR: Array[Array[Float]], gammaR: Array[Float],
      priorsC: Array[Array[Float]], factorsC: Array[Array[Float]], 
      precsC: Array[Array[Float]], gammaC: Array[Float]): Unit = {
    
    runCDPP(data, numOuterIter, numInnerIter, multicore, 
      priorsR = null, factorsR, precsR, gammaR, priorsC, factorsC, precsC, gammaC)
  }
  
  def runCDPP(data: SparseMatrix,
      numOuterIter: Int, numInnerIter: Int, multicore: Boolean,
      priorsR: Array[Array[Float]], factorsR: Array[Array[Float]], 
      precsR: Array[Array[Float]], gammaR: Array[Float],
      factorsC: Array[Array[Float]], 
      precsC: Array[Array[Float]], gammaC: Array[Float]): Unit = {
    
    runCDPP(data, numOuterIter, numInnerIter, multicore, 
      priorsR, factorsR, precsR, gammaR, priorsC = null, factorsC, precsC, gammaC)
  }
  
  def runCDPP(data: SparseMatrix,
      numOuterIter: Int, numInnerIter: Int, multicore: Boolean,
      factorsR: Array[Array[Float]], 
      precsR: Array[Array[Float]], gammaR: Array[Float],
      factorsC: Array[Array[Float]], 
      precsC: Array[Array[Float]], gammaC: Array[Float]): Unit = {
    
    runCDPP(data, numOuterIter, numInnerIter, multicore, priorsR = null, factorsR,
      precsR, gammaR, priorsC = null, factorsC, precsC, gammaC)
  }
  
  def runCDPP(data: SparseMatrix,
      numOuterIter: Int, numInnerIter: Int, multicore: Boolean,
      priorsR: Array[Array[Float]], factorsR: Array[Array[Float]], 
      precsR: Array[Array[Float]], gammaR: Array[Float],
      priorsC: Array[Array[Float]], factorsC: Array[Array[Float]], 
      precsC: Array[Array[Float]], gammaC: Array[Float]): Unit = {
    
    val hasPriorsR = priorsR != null
    val hasPriorsC = priorsC != null
    val isVB = precsR != null && precsC != null
    //priors are M*K and N*K respectively, while factors are K*M and K*N respectively
    val numFactors = factorsR.length
    val numRows = factorsR(0).length
    val numCols = factorsC(0).length
    val row_ptr = data.row_ptr
    val row_idx = data.row_idx
    val col_ptr = data.col_ptr
    val col_idx = data.col_idx
    //here value_c and value_r are residuals
    val res_r = data.value_r
    val res_c = data.value_c
    val rows = Array.tabulate(numRows)(i => i)
    val cols = Array.tabulate(numCols)(i => i)
    var iter = 0
    while (iter < numOuterIter) {
      var k = 0
      while (k<numFactors) {
        updateResidual(row_ptr, col_idx, res_r, factorsR(k), factorsC(k), 
          true, multicore)
        updateResidual(col_ptr, row_idx, res_c, factorsC(k), factorsR(k), 
          true, multicore)
        var i = 0
        while (i < numInnerIter) {
          if (multicore) {
            cols.par.map(c => {
              val priorC = if (hasPriorsC) priorsC(c)(k) else 0
              if (isVB) {
                updatepp(c, col_ptr, row_idx, res_c, factorsR(k), precsR(k),
                  priorC, gammaC(k), factorsC(k), precsC(k))
              }
              else { 
                updatepp(c, col_ptr, row_idx, res_c, factorsR(k), 
                  priorC, gammaC(k), factorsC(k))
              }
            })
            rows.par.map(r => {
              val priorR = if (hasPriorsR) priorsR(r)(k) else 0
              if (isVB) {
                updatepp(r, row_ptr, col_idx, res_r, factorsC(k), precsC(k),
                  priorR, gammaR(k), factorsR(k), precsR(k))
              }
              else {
                updatepp(r, row_ptr, col_idx, res_r, factorsC(k), 
                  priorR, gammaR(k), factorsR(k))
              }
            })
          }
          else {
            var c = 0
            while (c < numCols) {
              val priorC = if (hasPriorsC) priorsC(c)(k) else 0
              if (isVB) {
                updatepp(c, col_ptr, row_idx, res_c, factorsR(k), precsR(k),
                  priorC, gammaC(k), factorsC(k), precsC(k))
              }
              else { 
                updatepp(c, col_ptr, row_idx, res_c, factorsR(k), 
                  priorC, gammaC(k), factorsC(k))
              }
              c += 1
            }
            var r = 0
            while (r < numRows) {
              val priorR = if (hasPriorsR) priorsR(r)(k) else 0
              if (isVB) {
                updatepp(r, row_ptr, col_idx, res_r, factorsC(k), precsC(k),
                  priorR, gammaR(k), factorsR(k), precsR(k))
              }
              else {
                updatepp(r, row_ptr, col_idx, res_r, factorsC(k), 
                  priorR, gammaR(k), factorsR(k))
              }
              r += 1
            }
          }
          i += 1
        }
        updateResidual(row_ptr, col_idx, res_r, factorsR(k), factorsC(k), 
          false, multicore)
        updateResidual(col_ptr, row_idx, res_c, factorsC(k), factorsR(k), 
          false, multicore)
        k += 1
      }
      iter += 1
    }
//    getRMSE(res_c, multicore)
  }
  
  def update(r: Int, row_ptr: Array[Int], col_idx: Array[Int], res_r: Array[Float],
      factorsC: Array[Float], prior: Float, gammaR: Float, factorsR: Array[Float]) {
    var i = row_ptr(r); var numerator = 0f; var denominator = 0f
    val rowFactor_rk = factorsR(r)
    while (i < row_ptr(r+1)) {
      val c = col_idx(i)
      numerator += (res_r(i)+rowFactor_rk*factorsC(c))*factorsC(c)
      denominator += factorsC(c)*factorsC(c)
      i += 1
    }
    factorsR(r) = (numerator+gammaR*prior)/(denominator+gammaR)
    i = row_ptr(r)
    while (i < row_ptr(r+1)) {
      val c = col_idx(i)
      res_r(i) += (rowFactor_rk - factorsR(r))*factorsC(c) 
      i += 1
    }
  }
  
  def updatepp(r: Int, row_ptr: Array[Int], col_idx: Array[Int], res_r: Array[Float], 
      factorsC: Array[Float], prior: Float, gammaR: Float, factorsR: Array[Float]) {
    var i = row_ptr(r); var numerator = 0f; var denominator = 0f
    while (i < row_ptr(r+1)) {
      val c = col_idx(i)
      numerator += res_r(i)*factorsC(c)
      denominator += factorsC(c)*factorsC(c)
      i += 1
    }
    factorsR(r) = (numerator+gammaR*prior)/(denominator+gammaR)
  }
  
  def update(r: Int, row_ptr: Array[Int], col_idx: Array[Int], res_r: Array[Float],
      factorC: Array[Float], precC: Array[Float], prior: Float, gammaR: Float, 
      factorR: Array[Float], precR: Array[Float]) {
    var i = row_ptr(r); var numerator = 0f; var denominator = 0f; var c = 0
    val factorKR = factorR(r)
    while (i < row_ptr(r+1)) {
      c = col_idx(i)
      numerator += (res_r(i)+factorKR*factorC(c))*factorC(c)
      denominator += factorC(c)*factorC(c) + 1/precC(c)
      i += 1
    }
    precR(r) = denominator+gammaR
    factorR(r) = (numerator+gammaR*prior)/precR(r)
    i = row_ptr(r)
    while (i < row_ptr(r+1)) {
      c = col_idx(i)
      res_r(i) += (factorKR - factorR(r))*factorC(c) 
      i += 1
    }
  }
  
  def updatepp(r: Int, row_ptr: Array[Int], col_idx: Array[Int], res_r: Array[Float], 
      factorC: Array[Float], precC: Array[Float], prior: Float, gammaR: Float, 
      factorR: Array[Float], precR: Array[Float]) {
    var numerator = 0f
    var denominator = 0f
    var i = row_ptr(r)
    while (i < row_ptr(r+1)) {
      val c = col_idx(i)
      numerator += res_r(i)*factorC(c)
      denominator += factorC(c)*factorC(c) + 1/precC(c)
      i += 1
    }
    precR(r) = denominator+gammaR
    factorR(r) = (numerator+gammaR*prior)/precR(r)
  }
  
  def updateResidual(
      row_ptr: Array[Int], col_idx: Array[Int], res_r: Array[Float], 
      factorsR: Array[Float], factorsC: Array[Float], 
      add: Boolean, multicore: Boolean = false) = {
    
    val numRows = factorsR.length
    if (add) {
      if (multicore) {
        for (r <- (0 until numRows).par) {
          var i = row_ptr(r)
          while (i < row_ptr(r+1)) {
            res_r(i) = res_r(i) + factorsR(r)*factorsC(col_idx(i)) 
            i += 1
          }
        }
      }
      else {
        var r = 0
        while (r < numRows) {
          var i = row_ptr(r)
          while (i < row_ptr(r+1)) {
            res_r(i) = res_r(i) + factorsR(r)*factorsC(col_idx(i)) 
            i += 1
          }
          r += 1
        }
      }
    }
    else {
      if (multicore) {
        for (r <- (0 until numRows).par)  {
          var i = row_ptr(r)
          while (i < row_ptr(r+1)) {
            res_r(i) = res_r(i) - factorsR(r)*factorsC(col_idx(i)) 
            i += 1
          }
        }
      }
      else {
        var r = 0
        while (r < numRows) {
          var i = row_ptr(r)
          while (i < row_ptr(r+1)) {
            res_r(i) = res_r(i) - factorsR(r)*factorsC(col_idx(i)) 
            i += 1
          }
          r += 1
        }
      }
    }
  }//end of updateResidual
  
  def getResidual(
      row_ptr: Array[Int], col_idx: Array[Int], value_r: Array[Float], 
      factorsR: Array[Array[Float]], factorsC: Array[Array[Float]],
      multicore: Boolean, res_r: Array[Float]) : Array[Float] = {
    
    val numFactors = factorsR.length; val numRows = factorsR(0).length
    if (multicore) {
      for (r <- (0 until numRows).par) {
        var i = row_ptr(r)
        while (i < row_ptr(r+1)) {
          var k = 0; var pred = 0f
          while (k < numFactors) {
            pred += factorsR(k)(r)*factorsC(k)(col_idx(i))
            k += 1
          }
          res_r(i) = value_r(i) - pred
          i += 1
        }
      }
    }
    else {
      var r = 0
      while (r < numRows) {
        var i = row_ptr(r)
        while (i < row_ptr(r+1)) {
          var k = 0; var pred = 0f
          while (k < numFactors) {
            pred += factorsR(k)(r)*factorsC(k)(col_idx(i))
            k += 1
          }
          res_r(i) = value_r(i) - pred
          i += 1
        }
        r += 1
      }
    }
    value_r
  } // end of getResidual
  
  def getRMSE(res: Array[Float], multicore: Boolean = false) : Float = {
    val nnz = res.length
    val se =
      if (multicore) res.par.map(r => r*r).fold(0f)(_+_)
      else res.view.map(r => r*r).sum
    math.sqrt(se/nnz).toFloat
  }
  
  def updateGamma(means: Array[Array[Float]], precisions: Array[Array[Float]],
      priors: Array[Array[Float]], gamma: Array[Float]) {
    val numFactors = means.length; val numRows = means(0).length
    var k=0
    while (k < numFactors) {
      var r = 0
      var denominator = 0f
      while (r < numRows) {
        val res = if (priors == null) means(k)(r) else means(k)(r) - priors(r)(k)
        denominator += res*res + 1/precisions(k)(r)
        r += 1
      }
      gamma(k) = (numRows-1+0.01f)/(denominator+0.01f)
      k += 1
    }
  } //end of updateGamma
}
