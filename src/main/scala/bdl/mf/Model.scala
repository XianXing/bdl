package mf

import utilities._
import java.util._

class Model (
    val rowFactor: Array[Array[Float]], val colFactor: Array[Array[Float]],
    rowPrecision: Array[Array[Float]], colPrecision: Array[Array[Float]],
    rowLags: Array[Array[Float]], colLags: Array[Array[Float]],
    val rowMap: Array[Int], val colMap: Array[Int], 
    val gamma_r: Array[Float], val gamma_c: Array[Float], var gamma_x: Float
    ) extends Serializable {
  
  val numRows = rowFactor(0).length 
  val numCols = colFactor(0).length
  val numFactors = rowFactor.length
  
  def setGammaR(gamma: Float) = {
    val l = gamma_r.length
    var i = 0; while(i < l) { gamma_r(i) = gamma; i += 1}
  }
  
  def setGammaC(gamma: Float) = {
    val l = gamma_c.length
    var i = 0; while(i < l) { gamma_c(i) = gamma; i += 1}
  }
  
  def setRowFactor(rowFactor: Array[Array[Float]], transpose: Boolean) = {
    var r = 0; var k = 0
    if (transpose) {
      while (k< numFactors) {
        while (r < numRows) {
          this.rowFactor(k)(r) = rowFactor(r)(k)
          r += 1
        }
        k += 1
      }
    }
    else {
      while (k< numFactors) {
        while (r < numRows) {
          this.rowFactor(k)(r) = rowFactor(k)(r)
          r += 1
        }
        k += 1
      }
    }
  }
  
  def setColFactor(colFactor: Array[Array[Float]], transpose: Boolean) = {
    var c = 0; var k = 0
    if (transpose) {
      while (k< numFactors) {
        while (c < numCols) {
          this.colFactor(k)(c) = colFactor(c)(k)
          c += 1
        }
        k += 1
      }
    }
    else {
      while (k< numFactors) {
        while (c < numCols) {
          this.colFactor(k)(c) = colFactor(k)(c)
          c += 1
        }
        k += 1
      }
    }
  }
  
  def getRowStats(admm_r: Boolean) : Array[Array[Float]] = {
    if (admm_r) {
      var k = 0
      val rowStats = Array.ofDim[Float](numFactors, numRows)
      while (k < numFactors) {
        var r = 0
        while (r < numRows) {rowStats(k)(r) = rowFactor(k)(r) + rowLags(k)(r); r+=1}
        k += 1
      }
      rowStats
    }
    else rowFactor
  }
  def getColStats(admm_c: Boolean) : Array[Array[Float]] = {
    if (admm_c) {
      var k = 0
      val colStats = Array.ofDim[Float](numFactors, numCols)
      while (k < numFactors) {
        var c = 0
        while (c < numCols) {colStats(k)(c) = colFactor(k)(c) + colLags(k)(c); c+=1}
        k += 1
      }
      colStats
    }
    else 
      colFactor
  }
  
  def ccd(
      data: SparseMatrix, maxIter: Int, thre: Float, 
      multicore: Boolean, vb: Boolean, admm_r: Boolean, admm_c: Boolean, 
      updateGammaR: Boolean = false, updateGammaC: Boolean = false,
      rowPriors: Array[Array[Float]] = Array.ofDim[Float](numRows, numFactors),
      colPriors: Array[Array[Float]] = Array.ofDim[Float](numCols, numFactors)
      ): (Model, Int, Float, Float, Float) = {
    
    //priors are M*K and N*K respectively, while factors are K*M and K*N respectively
    val row_ptr = data.row_ptr; val row_idx = data.row_idx
    val col_ptr = data.col_ptr; val col_idx = data.col_idx
    val value_r = data.value_r; val value_c = data.value_c
    
    //update the scaled Lagrangian multipilers
    if (admm_r) ADMM.updateLag(rowFactor, multicore, rowLags, rowPriors)
    if (admm_c) ADMM.updateLag(colFactor, multicore, colLags, colPriors)
    
    var iter = 0; var rmse_old = Float.MinValue; var rmse = 0f
    var rowSDe = 0f; var colSDe = 0f
    val rows = Array.tabulate(numRows)(i => i)
    val cols = Array.tabulate(numCols)(i => i)
    val res_r = Array.ofDim[Float](value_r.length)
    val res_c = Array.ofDim[Float](value_c.length)
    while (iter < maxIter && math.abs(rmse_old-rmse) > thre) {
      Model.getResidual(col_ptr, row_idx, value_c, colFactor, rowFactor, 
          multicore, res_c)
      Model.getResidual(row_ptr, col_idx, value_r, rowFactor, colFactor, 
          multicore, res_r)
      if (multicore) {
        //update col factors
        colSDe = cols.par.map(c => {
          var sd = 0f; var k = 0
          while (k<numFactors) {
            if (vb)
              VB.update(c, col_ptr, row_idx, res_c, rowFactor(k), rowPrecision(k),
                colPriors(c)(k), gamma_c(k), colFactor(k), colPrecision(k))
            else
              MAP.update(c, col_ptr, row_idx, res_c, rowFactor(k),
                colPriors(c)(k), gamma_c(k), gamma_x, colFactor(k))
            val diff = colFactor(k)(c) - colPriors(c)(k)
            val variance = if (vb) 1/colPrecision(k)(c) else 0
            sd += diff*diff + variance
            k += 1
          }
          sd
        }).fold(0f)(_+_)
        //update row factors
        rowSDe = rows.par.map(r => {
          var sd = 0f; var k = 0
          while (k<numFactors) {
            if (vb)
              VB.update(r, row_ptr, col_idx, res_r, colFactor(k), colPrecision(k),
                rowPriors(r)(k), gamma_r(k), rowFactor(k), rowPrecision(k))
            else
              MAP.update(r, row_ptr, col_idx, res_r, colFactor(k),
                rowPriors(r)(k), gamma_r(k), gamma_x, rowFactor(k))
            val variance = if (vb) 1/rowPrecision(k)(r) else 0
            val diff = rowFactor(k)(r) - rowPriors(r)(k) + variance
            sd += diff*diff
            k += 1
          }
          sd
        }).fold(0f)(_+_)
      }
      else //not using multi-thread
        colSDe = cols.map(c => {
          var sd = 0f; var k = 0
          while (k<numFactors) {
            if (vb)
              VB.update(c, col_ptr, row_idx, res_c, rowFactor(k), rowPrecision(k),
                colPriors(c)(k), gamma_c(k), colFactor(k), colPrecision(k))
            else
              MAP.update(c, col_ptr, row_idx, res_c, rowFactor(k),
                colPriors(c)(k), gamma_c(k), gamma_x, colFactor(k))
            val diff = colFactor(k)(c) - colPriors(c)(k)
            val variance = if (vb) 1/colPrecision(k)(c) else 0
            sd += diff*diff + variance
            k += 1
          }
          sd
        }).fold(0f)(_+_)
        rowSDe = rows.map(r => {
          var sd = 0f; var k = 0
          while (k<numFactors) {
            if (vb)
              VB.update(r, row_ptr, col_idx, res_r, colFactor(k), colPrecision(k),
                rowPriors(r)(k), gamma_r(k), rowFactor(k), rowPrecision(k))
            else
              MAP.update(r, row_ptr, col_idx, res_r, colFactor(k),
                rowPriors(r)(k), gamma_r(k), gamma_x, rowFactor(k))
            val diff = rowFactor(k)(r) - rowPriors(r)(k)
            val variance = if (vb) 1/rowPrecision(k)(r) else 0
            sd += diff*diff + variance
            k += 1    
          }
          sd
        }).fold(0f)(_+_)
      
      rmse = Model.getRMSE(res_r)
      iter += 1
    }
    if (updateGammaR) {
      if (vb) VB.updateGamma(rowFactor, rowPrecision, rowPriors, gamma_r)
      else MAP.updateGamma(rowFactor, rowPriors, gamma_r)
    }
    if (updateGammaC)
      if (vb) VB.updateGamma(colFactor, colPrecision, colPriors, gamma_c)
      else MAP.updateGamma(colFactor, colPriors, gamma_c)
    rowSDe = math.sqrt(rowSDe/(numRows*numFactors)).toFloat
    colSDe = math.sqrt(colSDe/(numCols*numFactors)).toFloat
    (this, iter, rmse, rowSDe, colSDe)
  }
  
  def ccdpp(
      data: SparseMatrix, maxOuterIter: Int, maxInnerIter: Int, thre: Float, 
      multicore: Boolean, vb: Boolean, admm_r: Boolean, admm_c: Boolean, 
      updateGammaR: Boolean = false, updateGammaC: Boolean = false,
      rowPriors: Array[Array[Float]] = Array.ofDim[Float](numRows, numFactors),
      colPriors: Array[Array[Float]] = Array.ofDim[Float](numCols, numFactors)
      ): (Model, Int, Float, Float, Float) = {
    
    //priors are M*K and N*K respectively, while factors are K*M and K*N respectively
    val row_ptr = data.row_ptr; val row_idx = data.row_idx
    val col_ptr = data.col_ptr; val col_idx = data.col_idx
    val value_r = data.value_r; val value_c = data.value_c
    
    //here value_c and value_r are actually res_r and res_c respectively
    val res_r = value_r
    val res_c = value_c
    
    if (admm_r) ADMM.updateLag(rowFactor, multicore, rowLags, rowPriors)
    if (admm_c) ADMM.updateLag(colFactor, multicore, colLags, colPriors)
    
    var iter = 0; var rmse_old = Float.MinValue; var rmse = 0f
    var rowSDe = 0f; var colSDe = 0f
    val rows = Array.tabulate(numRows)(i => i)
    val cols = Array.tabulate(numCols)(i => i)
    while (iter < maxOuterIter && math.abs(rmse_old-rmse) > thre) {
      iter += 1; var k = 0; rmse_old = rmse; rowSDe = 0f; colSDe = 0f
      while (k<numFactors) {
        Model.updateResidual(row_ptr, col_idx, res_r, 
            rowFactor(k), colFactor(k), true, multicore)
        Model.updateResidual(col_ptr, row_idx, res_c, 
            colFactor(k), rowFactor(k), true, multicore)
        var i = 0
        while (i < maxInnerIter) {
          i += 1
          val colDe = if (multicore) 
            cols.par.map(c => {
              if (vb)
                VB.updatepp(c, col_ptr, row_idx, res_c, rowFactor(k), rowPrecision(k),
                  colPriors(c)(k), gamma_c(k), colFactor(k), colPrecision(k))
              else 
                MAP.updatepp(c, col_ptr, row_idx, res_c, rowFactor(k), 
                  colPriors(c)(k), gamma_c(k), gamma_x, colFactor(k))
              val diff = colFactor(k)(c) - colPriors(c)(k)
              val de = if (vb) diff*diff + 1/colPrecision(k)(c) else diff*diff
              de
            }).fold(0f)(_+_)
          else 
            cols.map(c => {
              if (vb)
                VB.updatepp(c, col_ptr, row_idx, res_c, rowFactor(k), rowPrecision(k),
                  colPriors(c)(k), gamma_c(k), colFactor(k), colPrecision(k))
              else 
                MAP.updatepp(c, col_ptr, row_idx, res_c, rowFactor(k), 
                  colPriors(c)(k), gamma_c(k), gamma_x, colFactor(k))
              val diff = colFactor(k)(c) - colPriors(c)(k)
              val de = if (vb) diff*diff + 1/colPrecision(k)(c) else diff*diff
              de
            }).fold(0f)(_+_)
          
          if (updateGammaC && iter == maxOuterIter && i == maxInnerIter) {
            gamma_c(k) = (numCols-1)/(colDe+0.01f)
          }
          if (i == maxInnerIter) colSDe += colDe
          //update numFactors-1 row latent factors
          val rowDe = if (multicore) 
            rows.par.map(r => {
              if (vb)
                VB.updatepp(r, row_ptr, col_idx, res_r, colFactor(k), colPrecision(k),
                  rowPriors(r)(k), gamma_r(k), rowFactor(k), rowPrecision(k))
              else
                MAP.updatepp(r, row_ptr, col_idx, res_r, colFactor(k), 
                  rowPriors(r)(k), gamma_r(k), gamma_x, rowFactor(k))
              val diff = rowFactor(k)(r) - rowPriors(r)(k)
              val de = if (vb) diff*diff + 1/rowPrecision(k)(r) else diff*diff
              de
            }).fold(0f)(_+_)
          else 
            rows.map(r => {
              if (vb)
                VB.updatepp(r, row_ptr, col_idx, res_r, colFactor(k), colPrecision(k),
                  rowPriors(r)(k), gamma_r(k), rowFactor(k), rowPrecision(k))
              else
                MAP.updatepp(r, row_ptr, col_idx, res_r, colFactor(k), 
                  rowPriors(r)(k), gamma_r(k), gamma_x, rowFactor(k))
              val diff = rowFactor(k)(r) - rowPriors(r)(k)
              val de = if (vb) diff*diff + 1/rowPrecision(k)(r) else diff*diff
              de
            }).fold(0f)(_+_)
          
          if (updateGammaR && iter == maxOuterIter && i == maxInnerIter) {
            gamma_r(k) = (numRows-1)/(rowDe+0.01f)
          }
          if (i == maxInnerIter) rowSDe += rowDe
        }
        Model.updateResidual(row_ptr, col_idx, res_r, 
            rowFactor(k), colFactor(k), false, multicore)
        Model.updateResidual(col_ptr, row_idx, res_c, 
            colFactor(k), rowFactor(k), false, multicore)
        k += 1
      }
      rmse = Model.getRMSE(res_c)
//      println("training rmse: " + rmse)
    }
    if (iter < maxOuterIter && math.abs(rmse_old-rmse) <= thre) {
      if (updateGammaR) {
        if (vb) VB.updateGamma(rowFactor, rowPrecision, rowPriors, gamma_r)
        else MAP.updateGamma(rowFactor, rowPriors, gamma_r)
      }
      if (updateGammaC)
        if (vb) VB.updateGamma(colFactor, colPrecision, colPriors, gamma_c)
        else MAP.updateGamma(colFactor, colPriors, gamma_c)
    }
    rowSDe = math.sqrt(rowSDe/(numRows*numFactors)).toFloat
    colSDe = math.sqrt(colSDe/(numCols*numFactors)).toFloat
    (this, iter, rmse, rowSDe, colSDe)
  }
}

object Model {
  
  def apply (
      data: SparseMatrix, numFactors: Int, gamma_r_init: Float,
      gamma_c_init: Float, gamma_x_init: Float, 
      admm_r: Boolean, admm_c: Boolean, vb: Boolean
      ) : Model = {
    
    def hash(x: Int): Int = {
      val r = x ^ (x >>> 20) ^ (x >>> 12)
      r ^ (r >>> 7) ^ (r >>> 4)
    }
    
    val rowMap = data.rowMap; val colMap = data.colMap
    val numRows = data.numRows; val numCols = data.numCols
    val rowFactor = Array.ofDim[Float](numFactors, numRows)
    var r = 0
    val rand = new Random()
    while (r < numRows) {
      var k = 0
      rand.setSeed(rowMap(r))
      while (k < numFactors) {
        rowFactor(k)(r) = 0.1f*(rand.nextFloat-0.5f)
        k += 1
      }
      r += 1
    }
//  val rowFactor = Array.fill(numFactors, numRows)(0.1f*(Random.nextFloat-0.5f))
//  val colLatentFactor = Array.fill(numCols, numFactors)(0.1f*(Random.nextFloat-0.5f))
    val colFactor = Array.ofDim[Float](numFactors, numCols)
    val rowPrecision = 
      if (vb) Array.fill(numFactors, numRows)(gamma_r_init)
      else null
    val colPrecision = 
      if (vb) Array.fill(numFactors, numCols)(gamma_c_init)
      else null
    val rowLags = 
      if (admm_r) Array.ofDim[Float](numFactors, numRows)
      else null
    val colLags = 
      if (admm_c) Array.ofDim[Float](numFactors, numCols)
      else null
    val gamma_r = Array.fill(numFactors)(gamma_r_init)
    val gamma_c = Array.fill(numFactors)(gamma_c_init)
    val gamma_x = gamma_x_init
    new Model(rowFactor, colFactor, rowPrecision, colPrecision,
        rowLags, colLags, rowMap, colMap, gamma_r, gamma_c, gamma_x)
  }
  
  def toLocal(rowMap: Array[Int], pairs: Array[(Int, Array[Float])]) 
    : Array[Array[Float]] = {
    val numRows = rowMap.length; val numFactors = pairs(0)._2.length
    var i = 0; var j = 0
    var prior_r = Array.ofDim[Array[Float]](numRows)
    val l = pairs.length
    while (i < numRows && j < l) {
      if (rowMap(i) == pairs(j)._1) { 
        prior_r(i) = pairs(j)._2
        i+=1
        j+=1 
      }
      else if (rowMap(i) < pairs(j)._1) {
        prior_r(i) = Array.ofDim[Float](numFactors)
        i+=1
      }
      else j+=1
    }
    while (i < numRows) { prior_r(i) = Array.ofDim[Float](numFactors); i+=1 }
    prior_r
  }
  
  def getRMSE(res: Array[Float], multicore: Boolean = false) : Float = {
    val nnz = res.length
    val se =
      if (multicore) res.par.map(r => r*r).fold(0f)(_+_)
      else { var i = 0; var sd = 0f; while (i < nnz) { sd += res(i)*res(i); i+=1}; sd }
    math.sqrt(se/nnz).toFloat
  }
  
  def updateResidual(
      row_ptr: Array[Int], col_idx: Array[Int], res_r: Array[Float], 
      rowFactor: Array[Float], colFactor: Array[Float], 
      add: Boolean, multicore: Boolean = false) = {
    
    val numRows = rowFactor.length
    if (add) {
      if (multicore) {
        for (r <- (0 until numRows).par) {
          var i = row_ptr(r)
          while (i < row_ptr(r+1)) {
            res_r(i) = res_r(i) + rowFactor(r)*colFactor(col_idx(i)) 
            i += 1
          }
        }
      }
      else {
        var r = 0
        while (r < numRows) {
          var i = row_ptr(r)
          while (i < row_ptr(r+1)) {
            res_r(i) = res_r(i) + rowFactor(r)*colFactor(col_idx(i)) 
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
            res_r(i) = res_r(i) - rowFactor(r)*colFactor(col_idx(i)) 
            i += 1
          }
        }
      }
      else {
        var r = 0
        while (r < numRows) {
          var i = row_ptr(r)
          while (i < row_ptr(r+1)) {
            res_r(i) = res_r(i) - rowFactor(r)*colFactor(col_idx(i)) 
            i += 1
          }
          r += 1
        }
      }
    }
  }
  
  def getResidual(
      row_ptr: Array[Int], col_idx: Array[Int], value_r: Array[Float], 
      rowFactor: Array[Array[Float]], colFactor: Array[Array[Float]],
      multicore: Boolean = false, res_r: Array[Float]
      ) : Array[Float] = {
    
    val numFactors = rowFactor.length; val numRows = rowFactor(0).length
    if (multicore) {
      for (r <- (0 until numRows).par) {
        var i = row_ptr(r)
        while (i < row_ptr(r+1)) {
          var k = 0; var pred = 0f
          while (k < numFactors) {
            pred += rowFactor(k)(r)*colFactor(k)(col_idx(i))
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
            pred += rowFactor(k)(r)*colFactor(k)(col_idx(i))
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
}
