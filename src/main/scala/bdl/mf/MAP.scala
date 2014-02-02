package mf

object MAP{
  
  def update(
      r: Int, row_ptr: Array[Int], col_idx: Array[Int], 
      res_r: Array[Float], colFactors: Array[Float], prior: Float,
      gamma_r: Float, rowFactors: Array[Float]
      ) = {
    var i = row_ptr(r); var numerator = 0f; var denominator = 0f
    val rowFactor_rk = rowFactors(r)
    while (i < row_ptr(r+1)) {
      val c = col_idx(i)
      numerator += (res_r(i)+rowFactor_rk*colFactors(c))*colFactors(c)
      denominator += colFactors(c)*colFactors(c)
      i += 1
    }
    rowFactors(r) = (numerator+gamma_r*prior)/(denominator+gamma_r)
    i = row_ptr(r)
    while (i < row_ptr(r+1)) {
      val c = col_idx(i)
      res_r(i) += (rowFactor_rk - rowFactors(r))*colFactors(c) 
      i += 1
    }
  }
  def updatepp(
      r: Int, row_ptr: Array[Int], col_idx: Array[Int], 
      res_r: Array[Float], colFactors: Array[Float], prior: Float,
      gamma_r: Float, rowFactors: Array[Float]
      ) = {
    var i = row_ptr(r); var numerator = 0f; var denominator = 0f
    while (i < row_ptr(r+1)) {
      val c = col_idx(i)
      numerator += res_r(i)*colFactors(c)
      denominator += colFactors(c)*colFactors(c)
      i += 1
    }
    rowFactors(r) = (numerator+gamma_r*prior)/(denominator+gamma_r)
  }
  
  def updatepp(
      r: Int, row_ptr: Array[Int], col_idx: Array[Int], res_r: Array[Float], 
      colMean: Array[Float], colPrecision: Array[Float], prior: Float, gamma_r: Float, 
      rowMean: Array[Float], rowPrecision: Array[Float]
      ) = {
    var i = row_ptr(r); var numerator = 0f; var denominator = 0f; var c = 0
    while (i < row_ptr(r+1)) {
      c = col_idx(i)
      numerator += res_r(i)*colMean(c)
      denominator += colMean(c)*colMean(c) + 1/colPrecision(c)
      i += 1
    }
    rowPrecision(r) = denominator+gamma_r
    rowMean(r) = (numerator+gamma_r*prior)/rowPrecision(r)
  }
  
  def updateGamma(factors: Array[Array[Float]], priors: Array[Array[Float]], 
      gamma: Array[Float]) = {
    val numRows = priors.length; val numFactors = factors.length
    var k = 0
    while(k < numFactors) {
      var r = 0
      var denominator = 0f
      while (r < numRows) {
        val res = factors(k)(r) - priors(r)(k)
        denominator += res*res
        r += 1
      }
      gamma(k) = (numRows-1)/denominator
      k+=1
    }
  }
}