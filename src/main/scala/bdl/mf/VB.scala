package mf

object VB{
  
  def update(
      r: Int, row_ptr: Array[Int], col_idx: Array[Int], res_r: Array[Float],
      colMean: Array[Float], colPrecision: Array[Float], prior: Float, gamma_r: Float, 
      rowMean: Array[Float], rowPrecision: Array[Float]
      ) = {
    var i = row_ptr(r); var numerator = 0f; var denominator = 0f; var c = 0
    val rowMean_rk = rowMean(r)
    while (i < row_ptr(r+1)) {
      c = col_idx(i)
      numerator += (res_r(i)+rowMean_rk*colMean(c))*colMean(c)
      denominator += colMean(c)*colMean(c) + 1/colPrecision(c)
      i += 1
    }
    rowPrecision(r) = denominator+gamma_r
    rowMean(r) = (numerator+gamma_r*prior)/rowPrecision(r)
    i = row_ptr(r)
    while (i < row_ptr(r+1)) {
      c = col_idx(i)
      res_r(i) += (rowMean_rk - rowMean(r))*colMean(c) 
      i += 1
    }
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
  
  def updateGamma(means: Array[Array[Float]], precisions: Array[Array[Float]],
      priors: Array[Array[Float]], gamma: Array[Float]) = {
    val numFactors = means.length; val numRows = priors.length
    var k=0; var r=0; var denominator = 0f; var res = 0f
    while (k < numFactors) {
      r = 0
      denominator = 0f
      while (r < numRows) {
        res = means(k)(r) - priors(r)(k)
        denominator += res*res + 1/precisions(k)(r)
        r += 1
      }
      gamma(k) = (numRows-1+0.01f)/(denominator+0.01f)
      k += 1
    }
  }
}