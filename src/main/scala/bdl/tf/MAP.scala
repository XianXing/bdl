package tf

object MAP{
  
  def update(
      d: Int, arr: Array[Int], idx2: Array[Int], idx3: Array[Int], res: Array[Float], 
      factor2: Array[Float], factor3: Array[Float], prior: Float, gamma: Float, 
      factor1: Array[Float]) = {
    var numerator = 0f; var denominator = 0f
    for (n <- arr) {
      val prod = factor2(idx2(n))*factor3(idx3(n))
      numerator += res(n)*prod
      denominator += prod*prod
    }
    factor1(d) = (numerator+gamma*prior)/(denominator+gamma)
  }
  
  def update(
      d: Int, start: Int, end: Int, 
      idx2: Array[Int], idx3: Array[Int], res: Array[Float], 
      factor2: Array[Float], factor3: Array[Float], prior: Float, gamma: Float, 
      factor1: Array[Float]) = {
    var numerator = 0f; var denominator = 0f; var n = start
    while (n < end) {
      val prod = factor2(idx2(n))*factor3(idx3(n))
      numerator += res(n)*prod
      denominator += prod*prod
      n += 1
    }
    factor1(d) = (numerator+gamma*prior)/(denominator+gamma)
  }
  
  def update(idx1: Array[Int], idx2: Array[Int], idx3: Array[Int], res: Array[Float],
      factor2: Array[Float], factor3: Array[Float], prior1: Array[Float], gamma1: Float,
      factor1: Array[Float]) : Float = {
    val length = factor1.length
    var l = 0
    while (l < length) {
      factor1(l) = gamma1*prior1(l)
      l += 1
    }
    val prec1 = Array.fill(length)(gamma1)
    val nnz = idx1.length
    var n = 0
    while (n < nnz) {
      val prod = factor2(idx2(n))*factor3(idx3(n))
      prec1(idx1(n)) += prod*prod
      factor1(idx1(n)) += res(n)*prod
      n += 1
    }
    var de = 0f
    l = 0
    while (l < length) {
      factor1(l) /= prec1(l)
      val diff = factor1(l) - prior1(l)
      de += diff*diff
      l += 1
    }
    de
  }
  
  def updateGamma(factors: Array[Array[Float]], priors: Array[Array[Float]], 
      gamma: Array[Float]) = {
    val length = priors.length; val numFactors = factors.length
    var k = 0
    while(k < numFactors) {
      var r = 0
      var denominator = 0f
      while (r < length) {
        val res = factors(k)(r) - priors(r)(k)
        denominator += res*res
        r += 1
      }
      gamma(k) = (length-1)/denominator
      k+=1
    }
  }
}