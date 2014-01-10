package tf

object VB{
  
  def update(
      d: Int, arr: Array[Int], idx2: Array[Int], idx3: Array[Int], res: Array[Float], 
      factor2: Array[Float], prec2: Array[Float], 
      factor3: Array[Float], prec3: Array[Float], 
      prior: Float, gamma: Float, 
      factor1: Array[Float], prec1: Array[Float]) = {
    var numerator = 0f; var denominator = 0f; var i = 0; val length = arr.length
    while (i < length) {
      val n = arr(i)
      val f2 = factor2(idx2(n)); val f3 = factor3(idx3(n))
      numerator += res(n)*f2*f3
      denominator += (f2*f2 + 1/prec2(idx2(n)))*(f3*f3 + 1/prec3(idx3(n)))
      i += 1
    }
    prec1(d) = denominator+gamma
    factor1(d) = (numerator+gamma*prior)/prec1(d)
  }
  
  def update(idx1: Array[Int], idx2: Array[Int], idx3: Array[Int], res: Array[Float],
      factor2: Array[Float], prec2: Array[Float], 
      factor3: Array[Float], prec3: Array[Float], prior1: Array[Float], gamma1: Float, 
      factor1: Array[Float], prec1: Array[Float]) : Float = {
    val length = factor1.length
    var l = 0
    while (l < length) {
      factor1(l) = gamma1*prior1(l)
      prec1(l) = gamma1
      l += 1
    }
    val nnz = idx1.length
    var n = 0
    while (n < nnz) {
      val f2 = factor2(idx2(n)); val f3 = factor3(idx3(n))
      prec1(idx1(n)) += (f2*f2 + 1/prec2(idx2(n)))*(f3*f3 + 1/prec3(idx3(n)))
      factor1(idx1(n)) += res(n)*f2*f3
      n += 1
    }
    var de = 0f
    l = 0
    while (l < length) {
      factor1(l) /= prec1(l)
      val diff = factor1(l) - prior1(l)
      de += diff*diff + 1/prec1(l)
      l += 1
    }
    de
  }
  
  def update(
      d: Int, start: Int, end: Int, 
      idx2: Array[Int], idx3: Array[Int], res: Array[Float], 
      factor2: Array[Float], prec2: Array[Float], 
      factor3: Array[Float], prec3: Array[Float], 
      prior: Float, gamma: Float, 
      factor1: Array[Float], prec1: Array[Float]) = {
    var numerator = 0f; var denominator = 0f; var n = start
    while (n < end) {
      val f2 = factor2(idx2(n)); val f3 = factor3(idx3(n))
      numerator += res(n)*f2*f3
      denominator += (f2*f2 + 1/prec2(idx2(n)))*(f3*f3 + 1/prec3(idx3(n)))
      n += 1
    }
    prec1(d) = denominator+gamma
    factor1(d) = (numerator+gamma*prior)/prec1(d)
  }
  
  def updateGamma(factors: Array[Array[Float]], precisions: Array[Array[Float]],
      priors: Array[Array[Float]], gamma: Array[Float]) = {
    val numFactors = factors.length; val length = priors.length
    var k=0; var r=0; var denominator = 0f; var res = 0f
    while (k < numFactors) {
      r = 0
      denominator = 0f
      while (r < length) {
        res = factors(k)(r) - priors(r)(k)
        denominator += res*res + 1/precisions(k)(r)
        r += 1
      }
      gamma(k) = (length-1+0.01f)/(denominator+0.01f)
      k += 1
    }
  }
}