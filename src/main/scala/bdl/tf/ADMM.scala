package tf

object ADMM{
  
  def updateLag(rowFactors: Array[Array[Float]], multicore: Boolean,
      rowLags: Array[Array[Float]], rowPriors: Array[Array[Float]]) = {
    //update the scaled Lagrangian multipilers
    val numFactors = rowLags.length
    val numRows = rowPriors.length
    
    if (multicore)
    for (r <- (0 until numRows).par) {
      var k = 0
      while (k < numFactors) {
        rowLags(k)(r) += rowFactors(k)(r) - rowPriors(r)(k)
        rowPriors(r)(k) -= rowLags(k)(r)
        k += 1
      }
    }
    else {
      var k = 0
      while (k < numFactors) {
        var r = 0
        while (r < numRows) {
          rowLags(k)(r) += rowFactors(k)(r) - rowPriors(r)(k)
          rowPriors(r)(k) -= rowLags(k)(r)
          r+=1
        }
        k += 1
      }
    }
  }
}