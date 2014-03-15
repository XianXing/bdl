package utilities

object Math {
  def sigmoid(value : Double) : Double = {
    if (value < -10) 4.5398e-05
    else if (value > 10) 1-1e-05f
    else 1 / (1 + math.exp(-value))
  }
}