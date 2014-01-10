package utilities
import org.apache.hadoop.io.Writable
import java.io.DataInput;
import java.io.DataOutput;

class Triplet (var _1: Int, var _2: Int, var _3: Int, var value: Float) 
  extends Writable with Serializable {
  
  def this() = this(0, 0, 0, 0f)
  
  def this (tuple4: (Int, Int, Int, Float)) 
    = this(tuple4._1, tuple4._2, tuple4._3, tuple4._4)
  
  def readFields(in: DataInput) = {
    _1 = in.readInt
    _2 = in.readInt
    _3 = in.readInt
    value = in.readFloat
  }
  
  def write(out: DataOutput) = {
    out.writeInt(_1)
    out.writeInt(_2)
    out.writeInt(_3)
    out.writeFloat(value)
  }

}
