package utilities
import org.apache.hadoop.io.Writable
import java.io.DataInput;
import java.io.DataOutput;

class Quadruplet (var _1: Int, var _2: Int, var _3: Int, var _4: Int, var value: Float) 
  extends Writable with Serializable {
  
  def this() = this(0, 0, 0, 0, 0f)
  
  def this (tuple5: (Int, Int, Int, Int, Float)) 
    = this(tuple5._1, tuple5._2, tuple5._3, tuple5._4, tuple5._5)
  
  def readFields(in: DataInput) = {
    _1 = in.readInt
    _2 = in.readInt
    _3 = in.readInt
    _4 = in.readInt
    value = in.readFloat
  }
  
  def write(out: DataOutput) = {
    out.writeInt(_1)
    out.writeInt(_2)
    out.writeInt(_3)
    out.writeInt(_4)
    out.writeFloat(value)
  }

}