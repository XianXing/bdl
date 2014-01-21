package utilities

import java.io.DataInput
import java.io.DataOutput
import org.apache.hadoop.io.Writable

/**
 * A more compact class to represent a rating than Tuple3[Int, Int, Float].
 */

class Record(var rowIdx: Int, var colIdx: Int, var value: Float)
  extends Writable with Serializable {
  
  def this() = this(0, 0, 0f)
  
  def readFields(in: DataInput) = {
    rowIdx = in.readInt
    colIdx = in.readInt
    value = in.readFloat
  }
  
  def write(out: DataOutput) = {
    out.writeInt(rowIdx)
    out.writeInt(colIdx)
    out.writeFloat(value)
  }
}