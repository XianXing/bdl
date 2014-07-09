package tm

object ModelType extends Enumeration with Serializable {
  type ModelType = Value
  val VADMM, MRLDA, HDLDA = Value
  override def toString = Value match  {
    case VADMM => "VADMM"
    case MRLDA => "MRLDA"
    case HDLDA => "HDLDA"
  }
}