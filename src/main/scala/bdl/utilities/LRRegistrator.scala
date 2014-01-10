package utilities

import org.apache.spark.serializer.KryoRegistrator
import com.esotericsoftware.kryo.Kryo

class LRRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[SparseVector])
    kryo.register(classOf[Vector])
    //This avoids a large number of hash table look-ups
    kryo.setReferences(false)
  }
}
