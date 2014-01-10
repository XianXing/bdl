package utilities

import org.apache.spark.serializer.KryoRegistrator
import com.esotericsoftware.kryo.Kryo

class Registrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[SparseMatrix])
    kryo.register(classOf[SparseCube])
    kryo.register(classOf[SparseVector])
    kryo.register(classOf[Record])
    //This avoids a large number of hash table look-ups
    kryo.setReferences(false)
  }
}