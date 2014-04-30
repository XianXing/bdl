package tm

import breeze.linalg._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkContext, Partitioner, HashPartitioner}
import org.apache.spark.broadcast.Broadcast


class DivideAndConquer (
    val trainingDocs: RDD[(Int, CSCMatrix[Int])],
    val validatingDocs: RDD[(Int, CSCMatrix[Int])],
    val localModels: RDD[(Int, LDA)],
    override val topics: DenseMatrix[Double],
    val dualVariables: RDD[(Int, DenseMatrix[Double])]
    ) extends Model(topics) {

}

object DivideAndConquer {
  def apply(sc: SparkContext, partitioner: Partitioner,
      trainingDocsDir: String, validatingDocsDir: String,
      numTopics: Int, numDocs: Int, numBlocks: Int, alphaInit: Double, beta0Init: Double,
      isEC: Boolean, hasPrior: Boolean, isVB: Boolean, emBayes: Boolean,
      multicore: Boolean) = {
    
    val trainingDocs = sc.textFile(trainingDocsDir, numBlocks).zipWithIndex
      .map{case(line, id) => (id % numBlocks, preprocess.TM.toSparseVector(line))}
      .groupByKey(partitioner).map(_._2.toArray)
    
    
  }
  
  
}