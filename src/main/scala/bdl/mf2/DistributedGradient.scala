package mf2

import scala.util.{Random, Sorting}
import scala.collection.mutable.{BitSet, ArrayBuffer}

import org.apache.spark.{SparkContext, Partitioner, HashPartitioner}
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import org.apache.hadoop.io.NullWritable

import org.jblas.{FloatMatrix, SimpleBlas, Solve} 

import utilities.{Record, SparseMatrix}
import OptimizerType._
import RegularizerType._
import preprocess.MF._

private[mf2] case class OutLinkBlock(elementIds: Array[Int], shouldSend: Array[BitSet])
private[mf2] case class InLinkBlock(
  elementIds: Array[Int], ratingsForBlock: Array[Array[(Array[Int], Array[Float])]])

class DistributedGradient(
    val rowFactors: RDD[(Int, Array[Float])],
    val colFactors: RDD[(Int, Array[Float])],
    val rowBlockedParas: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))],
    val colBlockedParas: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))],
    val rowLinkBlocks: RDD[(Int, (InLinkBlock, OutLinkBlock))],
    val colLinkBlocks: RDD[(Int, (InLinkBlock, OutLinkBlock))],
    val partitioner: Partitioner,
    val validatingData: RDD[(Int, SparseMatrix)], val nnzVal: Int,
    val rowValIDToSend: RDD[(Int, Array[Int])],
    val colValIDToSend: RDD[(Int, Array[Int])],
    val numFactors: Int, val isVB: Boolean, val weightedReg: Boolean)
  extends Model(rowFactors, colFactors) {
  
  private def creat(rowFactors: RDD[(Int, Array[Float])],
    colFactors: RDD[(Int, Array[Float])],
    rowBlockedParas: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))],
    colBlockedParas: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))])
    : DistributedGradient = {
    new DistributedGradient(rowFactors, colFactors, rowBlockedParas, colBlockedParas,
        rowLinkBlocks, colLinkBlocks, partitioner, validatingData, nnzVal, 
        rowValIDToSend, colValIDToSend, numFactors, isVB, weightedReg)
  }
  
  override def init(numIter: Int, optType: OptimizerType, 
    regPara: Float, regType: RegularizerType): DistributedGradient = {
    train(numIter, optType, regPara, regType)
  }
  
  override def train(numIter: Int, optType: OptimizerType, 
    regPara: Float, regType: RegularizerType): DistributedGradient = {
    
    var oldRowParas = rowBlockedParas
    var oldColParas = colBlockedParas
    val rowInLinkBlocks = rowLinkBlocks.mapValues(_._1)
    val rowOutLinkBlocks = rowLinkBlocks.mapValues(_._2)
    val colInLinkBlocks = colLinkBlocks.mapValues(_._1)
    val colOutLinkBlocks = colLinkBlocks.mapValues(_._2)
    for (iter <- 1 to numIter) {
      val updatedColParas = DistributedGradient.update(oldRowParas, rowOutLinkBlocks, 
        oldColParas, colInLinkBlocks, partitioner, isVB, weightedReg, 
        numFactors, regPara, optType)
      updatedColParas.cache
      updatedColParas.count
      oldColParas.unpersist(true)
      oldColParas = updatedColParas
      val updatedRowParas = DistributedGradient.update(oldColParas, colOutLinkBlocks, 
        oldRowParas, rowInLinkBlocks, partitioner, isVB, weightedReg, 
        numFactors, regPara, optType)
      updatedRowParas.cache
      updatedRowParas.count
      oldRowParas.unpersist(true)
      oldRowParas = updatedRowParas
    }
    val updatedRowBlockedParas = oldRowParas
    val updatedColBlockedParas = oldColParas
    val updatedRowFactors = DistributedGradient.unblockFactors(updatedRowBlockedParas,
      rowOutLinkBlocks).cache
    updatedRowFactors.count
    rowFactors.unpersist(true)
    val updatedColFactors = DistributedGradient.unblockFactors(updatedColBlockedParas,
      colOutLinkBlocks).cache
    updatedColFactors.count
    colFactors.unpersist(true)
    creat(updatedRowFactors, updatedColFactors, 
      updatedRowBlockedParas, updatedColBlockedParas)
  }
  
  override def getValidatingRMSE(global: Boolean): Double = {
    Model.getGlobalRMSE(validatingData, nnzVal, factorsR, factorsC, 
      rowValIDToSend, colValIDToSend)
  }
}

object DistributedGradient{
  
  private val storageLevel = MEMORY_AND_DISK
  
  def apply(sc: SparkContext, trainingDir: String, validatingDir: String,
      mean: Float, scale: Float, syn: Boolean, numBlocks: Int, 
      numFactors: Int, regPara: Float, isVB: Boolean, weightedReg: Boolean)
    : DistributedGradient = {
    
    val trainingRecords = 
      DistributedGradient.toRecords(sc, trainingDir, numBlocks, syn, mean, scale)
      .persist(storageLevel)
    val partitioner = new HashPartitioner(numBlocks)
    val recordsByRowBlock = trainingRecords.map{
      record => (record.rowIdx % numBlocks, record)
    }.groupByKey(partitioner)
    val recordsByColBlock = trainingRecords.map{ 
      record => (record.colIdx % numBlocks, 
          new Record(record.colIdx, record.rowIdx, record.value))
    }.groupByKey(partitioner)
    val rowLinks = makeLinkRDDs(numBlocks, recordsByRowBlock).persist(storageLevel)
    val rowInLinks = rowLinks.mapValues(_._1)
    val rowOutLinks = rowLinks.mapValues(_._2)
    val colLinks = makeLinkRDDs(numBlocks, recordsByColBlock).persist(storageLevel)
    val colInLinks = colLinks.mapValues(_._1)
    val colOutLinks = colLinks.mapValues(_._2)
    val seedGen = new Random()
    val rowSeed = seedGen.nextInt()
    val colSeed = seedGen.nextInt()
    val rowBlockedParas = 
      if (isVB) initVBParas(rowInLinks, numFactors, regPara, rowSeed)
      else initParas(rowOutLinks, numFactors, rowSeed)
    rowBlockedParas.cache
    rowBlockedParas.count
    val colBlockedParas = 
      if (isVB) initVBParas(colInLinks, numFactors, regPara, colSeed)
      else initParas(colOutLinks, numFactors, colSeed)
    colBlockedParas.cache
    colBlockedParas.count
    val rowFactors = unblockFactors(rowBlockedParas, rowOutLinks)
    val colFactors = unblockFactors(colBlockedParas, colOutLinks)
    val numRows = rowInLinks.map{
      case(id, inlink) => inlink.elementIds.last
    }.reduce(math.max(_, _))
    println("num rows:" + numRows)
    val numCols = colInLinks.map{
      case(id, inlink) => inlink.elementIds.last
    }.reduce(math.max(_, _))
    println("num cols:" + numCols)
    val numRowBlocks = math.sqrt(numBlocks).toInt
    val numColBlocks = numBlocks/numRowBlocks
    val rowBlockMap = 
      if (!syn) sc.broadcast(getPartitionMap(numRows+1, numRowBlocks, rowSeed))
      else null
    val colBlockMap = 
      if(!syn) sc.broadcast(getPartitionMap(numCols+1, numColBlocks, colSeed))
      else null
    val validatingData = toSparseMatrixBlocks(sc, validatingDir, rowBlockMap, 
        colBlockMap, numRowBlocks, numColBlocks, partitioner, 
        syn, mean, scale).cache
    val nnzVal = validatingData.map(_._2.col_idx.length).reduce(_+_)
    val rowValIDToSend = validatingData.flatMap{
      case (pid, data) => data.rowMap.map((_, List(pid)))
    }.reduceByKey(_:::_).mapValues(_.toArray).cache
    val colValIDToSend = validatingData.flatMap{
      case (pid, data) => data.colMap.map((_, List(pid)))
    }.reduceByKey(_:::_).mapValues(_.toArray).cache
    trainingRecords.unpersist(true)
    new DistributedGradient(rowFactors, colFactors, rowBlockedParas, colBlockedParas,
        rowLinks, colLinks, partitioner, validatingData, nnzVal, rowValIDToSend, 
        colValIDToSend, numFactors, isVB, weightedReg)
  }
  
  private def makeInLinkBlock(numBlocks: Int, records: Array[Record]): InLinkBlock = {
    val rowIds = records.map(_.rowIdx).distinct.sorted
    val numRows = rowIds.length
    val userIdToPos = rowIds.zipWithIndex.toMap
    // Split out our ratings by column block
    val blockRatings = Array.fill(numBlocks)(new ArrayBuffer[Record])
    for (r <- records) {
      blockRatings(r.colIdx % numBlocks) += r
    }
    val ratingsForBlock = new Array[Array[(Array[Int], Array[Float])]](numBlocks)
    for (colBID <- 0 until numBlocks) {
      // Create an array of (colID, Seq(Records)) ratings
      val groupedRatings = blockRatings(colBID).groupBy(_.colIdx).toArray
      // Sort them by product ID
      val ordering = new Ordering[(Int, ArrayBuffer[Record])] {
        def compare(a: (Int, ArrayBuffer[Record]), b: (Int, ArrayBuffer[Record])): Int
          = a._1 - b._1
      }
      Sorting.quickSort(groupedRatings)(ordering)
      // Translate the user IDs to indices based on userIdToPos
      ratingsForBlock(colBID) = groupedRatings.map { case (colID, rs) =>
        (rs.view.map(r => userIdToPos(r.rowIdx)).toArray, rs.view.map(_.value).toArray)
      }
    }
    InLinkBlock(rowIds, ratingsForBlock)
  }
  
  private def makeOutLinkBlock(numBlocks: Int, records: Array[Record])
    : OutLinkBlock = {
    val rowIds = records.map(_.rowIdx).distinct.sorted
    val numRows = rowIds.length
    val rowIdToPos = rowIds.zipWithIndex.toMap
    val shouldSend = Array.fill(numRows)(new BitSet(numBlocks))
    for (r <- records) {
      shouldSend(rowIdToPos(r.rowIdx))(r.colIdx % numBlocks) = true
    }
    OutLinkBlock(rowIds, shouldSend)
  }
  
  private def makeLinkRDDs(numBlocks: Int, groupedRecords: RDD[(Int, Iterable[Record])])
    : RDD[(Int, (InLinkBlock, OutLinkBlock))] = {
    groupedRecords.mapValues(records => {
      val recordsArr = records.toArray
      val inLinkBlock = makeInLinkBlock(numBlocks, recordsArr)
      val outLinkBlock = makeOutLinkBlock(numBlocks, recordsArr)
      (inLinkBlock, outLinkBlock)
    })
  }
  
  private def toRecords(sc: SparkContext, inputDir: String, numBlocks: Int, 
      syn: Boolean, mean: Float, scale: Float)
    : RDD[Record] = {
    if (syn) {
      sc.objectFile[((Int, Int), 
          (Array[Int], Array[Int], Array[Float]))](inputDir, numBlocks)
      .flatMap{case(id, (rowIndices, colIndices, values)) => 
        (rowIndices.view.zip(colIndices)).view.zip(values).map{
           case((rowIdx, colIdx), value) => new Record(rowIdx, colIdx, value-mean)
        }
      }
    }
    else if (inputDir.toLowerCase().contains("ml")) {
      sc.sequenceFile[NullWritable, Record](inputDir, numBlocks).map(pair => 
        new Record(pair._2.rowIdx, pair._2.colIdx, (pair._2.value-mean)/scale))
    } else {
      sc.textFile(inputDir, numBlocks).flatMap(line => parseLine(line, mean, scale))
    }
  }
  
  //updated rowBlockedParas
  private def update(
      colBlockedParas: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))],
      colOutLinks: RDD[(Int, OutLinkBlock)],
      rowBlockedParas: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))],
      rowInLinks: RDD[(Int, InLinkBlock)],
      partitioner: Partitioner,
      isVB: Boolean,
      weightedReg: Boolean,
      numFactors: Int,
      regPara: Float,
      optimizerType: OptimizerType)
    : RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))] = {
    val numBlocks = colBlockedParas.partitions.size
    val colBlockedMessages = colOutLinks.join(colBlockedParas).flatMap {
      case (colBID, (colOutLink, (colFactors, colPrecisions))) => 
      val colToSendFactors = Array.fill(numBlocks)(new ArrayBuffer[Array[Float]])
      val colToSendPrecisions = Array.fill(numBlocks)(new ArrayBuffer[Array[Float]])
      for (p <- 0 until colOutLink.elementIds.length; rowBID <- 0 until numBlocks) {
        if (colOutLink.shouldSend(p)(rowBID)) {
          colToSendFactors(rowBID) += colFactors(p)
          if (isVB) colToSendPrecisions(rowBID) += colPrecisions(p)
        }
      }
      colToSendFactors.zip(colToSendPrecisions).zipWithIndex.map{ 
        case ((colFactorsBuf, colPrecisionBuf), rowBID) => 
          (rowBID, (colBID, (colFactorsBuf.toArray, colPrecisionBuf.toArray))) 
      }
    }.groupByKey(partitioner).mapValues(_.toArray)
    
    optimizerType match {
      case ALS => colBlockedMessages.join(rowInLinks).mapValues{ 
        case (colMessages, rowInLink) => 
          ALS(colMessages, rowInLink, numFactors, regPara, weightedReg)
      }
      case CD => colBlockedMessages.join(rowInLinks.join(rowBlockedParas)).mapValues{
        case (colMessages, (rowInLink, rowBlockedPara)) => 
          CD(colMessages, rowBlockedPara, rowInLink, numFactors, regPara, 
            isVB, weightedReg)
      }
      case _ => {System.err.print("Only supports ALS and CD"); System.exit(-1); null}
    }
  }
  
  def ALS(colMessages: Seq[(Int, (Array[Array[Float]], Array[Array[Float]]))], 
      rowInLink: InLinkBlock, numFactors: Int, regPara: Float, weightedReg: Boolean)
    : (Array[Array[Float]], Array[Array[Float]]) = {
    // Sort the incoming block para messages by block ID and make them an array
    val colBlockParas = colMessages.sortBy(_._1).map(_._2).toArray
    val numBlocks = colBlockParas.length
    val numRows = rowInLink.elementIds.length
    
    // We'll sum up the XtXes using vectors that represent only the 
    // lower-triangular part, since the matrices are symmetric
    val triangleSize = numFactors * (numFactors + 1) / 2
    val rowXtX = Array.fill(numRows)(FloatMatrix.zeros(triangleSize))
    val rowXy = Array.fill(numRows)(FloatMatrix.zeros(numFactors))
    
    // Some temp variables to avoid memory allocation
    val tempXtX = FloatMatrix.zeros(triangleSize)
    val fullXtX = FloatMatrix.zeros(numFactors, numFactors)
    
    // Number of obs per row
    val numObsPerRow = new Array[Int](numRows)
    
    // Compute the XtX and Xy values for each user by adding products 
    // it rated in each product block
    for (bid <- 0 until numBlocks) {
      val colFactors = colBlockParas(bid)._1
      for (p <- 0 until colFactors.length) {
        val colFactor = new FloatMatrix(colFactors(p))
        fillXtX(colFactor, tempXtX)
        val (rowIDs, rowValues) = rowInLink.ratingsForBlock(bid)(p)
        for (i <- 0 until rowIDs.length) {
          numObsPerRow(rowIDs(i)) += 1
          rowXtX(rowIDs(i)).addi(tempXtX)
          SimpleBlas.axpy(rowValues(i), colFactor, rowXy(rowIDs(i)))
        }
      }
    }
    //Solve the least-squares problem for each user and return the new feature vectors
    val rowFactors = rowXtX.zipWithIndex.map{ case (triangularXtX, index) =>
      // Compute the full XtX matrix from the lower-triangular part we got above
      fillFullMatrix(triangularXtX, fullXtX)
      // Add regularization
      (0 until numFactors).foreach(i => {
        if (weightedReg) fullXtX.data(i*numFactors + i) += regPara*numObsPerRow(index)
        else fullXtX.data(i*numFactors + i) += regPara
      })
      // Solve the resulting matrix, which is symmetric and positive-definite
      Solve.solvePositive(fullXtX, rowXy(index)).data
    }
    (rowFactors, null)
  }
  
  def CD(colMessages: Seq[(Int, (Array[Array[Float]], Array[Array[Float]]))], 
      rowBlockParas: (Array[Array[Float]], Array[Array[Float]]),
      rowInLink: InLinkBlock, numFactors: Int, regPara: Float, 
      isVB: Boolean, weightedReg: Boolean)
    : (Array[Array[Float]], Array[Array[Float]]) = {
    // Sort the incoming block factor messages by block ID and make them an array
    val colBlockParas = colMessages.sortBy(_._1).map(_._2).toArray
    val numBlocks = colBlockParas.length
    val numRows = rowInLink.elementIds.length
    val residuals = rowInLink.ratingsForBlock.map(_.map(p => 
      new Array[Float](p._2.length)))
    val ratings = rowInLink.ratingsForBlock.map(_.map(_._2))
    val (rowFactors, rowPrecisions) = rowBlockParas
    val numObsPerRow = new Array[Int](numRows)
    // calculate the residuals
    for (bid <- 0 until numBlocks) {
      val colFactors = colBlockParas(bid)._1
      val numCols = colFactors.length
      var p = 0
      while (p < numCols) {
        val colFactor = colFactors(p)
        val (rowIDs, rowValues) = rowInLink.ratingsForBlock(bid)(p) 
        val res = residuals(bid)(p)
        var i = 0
        while (i < rowIDs.length) {
          val rowFactor = rowFactors(rowIDs(i))
          res(i) = rowValues(i)
          var k = 0
          while (k < numFactors) {
            res(i) -= rowFactor(k)*colFactor(k)
            k +=1 
          }
          numObsPerRow(rowIDs(i)) += 1
          i += 1
        }
        p += 1
      }
    }
    //coordinate descent
    val numerator = Array.fill(numRows)(0.0f)
    val denominator = 
      if (weightedReg) numObsPerRow.map(_*regPara) 
      else Array.fill(numRows)(regPara)
    for (k <- 0 until numFactors) {
      for (bid <- 0 until numBlocks) {
        val colFactors = colBlockParas(bid)._1
        val colPrecisions = colBlockParas(bid)._2
        var p = 0
        while (p < colFactors.length) {
          val colFactor = colFactors(p)
          val colPrecision = if (isVB) colPrecisions(p) else null
          val rowIDs = rowInLink.ratingsForBlock(bid)(p)._1
          val res = residuals(bid)(p)
          var i = 0
          while (i < rowIDs.length) {
            val rowFactor = rowFactors(rowIDs(i))
            if (k > 0) res(i) -= rowFactor(k-1)*colFactor(k-1)
            res(i) += rowFactor(k)*colFactor(k)
            numerator(rowIDs(i)) += res(i)*colFactor(k)
            denominator(rowIDs(i)) += colFactor(k)*colFactor(k)
            if (isVB) denominator(rowIDs(i)) += 1/colPrecision(k)
            i += 1
          }
          p += 1
        }
      }
      var n = 0
      while (n < numRows) {
        rowFactors(n)(k) = numerator(n)/denominator(n)
        if (isVB) rowPrecisions(n)(k) = denominator(n)
        numerator(n) = 0
        denominator(n) = if (weightedReg) numObsPerRow(n)*regPara else regPara
        n += 1
      }
    }
    if (isVB) (rowFactors, rowPrecisions)
    else (rowFactors, null)
  }
  
  def hash(x: Int): Int = {
    val r = x ^ (x >>> 20) ^ (x >>> 12)
    r ^ (r >>> 7) ^ (r >>> 4)
  }
  
  private def fillXtX(x: FloatMatrix, xtxDest: FloatMatrix) {
    var i = 0
    var pos = 0
    while (i < x.length) {
      var j = 0
      while (j <= i) {
        xtxDest.data(pos) = x.data(i) * x.data(j)
        pos += 1
        j += 1
      }
      i += 1
    }
  }
  
  private def fillFullMatrix(triangularMatrix: FloatMatrix, destMatrix: FloatMatrix) {
    val rank = destMatrix.rows
    var i = 0
    var pos = 0
    while (i < rank) {
      var j = 0
      while (j <= i) {
        destMatrix.data(i*rank + j) = triangularMatrix.data(pos)
        destMatrix.data(j*rank + i) = triangularMatrix.data(pos)
        pos += 1
        j += 1
      }
      i += 1
    }
  }
  def unblockFactors(
      blockedParas: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))], 
      outLinks: RDD[(Int, OutLinkBlock)]): RDD[(Int, Array[Float])] = {
    blockedParas.join(outLinks).flatMap{ 
      case (b, (paras, outLinkBlock)) => {
        for (i <- 0 until paras._1.length) 
          yield (outLinkBlock.elementIds(i), paras._1(i))
      }
    }
  }
  
  def initVBParas(inLinks: RDD[(Int, InLinkBlock)], numFactors:Int, reg: Float, 
      seed: Int): RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))] = {
    
    inLinks.mapPartitionsWithIndex{ (index, itr) =>
      val rand = new Random(hash(seed ^ index))
      itr.map{ case (bid, inLinkBlock) =>
        val length = inLinkBlock.elementIds.length
        val factors = Array.fill(length, numFactors)(0.1f*(rand.nextFloat))
        val numObs = new Array[Int](length)
        inLinkBlock.ratingsForBlock.foreach(_.foreach(_._1.foreach(numObs(_) += 1)))
        val precisions = Array.tabulate(length, numFactors)((i, j) => reg*numObs(i))
        (bid, (factors, precisions))
      }
    }
  }
  
  def initParas(outLinks: RDD[(Int, OutLinkBlock)], numFactors:Int, seed: Int)
    : RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))] = {
    outLinks.mapPartitionsWithIndex { (index, itr) =>
      val rand = new Random(hash(seed ^ index))
      itr.map { case (x, y) =>
        (x, (
          y.elementIds.map(_ =>(Array.fill(numFactors)(0.1f*(rand.nextFloat)))),
          null
          )
        )
      }
    }
  }
}