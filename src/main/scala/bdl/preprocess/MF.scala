package preprocess

import java.io._
import java.text.DecimalFormat

import scala.collection.mutable.{HashMap, HashSet, ArrayBuffer}
import scala.io._
import scala.util.Random

import org.apache.spark.{storage, Partitioner, SparkContext, rdd, broadcast}
import org.apache.spark.SparkContext._
import org.apache.hadoop.io.NullWritable

import utilities._

//preprocess for the matrix factorization problems
object MF {

  def parseLine(line: String, sep: String) : Record = {
    val tokens = line.split(sep)
    assert(tokens.length == 3, "unexpected record: " + line)
    new Record(tokens(0).toInt, tokens(1).toInt, tokens(2).toFloat)
  }
  
  def parseLine(line: String, sep: String, mean: Float, scale: Float) : Record = {
    val tokens = line.split(sep)
    assert(tokens.length == 3, "unexpected record: " + line)
    new Record(tokens(0).toInt, tokens(1).toInt, (tokens(2).toFloat-mean)/scale)
  }
  
  def parseLine(line: String) : Array[Record] = {
    var list = Nil
    var pair = line.split('\t')
    val rowInx = pair(0).toInt
    val ratings = pair(1).split(' ')
    for (record <- ratings) 
      yield {
        pair = record.split(":")
        new Record(rowInx, pair(0).toInt, pair(1).toFloat)
      }
  }
  
  def parseLine(line: String, mean: Float, scale: Float) : Array[Record] = {
    var list = Nil
    var pair = line.split('\t')
    val rowInx = pair(0).toInt
    val ratings = pair(1).split(' ')
    for (record <- ratings) 
      yield {
        pair = record.split(":")
        new Record(rowInx, pair(0).toInt, (pair(1).toFloat-mean)/scale)
      }
  }
  
  def getPartitionID(record: Record, numRowBlocks: Int, numColBlocks: Int) 
    : Int = (record.rowIdx%numRowBlocks)*numColBlocks + record.colIdx%numColBlocks
    
  def getPartitionMap(numRows: Int, numRowBlocks: Int, seed: Long) 
    : Array[Byte] = {
    val partitionMap = Array.ofDim[Byte](numRows)
    val random = new Random(seed)
    val stats = Array.ofDim[Int](numRowBlocks)
    var r = 0
    while (r < numRows) {
      val id = (random.nextInt(numRowBlocks)).toByte
      stats(id) += 1
      partitionMap(r) = id
      r += 1
    }
    var i = 0; while (i<stats.length) { 
      println("block " + i + " has " + stats(i) + " elements")
      i += 1
    }
    partitionMap
  }
  
  def getPartitionedData(records: rdd.RDD[Record], numColBlocks: Int,
      rowPartitionMap: broadcast.Broadcast[Array[Byte]], 
      colPartitionMap: broadcast.Broadcast[Array[Byte]],
      partitioner: Partitioner) : rdd.RDD[(Int, SparseMatrix)] = {
     
    //on the choice between ArrayBuffer and ListBuffer:
    //altough ArrayBuffer consumes more memory in this case, it has lower cache-miss
    //rate, which is deseriable and important
    def createCombiner(record: Record) = ArrayBuffer[Record](record)
    def mergeValue(buf: ArrayBuffer[Record], record: Record) = buf += record
    
    records.mapPartitions(_.map(record => {
      val rowPID = rowPartitionMap.value(record.rowIdx)
      val colPID = colPartitionMap.value(record.colIdx)
      (rowPID*numColBlocks + colPID, record)
    }))
    .combineByKey[ArrayBuffer[Record]](
      createCombiner _, mergeValue _, null, partitioner, mapSideCombine=false)
    .mapValues(buf => SparseMatrix(buf.toArray))
  }
  
  def getPartitionedData(records: rdd.RDD[Record], numColBlocks: Int,
      rowPartitionMapBC: broadcast.Broadcast[Array[Byte]], 
      colPartitionMapBC: broadcast.Broadcast[Array[Byte]], numReducers: Int)
    : rdd.RDD[(Int, SparseMatrix)] = {
    records.mapPartitions(_.map(record => {
      val rowPartitionMap = rowPartitionMapBC.value
      val colPartitionMap = colPartitionMapBC.value
      val rowPID = rowPartitionMap(record.rowIdx)
      val colPID = colPartitionMap(record.colIdx)
      (rowPID*numColBlocks + colPID, record)
    })).groupByKey(numReducers).mapValues(buf => SparseMatrix(buf.toArray))
  }
  
  def getPartitionedData(
      data: rdd.RDD[((Int, Int), (Array[Int], Array[Int], Array[Float]))],
      numRowBlocks: Int, numColBlocks: Int, partitioner: Partitioner) = {
    val (maxRowID, maxColID) = data.map(_._1)
      .reduce((p1, p2) => (math.max(p1._1, p2._1), math.max(p1._2, p2._2)))
    val rowBlockSize = (maxRowID+1)/numRowBlocks
    val colBlockSize = (maxColID+1)/numColBlocks
    data.map(pair=>{
      val rowBID = math.min(pair._1._1/rowBlockSize, numRowBlocks-1)
      val colBID = math.min(pair._1._2/rowBlockSize, numColBlocks-1)
      (rowBID*numColBlocks + colBID, pair._2)
    }).groupByKey(partitioner).mapValues(seq => SparseMatrix(seq.toArray))
    .mapValues{
      case (row_idx, col_idx, value_r) => SparseMatrix(row_idx, col_idx, value_r)
    }
  }
  
  def preprocessNetflix(
      inputDir: String, probePath: String, trainPath: String, testPath: String)= {
    
    def readProbe(probePath: String) = {
      val dict = new HashMap[String, HashSet[String]]
      var set: HashSet[String] = null
      for (line <- Source.fromFile(probePath).getLines) {
        if (line.contains(':')) {
          set = new HashSet[String]
          dict.put(line.substring(0, line.length-1), set)
        }
        else
          set.add(line)
      }
      dict
    }
    
    val dict = readProbe(probePath)
    val trainBW = new BufferedWriter(new FileWriter(new File(trainPath)))
    val testBW = new BufferedWriter(new FileWriter(new File(testPath)))
    val sbTrain: StringBuilder = new StringBuilder()
    val sbTest: StringBuilder = new StringBuilder()
    var numtrain = 0; var numtest = 0
    for (file <- new java.io.File(inputDir).listFiles) {
      val lines = Source.fromFile(file).getLines.toList
      val rowIdx = lines(0).substring(0, lines(0).length-1)
      val set = dict.getOrElse(rowIdx, null)
      var firstTrain = true; var firstTest = true
      for (line <- lines.tail) {
        val tokens = line.split(",")
        assert(tokens.length == 3, "unexpected line: " + line)
        if (set!=null && set.contains(tokens(0))) {
          numtest += 1
          if (firstTest) {
            sbTest.append(lines(0).substring(0, lines(0).length-1))
            sbTest.append('\t')
            firstTest = false
          }
          sbTest.append(tokens(0)+':'+tokens(1)+' ')
        }
        else {
          numtrain += 1
          if (firstTrain) {
            sbTrain.append(lines(0).substring(0, lines(0).length-1))
            sbTrain.append('\t')
            firstTrain = false
          }
          sbTrain.append(tokens(0)+':'+tokens(1)+' ')
        }
      }
      if (sbTrain.length > 0) { trainBW.write(sbTrain.toString + '\n'); sbTrain.clear }
      if (sbTest.length > 0) { testBW.write(sbTest.toString + '\n'); sbTest.clear }
    }
    println("num of train: " + numtrain + "\nnum of test: " + numtest)
    trainBW.close()
    testBW.close()
  }
  
  def generateDict(inputPath: String, thre: Int) 
    : scala.collection.Map[String, Int]= {
    val dict = new HashMap[String, Int]
    for (line <- Source.fromFile(inputPath).getLines) {
      val tokens = line.split(' ')
      for (token <- tokens) dict.put(token, dict.getOrElse(token, 0)+1)
    }
    dict.filter(pair => pair._2>thre).keySet.zipWithIndex.toMap
  }
  
  def preprocessNYTimes(
      inputPath: String, trainPath: String, testPath: String, ratio: Double) = {
    
    val doc = new HashMap[Int, Int]
    val trainBW = new BufferedWriter(new FileWriter(new File(trainPath)))
    val testBW = new BufferedWriter(new FileWriter(new File(testPath)))
    val sbTrain: StringBuilder = new StringBuilder()
    val sbTest: StringBuilder = new StringBuilder()
    val numberFormat = new DecimalFormat("0.00");
    val dict = generateDict(inputPath, 5)
    val numWords = dict.size
    println("dict size: " + numWords)
    var numDocs = 0
    var numTrain = 0L
    var numTest = 0
    for (line <- Source.fromFile(inputPath).getLines) {
      val tokens = line.split(' ')
      for (token <- tokens) {
        dict.get(token) match {
          case Some(id) => doc.put(id, doc.getOrElse(id, 0)+1)
          case None => None
        }
      }
      doc.map(pair => {
        if (Random.nextDouble < ratio) 
          sbTrain.append(pair._1 + ":" + numberFormat.format(math.log(pair._2+1))+" ") 
        else 
          sbTest.append(pair._1 + ":" + numberFormat.format(math.log(pair._2+1))+" ")
      })
      doc.clear
      if (sbTrain.length > 0) {
        trainBW.write(numDocs + "\t" + sbTrain.toString)
        trainBW.newLine()
        numTrain += sbTrain.length
        sbTrain.clear
      }
      if (sbTest.length > 0) {
        testBW.write(numDocs + "\t" + sbTest.toString)
        testBW.newLine()
        numTest += sbTest.length
        sbTest.clear
      }
      numDocs += 1
    }
    println("num of docs: " + numDocs + "\nnum of words: " + numWords)
    println("num of train: " + numTrain + "\nnum of test: " + numTest)
    trainBW.close()
    testBW.close()
  }
  
  def preprocessWiki(
      inputPath: String, trainDir: String, testPath: String, ratio: Double) {
    
    val sbTrain: StringBuilder = new StringBuilder()
    val sbTest: StringBuilder = new StringBuilder()
    var nnz = 0
    var fileCount = 0 
    var trainBW = new BufferedWriter(new FileWriter(
        new File(trainDir + "/" + fileCount)))
    val testBW = new BufferedWriter(new FileWriter(new File(testPath)))
    val random = new Random
    for (line <- Source.fromFile(inputPath).getLines) {
      if (nnz%5138830 == 0)  {
        trainBW.close()
        fileCount += 1
        trainBW = new BufferedWriter(new FileWriter(
            new File(trainDir + "/" + fileCount)))
      }
      if (nnz > 1) {  
        //skip the first two lines
        if (random.nextFloat < ratio) {
          trainBW.write(line)
          trainBW.newLine()
        }
        else {
          testBW.write(line)
          testBW.newLine()
        }
      }
      nnz += 1
    }
    trainBW.close
    testBW.close
  }
  
  def preprocessMovieLens(sc: SparkContext, inputDir: String, outputDir: String) = {
    
    val STORAGE_LEVEL = storage.StorageLevel.MEMORY_AND_DISK_SER
    val data = sc.textFile(inputDir).map(line => {
      val tokens = line.split("::")
      assert(tokens.length == 4)
      val rowIdx = tokens(0).toInt
      val colIdx = tokens(1).toInt
      val value = tokens(2).toFloat
      new Record(rowIdx, colIdx, value)
    }).persist(STORAGE_LEVEL)
    val (numRows, numCols, nnz, sum) = data.map(record => 
      (record.rowIdx, record.colIdx, 1, record.value)
    ).reduce((tuple1, tuple2) => (math.max(tuple1._1, tuple2._1), 
        math.max(tuple1._2, tuple2._2), tuple1._3+tuple2._3, tuple1._4+tuple2._4))
    val mean = sum/nnz
    println("numRows: "+(numRows+1) + "\tnumCols: " + (numCols+1) + "\tmean: " + mean)
    data.map(record => (NullWritable.get, record)).saveAsSequenceFile(outputDir)
  }
}