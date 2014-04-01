package mf

import java.io._

import scala.util._
import scala.collection.mutable.{HashSet, ListBuffer, ArrayBuffer}
import scala.math._

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, HashPartitioner, storage}
import org.apache.spark.SparkContext._
import org.apache.spark.serializer.KryoRegistrator
import org.apache.commons.cli._
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapred.SequenceFileOutputFormat

import utilities.SparseMatrix
import utilities.Settings._
import utilities.Record
import utilities.Vector
import preprocess.MF._

//generate synthetic dataset

object GenSynMat {
  
  val numFactors = 10
  val numCores = 1
  val numReducers = numCores
  val OUTPUT_DIR = "output/"
  val TRAINING_DIR = OUTPUT_DIR + "train" + PATH_SEPERATOR
  val TESTING_DIR = OUTPUT_DIR + "test" + PATH_SEPERATOR
  val numRows = 5000
  val numCols = 5000
  val numRowBlocks= 1
  val numColBlocks = 2
  val numSlices = Array[(Int, Int)]((4,4))
  val MODE = "local[" + numCores + "]"
  val JARS = Seq("sparkproject.jar")
  val lambda = 100f
  val sparsity = 0.1
  val train_ratio = 0.9
  
  def main(args : Array[String]) {
    
    val currentTime = System.currentTimeMillis()
    
    val options = new Options()
    options.addOption(HELP, false, "print the help message")
    options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
    options.addOption(OUTPUT_OPTION, true, "output path/directory")
    options.addOption(MEAN_OPTION, true, "mean option for input values")
    options.addOption(NUM_CORES_OPTION, true, "number of cores to use")
    options.addOption(NUM_REDUCERS_OPTION, true, "number of reducers to use")
    options.addOption(NUM_SLICES_OPTION, true, "number of slices of the data")
    options.addOption(NUM_COL_BLOCKS_OPTION, true, "number of column blocks")
    options.addOption(NUM_ROW_BLOCKS_OPTION, true, "number of row blocks")
    options.addOption(LAMBDA_INIT_OPTION, true, "set lambda value")
    options.addOption(NUM_LATENT_FACTORS_OPTION, true, "number of latent factors")
    options.addOption(JAR_OPTION, true, "the path to find jar file")
    options.addOption(TMP_DIR_OPTION, true, 
        "local dir for tmp files, including map output files and RDDs stored on disk")
    options.addOption(MEM_OPTION, true, 
        "amount of memory to use per executor process")
    options.addOption(NUM_ROWS_OPTION, true, "number of rows")
    options.addOption(NUM_COLS_OPTION, true, "number of cols")
    options.addOption(TRAINING_RATIO_OPTION, true, "training data ratio")
    options.addOption(SPARSITY_LEVEL_OPTION, true, "sparsity level")
    options.addOption(NUM_OUTPUT_BLOCKS_OPTION, true, "num of output blocks to HDFS")
    
    val parser = new GnuParser()
    val formatter = new HelpFormatter()
    val line = parser.parse(options, args)
    if (line.hasOption(HELP) || args.length == 0) {
      formatter.printHelp("Help", options)
      System.exit(0)
    }
    assert(line.hasOption(RUNNING_MODE_OPTION), "running mode not specified")
    assert(line.hasOption(OUTPUT_OPTION), "output path/directory not specified")
    assert(line.hasOption(JAR_OPTION), "jar file path not specified")
    
    val MODE = line.getOptionValue(RUNNING_MODE_OPTION)
    val JARS = Seq(line.getOptionValue(JAR_OPTION))
    val OUTPUT_DIR = 
      if (line.hasOption(OUTPUT_OPTION))
        if (!line.getOptionValue(OUTPUT_OPTION).endsWith(PATH_SEPERATOR))
          line.getOptionValue(OUTPUT_OPTION) + PATH_SEPERATOR
        else
          line.getOptionValue(OUTPUT_OPTION)
      else
        "output" + PATH_SEPERATOR
    val numRows = 
      if (line.hasOption(NUM_ROWS_OPTION))
        line.getOptionValue(NUM_ROWS_OPTION).toInt
      else 1000000
    val numCols = 
      if (line.hasOption(NUM_COLS_OPTION))
        line.getOptionValue(NUM_COLS_OPTION).toInt
      else 1000000
    val numRowBlocks = 
      if (line.hasOption(NUM_ROW_BLOCKS_OPTION)) 
        line.getOptionValue(NUM_ROW_BLOCKS_OPTION).toInt
      else 1
    val numColBlocks = 
      if (line.hasOption(NUM_COL_BLOCKS_OPTION)) 
        line.getOptionValue(NUM_COL_BLOCKS_OPTION).toInt
      else 1
    val numOutputBlocks = 
      if (line.hasOption(NUM_OUTPUT_BLOCKS_OPTION))
        line.getOptionValue(NUM_OUTPUT_BLOCKS_OPTION).toInt
      else numRowBlocks*numColBlocks
    val lambda = 
      if (line.hasOption(LAMBDA_INIT_OPTION))
        line.getOptionValue(LAMBDA_INIT_OPTION).toFloat
      else 1f
    val numFactors = 
      if (line.hasOption(NUM_LATENT_FACTORS_OPTION)) 
        line.getOptionValue(NUM_LATENT_FACTORS_OPTION).toInt
      else 20
    val train_ratio = 
      if (line.hasOption(TRAINING_RATIO_OPTION))
        line.getOptionValue(TRAINING_RATIO_OPTION).toFloat
      else 0.8f
    val sparsity = 
      if (line.hasOption(SPARSITY_LEVEL_OPTION))
        line.getOptionValue(SPARSITY_LEVEL_OPTION).toFloat
      else 0.001f
    val numSlices =
      if (line.hasOption(NUM_SLICES_OPTION)) {
        line.getOptionValue(NUM_SLICES_OPTION).split(",").map(token => {
          val pair = token.split(":") 
          (pair(0).toInt, pair(1).toInt)
        }) 
      }
      else new Array[(Int, Int)](0)
    if (line.hasOption(TMP_DIR_OPTION)) {
      System.setProperty("spark.local.dir", line.getOptionValue(TMP_DIR_OPTION))
    }
    if (line.hasOption(MEM_OPTION)) {
      System.setProperty("spark.executor.memory", line.getOptionValue(MEM_OPTION))
    }
    if (line.hasOption(NUM_REDUCERS_OPTION) || line.hasOption(NUM_CORES_OPTION)) {
      if (line.hasOption(NUM_REDUCERS_OPTION)) {
        System.setProperty("spark.default.parallelism", 
          line.getOptionValue(NUM_REDUCERS_OPTION))
      }
      else {
        System.setProperty("spark.default.parallelism", 
          line.getOptionValue(NUM_CORES_OPTION))
      }
    }
    
    System.setProperty("spark.serializer", 
        "org.apache.spark.serializer.KryoSerializer")
    System.setProperty("spark.kryo.registrator", "utilities.Registrator")
    System.setProperty("spark.kryo.referenceTracking", "false")
//    System.setProperty("spark.kryoserializer.buffer.mb", "1024")
    System.setProperty("spark.worker.timeout", "3600")
    System.setProperty("spark.storage.blockManagerSlaveTimeoutMs", "8000000")
    System.setProperty("spark.storage.blockManagerHeartBeatMs", "8000000")
    
    val StorageLevel = storage.StorageLevel.MEMORY_ONLY_SER
    val JOB_NAME = "Syn" + "_M_" + numRows + "_N_" + numCols + "_K_" + numFactors + 
      "_spa_" + sparsity + "_tr_" + train_ratio + "_lam_" + lambda
    val sc = new SparkContext(MODE, JOB_NAME, System.getenv("SPARK_HOME"), JARS)
    
    val sigma = 1/math.sqrt(math.sqrt(numFactors))
    
    def hash(x: Int): Int = {
      val r = x ^ (x >>> 20) ^ (x >>> 12)
      r ^ (r >>> 7) ^ (r >>> 4)
    }
    
    //randomly partition the sparse matrix into blocks, 
    //and generate synthetic data for each block
    val seedRow = hash(numRows)
    val rowBlockMapBC = sc.broadcast(getPartitionMap(numRows, numRowBlocks, seedRow))
    val seedCol = hash(numCols+numRows)
    val colBlockMapBC = sc.broadcast(getPartitionMap(numCols, numColBlocks, seedCol))
    
    val rowBlocks = sc.parallelize((0 until numRowBlocks), numRowBlocks).map(bid => {
      val rowIndices = rowBlockMapBC.value.zipWithIndex.filter(_._1==bid).map(_._2)
      val gaussian = new Random
      val rowFactors = rowIndices.map(rowIdx => {
        gaussian.setSeed(hash(rowIdx))
        Array.fill(numFactors)((gaussian.nextGaussian*sigma).toFloat)
      })
      (bid, rowFactors)
    }).persist(StorageLevel)
    println("number of row blocks: " + rowBlocks.count)
    val colBlocks = sc.parallelize((0 until numColBlocks), numColBlocks).map(bid => {
      val colIndices = colBlockMapBC.value.zipWithIndex.filter(_._1==bid).map(_._2)
      val gaussian = new Random
      val colFactors = colIndices.map(colIdx => {
        gaussian.setSeed(hash(colIdx+numRows))
        Array.fill(numFactors)((gaussian.nextGaussian*sigma).toFloat)
      })
      (bid, colFactors)
    }).persist(StorageLevel)
    println("number of col blocks: " + colBlocks.count)
    
    def drawColIdx(random: Random, map: HashSet[Int], numCols: Int) : Int = {
      val n = random.nextInt(numCols)
      if (map.contains(n)) drawColIdx(random, map, numCols)
      else {
        map.add(n)
        n
      }
    }
    
    val numTrain = sc.accumulator(0L)
    val numTest = sc.accumulator(0L)
    val secondMoment = sc.accumulator(0.0)
    val firstMoment = sc.accumulator(0.0)
    
    val syntheticData = rowBlocks.cartesian(colBlocks).map{
      case ((rowBID, rowFactors), (colBID, colFactors)) => {
        val time = System.currentTimeMillis()
        val rowIndices = 
          rowBlockMapBC.value.zipWithIndex.filter(_._1==rowBID).map(_._2)
        val colIndices = 
          colBlockMapBC.value.zipWithIndex.filter(_._1==colBID).map(_._2)
        val bid = rowBID*numColBlocks + colBID
        val numRows = rowIndices.length
        val numCols = colIndices.length
        val seed = hash(bid)
        val random_int = new Random(seed)
        val gaussian = new Random(seed)
        val colIdxSet = new HashSet[Int]()
        //pre-calculate how many train/test entries are observed in each row
        val nnzTrainRows = new Array[Int](numRows)
        val nnzTestRows = new Array[Int](numRows)
        val mean = sparsity*numCols
        val variance = numCols*sparsity*(1-sparsity)
        var m = 0
        var nnzTrain = 0
        var nnzTest = 0
        while (m < numRows) {
          val nnzRow = math.min(math.max(mean + 
              gaussian.nextGaussian*math.sqrt(variance), 1), numCols).toInt
          val meanRow = train_ratio*nnzRow
          val varianceRow = train_ratio*(1-train_ratio)*nnzRow
          nnzTrainRows(m) = math.min(math.max(meanRow + 
              gaussian.nextGaussian*math.sqrt(varianceRow), 1), nnzRow-1).toInt
          nnzTrain += nnzTrainRows(m)
          nnzTestRows(m) = nnzRow - nnzTrainRows(m)
          nnzTest += nnzTestRows(m)
          m += 1
        }
        val trainingRecords = new Array[Record](nnzTrain)
        val testingRecords = new Array[Record](nnzTest)
        println("nnzTrain: " + nnzTrain + "\tnnzTest: " + nnzTest)
        numTrain += nnzTrain
        numTest += nnzTest
        var countTrain = 0
        var countTest = 0
        m = 0
        while (m < numRows) {
          val rowFactor = rowFactors(m)
          val rowIdx = rowIndices(m)
          colIdxSet.clear
          var i = 0
          val nnzRow = nnzTrainRows(m) + nnzTestRows(m)
          while(i < nnzRow) {
            val n = drawColIdx(random_int, colIdxSet, numCols)
            val colIdx = colIndices(n)
            val value = Vector(rowFactor).dot(Vector(colFactors(n)))
            firstMoment += value
            secondMoment += value*value
            if (i < nnzTrainRows(m)) {
              val noise = (gaussian.nextGaussian/math.sqrt(lambda)).toFloat
              trainingRecords(countTrain) = new Record(rowIdx, colIdx, value+noise)
              countTrain += 1
            }
            else {
              testingRecords(countTest) = new Record(rowIdx, colIdx, value)
              countTest += 1
            }
            i += 1
          }
          m += 1
        }
        println("finished training/testing records generation in " + 
          (System.currentTimeMillis() - time)*0.0001 + " seconds")
        assert(trainingRecords.length == countTrain, 
            "length: " + trainingRecords.length + "\tcountTrain " + countTrain)
        assert(testingRecords.length == countTest,
            "length: " + testingRecords.length + "\tcountTest " + countTrain)
        (trainingRecords, testingRecords)
      }
    }.cache
    val TRAINING_DIR = OUTPUT_DIR + "train_" + JOB_NAME + PATH_SEPERATOR
    val TESTING_DIR = OUTPUT_DIR + "test_" + JOB_NAME + PATH_SEPERATOR
    
    syntheticData.flatMap(_._1.map((NullWritable.get, _)))
      .saveAsSequenceFile(TRAINING_DIR)
    syntheticData.flatMap(_._2.map((NullWritable.get, _)))
      .saveAsSequenceFile(TESTING_DIR)
      
    val bwLog = new BufferedWriter(new FileWriter(new File(JOB_NAME)))
    bwLog.write(JOB_NAME+"\n")
    println("number of training samples: " + numTrain.value)
    bwLog.write("number of training samples: " + numTrain.value + "\n")
    println("number of testing samples: " + numTest.value)
    bwLog.write("number of testing samples: " + numTest.value + "\n")
    val nnz = numTrain.value+numTest.value
    println("average first moment: " + firstMoment.value/nnz)
    println("average second moment: " + secondMoment.value/nnz + "\n")
    bwLog.write("average first moment: " + firstMoment.value/nnz + "\n")
    bwLog.write("average second moment: " + secondMoment.value/nnz)  
    
    def createCombiner(record: Record) = {
      val _1 = ArrayBuffer[Int](record.rowIdx)
      val _2 = ArrayBuffer[Int](record.colIdx)
      val _3 = ArrayBuffer[Float](record.value)
      (_1, _2, _3)
    }
    def mergeValue(buf: (ArrayBuffer[Int], ArrayBuffer[Int], ArrayBuffer[Float]), 
      record: Record) = {
      buf._1 += record.rowIdx
      buf._2 += record.colIdx
      buf._3 += record.value
      buf
    }
    var s = 0
    
    while (s < numSlices.length) {
      //generating pre-partitioned sparse matrices
      println("begin to construct synthetic matrix with block size " + numSlices(s))
      val (numRowBlocks, numColBlocks) = numSlices(s)
      val outputBlockSize = numRowBlocks*numColBlocks/numOutputBlocks
      val rowBlockSize = numRows/numRowBlocks
      val colBlockSize = numCols/numColBlocks
      val trainDir = OUTPUT_DIR + "train_mat_" + "br_" + numRowBlocks + 
        "_bc_" + numColBlocks + "_" + JOB_NAME + PATH_SEPERATOR
      val part = new HashPartitioner(numRowBlocks*numColBlocks)
      syntheticData.flatMap(_._1.map(record => {
        val rowPID = math.min(record.rowIdx/rowBlockSize, numRowBlocks-1)
        val colPID = math.min(record.colIdx/colBlockSize, numColBlocks-1)
        (rowPID*numColBlocks + colPID, record)
      }))
      .combineByKey[(ArrayBuffer[Int], ArrayBuffer[Int], ArrayBuffer[Float])](
        createCombiner _, mergeValue _, null, part, false)
      .map(pair => (pair._1 / outputBlockSize,
        ((pair._1/numColBlocks, pair._1%numColBlocks), 
        (pair._2._1.toArray, pair._2._2.toArray, pair._2._3.toArray))))
      .groupByKey(numOutputBlocks).flatMap(_._2).saveAsObjectFile(trainDir)
      
      val testDir = OUTPUT_DIR + "test_mat_" + "br_" + numRowBlocks + 
        "_bc_" + numColBlocks + "_" + JOB_NAME + PATH_SEPERATOR
      syntheticData.flatMap(_._2.map(record => {
        val rowPID = math.min(record.rowIdx/rowBlockSize, numRowBlocks-1)
        val colPID = math.min(record.colIdx/colBlockSize, numColBlocks-1)
        (rowPID*numColBlocks + colPID, record)
      }))
      .combineByKey[(ArrayBuffer[Int], ArrayBuffer[Int], ArrayBuffer[Float])](
        createCombiner _, mergeValue _, null, part, false)
      .map(pair => (pair._1 / outputBlockSize, 
        ((pair._1/numColBlocks, pair._1%numColBlocks), 
          (pair._2._1.toArray, pair._2._2.toArray, pair._2._3.toArray))))
      .groupByKey(numOutputBlocks).flatMap(_._2).saveAsObjectFile(testDir)
      s += 1
    }
    
    val elpasedTime = (System.currentTimeMillis() - currentTime)*0.001
    println("Generating synthetic matrix finished in " + elpasedTime + "(s)")
    bwLog.write("Generating synthetic matrix finished in " + elpasedTime + "(s)\n")
    bwLog.close
    System.exit(0)
  }
}