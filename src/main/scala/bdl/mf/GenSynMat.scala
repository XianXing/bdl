package mf

import utilities._
import preprocess.MF._

import java.io._
import scala.util._
import scala.collection.mutable.ArrayBuffer
import scala.math._
import scala.collection.mutable.ArrayBuilder
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, HashPartitioner, storage}
import org.apache.spark.SparkContext._
import org.apache.spark.serializer.KryoRegistrator
import org.apache.commons.cli._
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapred.SequenceFileOutputFormat

//generate synthetic dataset

object GenSynMat extends Settings{
  
  val numFactors = 20
  val numCores = 2
  val numReducers = numCores
  val OUTPUT_DIR = "output/"
  val TRAINING_DIR = OUTPUT_DIR + "train" + PATH_SEPERATOR
  val TESTING_DIR = OUTPUT_DIR + "test" + PATH_SEPERATOR
  val numRows = 1000
  val numCols = 1000
  val numRowBlocks= 2
  val numColBlocks = 2
  val numSlices = numRowBlocks*numColBlocks
  val MODE = "local[" + numCores + "]"
  val JARS = Seq("sparkproject.jar")
  val gamma = 1f
  val lambda = 10f
  val tmpDir = "tmp"
  val multi_thread = false
  val sparsity = 0.01
  val train_ratio = 0.9
  
  def main(args : Array[String]) {
    
    val currentTime = System.currentTimeMillis();
    
//    val options = new Options()
//    options.addOption(HELP, false, "print the help message")
//    options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
//    options.addOption(OUTPUT_OPTION, true, "output path/directory")
//    options.addOption(MEAN_OPTION, true, "mean option for input values")
//    options.addOption(NUM_CORES_OPTION, true, "number of cores to use")
//    options.addOption(NUM_REDUCERS_OPTION, true, "number of reducers to use")
//    options.addOption(NUM_SLICES_OPTION, true, "number of slices of the data")
//    options.addOption(NUM_COL_BLOCKS_OPTION, true, "number of column blocks")
//    options.addOption(NUM_ROW_BLOCKS_OPTION, true, "number of row blocks")
//    options.addOption(LAMBDA_INIT_OPTION, true, "set lambda value")
//    options.addOption(GAMMA_INIT_OPTION, true, "set gamma value")
//    options.addOption(NUM_LATENT_FACTORS_OPTION, true, "number of latent factors")
//    options.addOption(JAR_OPTION, true, "the path to find jar file")
//    options.addOption(TMP_DIR_OPTION, true, "the local dir for tmp files")
//    options.addOption(NUM_ROWS_OPTION, true, "number of rows")
//    options.addOption(NUM_COLS_OPTION, true, "number of cols")
//    options.addOption(TRAINING_RATIO_OPTION, true, "training data ratio")
//    options.addOption(SPARSITY_LEVEL_OPTION, true, "sparsity level")
//    options.addOption(MULTI_THREAD_OPTION, false, 
//        "using multi-threads for each partition")
//    
//    val parser = new GnuParser();
//    val formatter = new HelpFormatter();
//    val line = parser.parse(options, args);
//    if (line.hasOption(HELP) || args.length == 0) {
//      formatter.printHelp("Help", options);
//      System.exit(0);
//    }
//    assert(line.hasOption(RUNNING_MODE_OPTION), "running mode not specified")
//    assert(line.hasOption(OUTPUT_OPTION), "output path/directory not specified")
//    assert(line.hasOption(JAR_OPTION), "jar file path not specified")
//    
//    val MODE = line.getOptionValue(RUNNING_MODE_OPTION)
//    val JARS = Seq(line.getOptionValue(JAR_OPTION))
//    val OUTPUT_DIR = 
//      if (line.hasOption(OUTPUT_OPTION))
//        if (!line.getOptionValue(OUTPUT_OPTION).endsWith(PATH_SEPERATOR))
//          line.getOptionValue(OUTPUT_OPTION) + PATH_SEPERATOR
//        else
//          line.getOptionValue(OUTPUT_OPTION)
//      else
//        "output" + PATH_SEPERATOR
//    val numRows = 
//      if (line.hasOption(NUM_ROWS_OPTION))
//        line.getOptionValue(NUM_ROWS_OPTION).toInt
//      else 10000000
//    val numCols = 
//      if (line.hasOption(NUM_COLS_OPTION))
//        line.getOptionValue(NUM_COLS_OPTION).toInt
//      else 10000000
//    val numReducers =
//      if (line.hasOption(NUM_REDUCERS_OPTION))
//        line.getOptionValue(NUM_REDUCERS_OPTION).toInt
//      else 8
//    val numRowBlocks = 
//      if (line.hasOption(NUM_ROW_BLOCKS_OPTION)) 
//        line.getOptionValue(NUM_ROW_BLOCKS_OPTION).toInt
//      else 1
//    val numColBlocks = 
//      if (line.hasOption(NUM_COL_BLOCKS_OPTION)) 
//        line.getOptionValue(NUM_COL_BLOCKS_OPTION).toInt
//      else 1
//    val NUM_PARTITIONS = numRowBlocks*numColBlocks
//    val gamma = 
//      if (line.hasOption(GAMMA_INIT_OPTION)) 
//        line.getOptionValue(GAMMA_INIT_OPTION).toFloat
//      else 1f
//    val lambda = 
//      if (line.hasOption(LAMBDA_INIT_OPTION))
//        line.getOptionValue(LAMBDA_INIT_OPTION).toFloat
//      else 1f
//    val numFactors = 
//      if (line.hasOption(NUM_LATENT_FACTORS_OPTION)) 
//        line.getOptionValue(NUM_LATENT_FACTORS_OPTION).toInt
//      else 20
//    val numSlices =
//      if (line.hasOption(NUM_SLICES_OPTION))
//        math.min(line.getOptionValue(NUM_SLICES_OPTION).toInt, 
//            numRowBlocks*numColBlocks)
//      else numRowBlocks*numColBlocks
//    val tmpDir = if (line.hasOption(TMP_DIR_OPTION))
//      line.getOptionValue(TMP_DIR_OPTION)
//      else "tmp"
//    val multi_thread = line.hasOption(MULTI_THREAD_OPTION)
//    val train_ratio = 
//      if (line.hasOption(TRAINING_RATIO_OPTION))
//        line.getOptionValue(TRAINING_RATIO_OPTION).toFloat
//      else 0.8f
//    val sparsity = 
//      if (line.hasOption(SPARSITY_LEVEL_OPTION))
//        line.getOptionValue(SPARSITY_LEVEL_OPTION).toFloat
//      else 0.001f
    
    System.setProperty("spark.local.dir", tmpDir)
    System.setProperty("spark.default.parallelism", numReducers.toString)
    System.setProperty("spark.serializer", 
        "org.apache.spark.serializer.KryoSerializer")
    System.setProperty("spark.kryoserializer.buffer.mb", "100")
    System.setProperty("spark.kryo.registrator", "utilities.Registrator")
    System.setProperty("spark.kryo.referenceTracking", "false")
    System.setProperty("spark.storage.memoryFraction", "0.5")
    
//    System.setProperty("spark.worker.timeout", "3600")
//    System.setProperty("spark.storage.blockManagerSlaveTimeoutMs", "8000000")
    
    val StorageLevel = storage.StorageLevel.MEMORY_AND_DISK_SER
    val JOB_NAME = "Syn" + "_M_" + numRows + "_N_" + numCols + "_K_" + numFactors + 
      "_spa_" + sparsity + "_tr_" + train_ratio + "_ga_" + gamma + "_lam_" + lambda
    val sc = new SparkContext(MODE, JOB_NAME, System.getenv("SPARK_HOME"), JARS)  
      
    val rowBlocks = sc.parallelize((0 until numRowBlocks), numRowBlocks).map(pid => {
      val blockSize = numRows/numRowBlocks
      val random = new Random
      val rowFactors = Array.tabulate(blockSize)(i => 
        Array.fill(numFactors)((random.nextGaussian/math.sqrt(gamma)).toFloat))
      (pid, rowFactors)
    }).persist(StorageLevel)
    println("number of row blocks: " + rowBlocks.count)
    val colBlocks = sc.parallelize((0 until numColBlocks), numColBlocks).map(pid => {
      val blockSize = numCols/numColBlocks
      val random = new Random
      val colFactors = Array.tabulate(blockSize)(i => 
        Array.fill(numFactors)((random.nextGaussian/math.sqrt(gamma)).toFloat))
      (pid, colFactors)
    }).persist(StorageLevel)
    println("number of col blocks: " + colBlocks.count)
    
    val syntheticData = rowBlocks.cartesian(colBlocks).map{
      case ((rowBID, rowFactors), (colBID, colFactors)) => {
        val bid = rowBID*numColBlocks + colBID
        val numRows = rowFactors.length
        val numCols = colFactors.length
        val trainingRecords = new ArrayBuffer[Record]
        val testingRecords = new ArrayBuffer[Record]
        val random_spa = new Random
        val random_train = new Random
        val random_normal = new Random
        for (m <- 0 until numRows; n <- 0 until numCols) {
          if (random_spa.nextDouble < sparsity) {
            val rowFactor = rowFactors(m)
            val colFactor = colFactors(n)
            val rowIdx = m*numRowBlocks+rowBID
            val colIdx = n*numColBlocks+colBID
            val value = rowFactor.view.zip(colFactor).map{case(r, c)=>r*c}.sum
            if (random_train.nextDouble < train_ratio) {
              val noise = (random_normal.nextGaussian/math.sqrt(lambda)).toFloat
              trainingRecords += new Record(rowIdx, colIdx, value+noise)
            }
            else
              testingRecords += new Record(rowIdx, colIdx, value)
          }
        }
//        val trainingData = SparseMatrix(trainingRecords.toArray)
//        val testingData = SparseMatrix(testingRecords.toArray)
//        (bid, trainingData, testingData)
        (trainingRecords.toArray, testingRecords.toArray)
      }
    }.persist(StorageLevel)
    val numTrainingSamples = syntheticData.map(_._1.length).sum
    val numTestingSamples = syntheticData.map(_._2.length).sum
    val TRAINING_DIR = OUTPUT_DIR + "train_" +JOB_NAME + PATH_SEPERATOR
    val TESTING_DIR = OUTPUT_DIR + "test_" + JOB_NAME + PATH_SEPERATOR
    syntheticData.flatMap(_._1).map((NullWritable.get, _))
      .saveAsSequenceFile(TRAINING_DIR)
    syntheticData.flatMap(_._2).map((NullWritable.get, _))
      .saveAsSequenceFile(TESTING_DIR)
//    syntheticData.flatMap(_._2).map((NullWritable.get, _))
//      .saveAsHadoopFile[SequenceFileOutputFormat[NullWritable, Record]](TRAINING_DIR)
      
    val bwLog = new BufferedWriter(new FileWriter(new File("log")))
    bwLog.write(JOB_NAME+"\n")
    val elpasedTime = (System.currentTimeMillis() - currentTime)*0.001
    println("number of training samples: " + numTrainingSamples)
    bwLog.write("number of training samples: " + numTrainingSamples + "\n")
    println("number of testing samples: " + numTestingSamples)
    bwLog.write("number of testing samples: " + numTestingSamples + "\n")
    println("generating synthetic matrix finished in " + elpasedTime + "(s)")
    bwLog.write("generating synthetic matrix finished in " + elpasedTime + "(s)\n")
    bwLog.close
    System.exit(0)
  }
}