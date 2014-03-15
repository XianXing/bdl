package tf

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

import utilities.SparseCube
import utilities.Settings._
import utilities.Triplet
import utilities.Vector
import preprocess.MF._


//generate synthetic tensor dataset

object GenSynTensor{
  
  val numFactors = 20
  val numCores = 2
  val numReducers = numCores
  val OUTPUT_DIR = "input/tf/"
  val TRAINING_DIR = OUTPUT_DIR + "syn_train" + PATH_SEPERATOR
  val TESTING_DIR = OUTPUT_DIR + "syn_test" + PATH_SEPERATOR
  val size1 = 1000
  val size2 = 1000
  val size3 = 1000
  val numBlocks1= 2
  val numBlocks2 = 2
  val numBlocks3 = 2
  val numSlices = numBlocks1*numBlocks2*numBlocks3
  val MODE = "local[" + numCores + "]"
  val JARS = Seq("sparkproject.jar")
  val gamma = 1f
  val lambda = 10f
  val tmpDir = "tmp"
  val multicore = numSlices < numCores
  val sparsity = 0.01
  val train_ratio = 0.9
  
  def main(args : Array[String]) {
    
    val currentTime = System.currentTimeMillis();
    
    val options = new Options()
    options.addOption(HELP, false, "print the help message")
    options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
    options.addOption(OUTPUT_OPTION, true, "output path/directory")
    options.addOption(MEAN_OPTION, true, "mean option for input values")
    options.addOption(NUM_REDUCERS_OPTION, true, "number of reducers to use")
    options.addOption(NUM_BLOCKS_1_OPTION, true, "number of blocks for dim 1")
    options.addOption(NUM_BLOCKS_2_OPTION, true, "number of blocks for dim 2")
    options.addOption(NUM_BLOCKS_3_OPTION, true, "number of blocks for dim 3")
    options.addOption(LAMBDA_INIT_OPTION, true, "set lambda value")
    options.addOption(GAMMA_INIT_OPTION, true, "set gamma value")
    options.addOption(NUM_LATENT_FACTORS_OPTION, true, "number of latent factors")
    options.addOption(JAR_OPTION, true, "the path to find jar file")
    options.addOption(TMP_DIR_OPTION, true, "the local dir for tmp files")
    options.addOption(DIM_1_SIZE_OPTION, true, "size of dimention 1")
    options.addOption(DIM_2_SIZE_OPTION, true, "size of dimention 2")
    options.addOption(DIM_3_SIZE_OPTION, true, "size of dimention 3")
    options.addOption(TRAINING_RATIO_OPTION, true, "training data ratio")
    options.addOption(SPARSITY_LEVEL_OPTION, true, "sparsity level")
    options.addOption(MULTICORE_OPTION, false, 
        "using multicore computing on each worker machine")
    
    val parser = new GnuParser();
    val formatter = new HelpFormatter();
    val line = parser.parse(options, args);
    if (line.hasOption(HELP) || args.length == 0) {
      formatter.printHelp("Help", options);
      System.exit(0);
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
    val size1 = 
      if (line.hasOption(DIM_1_SIZE_OPTION))
        line.getOptionValue(DIM_1_SIZE_OPTION).toInt
      else 5000
    val size2 = 
      if (line.hasOption(DIM_2_SIZE_OPTION))
        line.getOptionValue(DIM_2_SIZE_OPTION).toInt
      else 5000
    val size3 = 
      if (line.hasOption(DIM_3_SIZE_OPTION))
        line.getOptionValue(DIM_3_SIZE_OPTION).toInt
      else 5000
    val numReducers =
      if (line.hasOption(NUM_REDUCERS_OPTION))
        line.getOptionValue(NUM_REDUCERS_OPTION).toInt
      else 8
    val numBlocks1 = 
      if (line.hasOption(NUM_BLOCKS_1_OPTION)) 
        line.getOptionValue(NUM_BLOCKS_1_OPTION).toInt
      else 1
    val numBlocks2 = 
      if (line.hasOption(NUM_BLOCKS_2_OPTION)) 
        line.getOptionValue(NUM_BLOCKS_2_OPTION).toInt
      else 1
    val numBlocks3 = 
      if (line.hasOption(NUM_BLOCKS_3_OPTION)) 
        line.getOptionValue(NUM_BLOCKS_3_OPTION).toInt
      else 1
    val gamma = 
      if (line.hasOption(GAMMA_INIT_OPTION)) 
        line.getOptionValue(GAMMA_INIT_OPTION).toFloat
      else 1f
    val lambda = 
      if (line.hasOption(LAMBDA_INIT_OPTION))
        line.getOptionValue(LAMBDA_INIT_OPTION).toFloat
      else 1f
    val numFactors = 
      if (line.hasOption(NUM_LATENT_FACTORS_OPTION)) 
        line.getOptionValue(NUM_LATENT_FACTORS_OPTION).toInt
      else 20
    val tmpDir = if (line.hasOption(TMP_DIR_OPTION))
      line.getOptionValue(TMP_DIR_OPTION)
      else "tmp"
    val multicore = line.hasOption(MULTI_THREAD_OPTION)
    val train_ratio = 
      if (line.hasOption(TRAINING_RATIO_OPTION))
        line.getOptionValue(TRAINING_RATIO_OPTION).toFloat
      else 0.8f
    val sparsity = 
      if (line.hasOption(SPARSITY_LEVEL_OPTION))
        line.getOptionValue(SPARSITY_LEVEL_OPTION).toFloat
      else 0.001f
    
    val numSlices = numBlocks1*numBlocks2*numBlocks3
    System.setProperty("spark.local.dir", tmpDir)
    System.setProperty("spark.default.parallelism", numReducers.toString)
//    System.setProperty("spark.serializer", 
//        "org.apache.spark.serializer.KryoSerializer")
//    System.setProperty("spark.kryoserializer.buffer.mb", "100")
//    System.setProperty("spark.kryo.registrator", "utilities.Registrator")
    System.setProperty("spark.kryo.referenceTracking", "false")
    System.setProperty("spark.storage.memoryFraction", "0.3")
    
//    System.setProperty("spark.worker.timeout", "3600")
//    System.setProperty("spark.storage.blockManagerSlaveTimeoutMs", "8000000")
    
    val StorageLevel = storage.StorageLevel.MEMORY_AND_DISK_SER
    val JOB_NAME = "Syn" + "_d1_" + size1 + "_d2_" + size2 + "_d3_" + size3 + 
      "_K_" + numFactors + "_spa_" + sparsity + "_tr_" + train_ratio + 
      "_ga_" + gamma + "_lam_" + lambda
    val sc = new SparkContext(MODE, JOB_NAME, System.getenv("SPARK_HOME"), JARS)  
      
    val blocksD1 = sc.parallelize((0 until numBlocks1), numBlocks1).map(bid => {
      val blockSize = size1/numBlocks1
      val random = new Random
      val factorMats1 = Array.tabulate(blockSize)(i => 
        Array.fill(numFactors)((random.nextGaussian/math.sqrt(gamma)).toFloat))
      (bid, factorMats1)
    }).persist(StorageLevel)
    println("number of blocks in dim1: " + blocksD1.count)
    val blocksD2 = sc.parallelize((0 until numBlocks2), numBlocks2).map(bid => {
      val blockSize = size2/numBlocks2
      val random = new Random
      val colFactors = Array.tabulate(blockSize)(i => 
        Array.fill(numFactors)((random.nextGaussian/math.sqrt(gamma)).toFloat))
      (bid, colFactors)
    }).persist(StorageLevel)
    println("number of blocks in dim2: " + blocksD2.count)
    val blocksD3 = sc.parallelize((0 until numBlocks3), numBlocks3).map(bid => {
      val blockSize = size3/numBlocks3
      val random = new Random
      val colFactors = Array.tabulate(blockSize)(i => 
        Array.fill(numFactors)((random.nextGaussian/math.sqrt(gamma)).toFloat))
      (bid, colFactors)
    }).persist(StorageLevel)
    println("number of blocks in dim3: " + blocksD3.count)
    
    def getValue(
        factor1: Array[Float], factor2: Array[Float], factor3: Array[Float])= {
      val numFactors = factor1.length
      var k = 0; var res = 0f
      while (k < numFactors) {
        res += factor1(k)*factor2(k)*factor3(k)
      }
      res
    }
    
    val syntheticData = blocksD1.cartesian(blocksD2).cartesian(blocksD3).map{
      case (((bid1, factorMat1), (bid2, factorMat2)), (bid3, factorMat3)) => {
        val bid = bid1*numBlocks2*numBlocks3 + bid2*numBlocks3 + bid3
        val size1 = factorMat1.length
        val size2 = factorMat2.length
        val size3 = factorMat3.length
        val trainingTriplets = new ArrayBuffer[Triplet]
        val testingTriplets = new ArrayBuffer[Triplet]
        val random_spa = new Random
        val random_train = new Random
        val random_normal = new Random
        for (l <- 0 until size1; m <- 0 until size2; n <- 0 until size3) {
          if (random_spa.nextDouble < sparsity) {
            val idx1 = l*numBlocks1 + bid1
            val idx2 = m*numBlocks2 + bid2
            val idx3 = n*numBlocks3 + bid3
            val value = getValue(factorMat1(l), factorMat2(m), factorMat3(n))
            if (random_train.nextDouble < train_ratio) {
              val noise = (random_normal.nextGaussian/math.sqrt(lambda)).toFloat
              trainingTriplets += new Triplet(idx1, idx2, idx3, value+noise)
            }
            else
              testingTriplets += new Triplet(idx1, idx2, idx3, value)
          }
        }
        (trainingTriplets.toArray, testingTriplets.toArray)
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