package tf

import utilities._
import preprocess.TF._

import java.io._
import scala.math._
import scala.util.Sorting._
import scala.collection.mutable.ArrayBuilder
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, HashPartitioner, storage}
import org.apache.spark.SparkContext._
import org.apache.spark.serializer.KryoRegistrator
import org.apache.commons.cli._
import org.apache.hadoop.io.NullWritable

// random row/col subsampling
object PTF extends Settings {
  
  val synthetic = false
  val TRAINING_PATH = if (synthetic) 
    "output/train_Syn_M_1000_N_1000_K_20_spa_0.01_tr_0.9_ga_1.0_lam_10.0" 
    else "input/ml-1m/tf_train"
  val TESTING_PATH = if (synthetic) 
    "output/test_Syn_M_1000_N_1000_K_20_spa_0.01_tr_0.9_ga_1.0_lam_10.0" 
    else "input/ml-1m/tf_test"
  val outputDir = "output/"
  val numCores = 2
  val bufferSize = 100
  val numFactors = 30
  val numBlocks1 = 1
  val numBlocks2 = 1
  val numBlocks3 = 1
  val gamma1_init = 10f
  val gamma2_init = 10f
  val gamma3_init = 10f
  val lambda1 = 5f
  val lambda2 = 5f
  val lambda3 = 5f
  val num_init = 0
  val maxOuterIter = 5
  val maxInnerIter = 10
  val updateDim3 = false
  val l1 = false
  val max_norm = false
  val vb = true
  val admm = true
  val updateGamma1 = true
  val updateGamma2 = true
  val updateGamma3 = false
  val interval = 2
  val mean = if (synthetic) 0f else 3.7668543f
  val scale = 1
  val size1 = if (synthetic) 50000 else 6041
  val size2 = if (synthetic) 50000 else 3953
  val size3 = if (synthetic) 50000 else 35
  val markov = numBlocks1 > 1 || numBlocks2 > 1
  val MODE = "local[" + numCores + "]"
  val JARS = Seq("sparkproject.jar")
  val tmpDir = "tmp"
  val numSlices = numBlocks1*numBlocks2*numBlocks3
  val multicore = numSlices < numCores
  val numReducers = numCores
  
  def toGlobal(factors: Array[Array[Float]], map: Array[Int]) 
    : Array[(Int, Array[Float])] = {
    val length = factors(0).length
    val transformed = Array.ofDim[(Int, Array[Float])](length);
    var i = 0
    while (i < length) {
      transformed(i) = (map(i), factors.map(array => array(i)))
      i += 1
    }
    transformed
  }
  
  def timesGamma(
      factors: Array[Array[Float]], gamma: Array[Float], map: Array[Int], pid: Int) 
    : Array[(Int, (Array[Float], Array[Float], ArrayBuilder.ofInt))] = {
    val length = factors(0).length; val rank = factors.length
    val stats = 
      new Array[(Int, (Array[Float], Array[Float], ArrayBuilder.ofInt))](length)
    var r = 0
    while (r < length) {
      val ab = new ArrayBuilder.ofInt
      ab += pid
      stats(r) = 
        (map(r), (Array.tabulate(rank)(k=>factors(k)(r)*gamma(k)), gamma.clone, ab))
      r += 1
    }
    stats
  }
  
  def l1Update(weightedSum: Array[Float], gammaSum: Array[Float], lambda: Float) = {
    val l = weightedSum.length; var j = 0
    while (j < l) {
      weightedSum(j) = 
      if (weightedSum(j) > lambda) (weightedSum(j)-lambda)/gammaSum(j)
      else if (weightedSum(j) < -lambda) (weightedSum(j)+lambda)/gammaSum(j)
      else 0
      j += 1 
    }
  }
  
  def maxNormUpdate(
      weightedSum: Array[Float], gammaSum: Array[Float], lambda: Float) = {
    val l = weightedSum.length; var j = 0; var norm = 0f
    while (j < l) {
      weightedSum(j) /= gammaSum(j)
      norm += weightedSum(j)*weightedSum(j)
      j += 1
    }
    if (norm > lambda) {
      val ratio = math.sqrt(lambda/norm).toFloat
      j = 0
      while (j < l) {
        weightedSum(j) = weightedSum(j)*ratio
        j += 1 
      }
    }
  }
  
  def l2Update(weightedSum: Array[Float], gammaSum: Array[Float], lambda: Float) = {
    val l = weightedSum.length; var j = 0
    while (j < l) { weightedSum(j) /= (gammaSum(j)+lambda); j += 1 }
  }
  
  def updateGlobalFactors(
      sc: SparkContext, numSlices: Int,
      stats: RDD[(Int, (Array[Float], Array[Float], ArrayBuilder.ofInt))], 
      factors: Array[Array[Float]], lambda: Float) = {
    val paras = stats.reduceByKey{
      // agregate the statistics for updating the global parameters
      case((factor1, gamma1, pids1), (factor2, gamma2, pids2)) => {
        var i = 0; while(i < factor2.length) { 
          factor1(i) += factor2(i); gamma1(i) += gamma2(i); i += 1 }
        pids1 ++= pids2.result
        (factor1, gamma1, pids1)
      }
    }.sortByKey(true).map{case(id, para) => para}.collect
//    assert(factors.length == paras.length, "dimension mismatch, dim(factors)=" 
//        + factors.length + " dim(para)=" + paras.length)
    val size = paras.length; val numFactors = factors(0).length
    for (k <- 0 until numFactors) {
      factors(0)(k) = (lambda*factors(1)(k) + paras(0)._1(k))/(lambda + paras(0)._2(k))
    }
    for (n <- 1 until size-1) {
      for (k <- 0 until numFactors) {
        val nume = lambda*(factors(n-1)(k) + factors(n+1)(k)) + paras(n)._1(k)
        val deno = 2*lambda + paras(n)._2(k)
        factors(n)(k) = nume/deno
      }
    }
    for (k <- 0 until numFactors) {
      val nume = lambda*factors(size-2)(k) + paras(size-1)._1(k)
      val deno = lambda + paras(size-1)._2(k)
      factors(size-1)(k) = nume/deno
    }
    val pids = paras.map{case(factor, gamma, pid) => pid.result}
    sc.parallelize(factors.zipWithIndex.zip(pids), numSlices).flatMap{
      case((factor, idx), pid) => {
        val distributed = Array.ofDim[(Int, (Int, Array[Float]))](pid.length)
        var i = 0; while (i < pid.length) {
          distributed(i) = (pid(i), (idx, factor)); i += 1
        }
       distributed
      }
    }
  }
  
  def updateGlobalFactors(
      stats: RDD[(Int, (Array[Float], Array[Float], ArrayBuilder.ofInt))], 
      l1: Boolean, max_norm: Boolean, lambda: Float) = {
    stats.reduceByKey{
      // agregate the statistics for updating the global parameters
      case((factor1, gamma1, pids1), (factor2, gamma2, pids2)) => {
        var i = 0; while(i < factor2.length) { 
          factor1(i) += factor2(i); gamma1(i) += gamma2(i); i += 1 
        }
        pids1 ++= pids2.result
        (factor1, gamma1, pids1)
      }
    }.mapValues{
      // update the global parameters
      case(weightedSum, gammaSum, pids) => {
        if (pids.result.length == 1) (Array.ofDim[Float](weightedSum.length), pids)
        else {
          if (l1) l1Update(weightedSum, gammaSum, lambda)
          else if (max_norm) maxNormUpdate(weightedSum, gammaSum, lambda)
          else l2Update(weightedSum, gammaSum, lambda)
          (weightedSum, pids)
        }
      }
    }.flatMap{
      // distribute the updated global parameters to local partitions
      case(idx, (factor, pids_buffer)) => {
        val pids = pids_buffer.result
        val distributed = Array.ofDim[(Int, (Int, Array[Float]))](pids.length)
        var i = 0; while (i < pids.length) {
          distributed(i) = (pids(i), (idx, factor)); i += 1
        }
       distributed
      }
    }
  }
  
  def main(args : Array[String]) {
    val currentTime = System.currentTimeMillis();
    
    val options = new Options()
    options.addOption(HELP, false, "print the help message")
    options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
    options.addOption(TRAINING_OPTION, true, "training data input path/directory")
    options.addOption(TESTING_OPTION, true, "testing data output path/directory")
    options.addOption(OUTPUT_OPTION, true, "output path/directory")
    options.addOption(MEAN_OPTION, true, "mean option for input values")
    options.addOption(SCALE_OPTION, true, "scale option for input values")
    options.addOption(NUM_REDUCERS_OPTION, true, "number of reducers to use")
    options.addOption(OUTER_ITERATION_OPTION, true, "max # of outer iterations")
    options.addOption(INNER_ITERATION_OPTION, true, "max # of inner iterations")
    options.addOption(VB_INFERENCE_OPTION, false, "Variational Bayesian inference")
    options.addOption(ADMM_OPTION, false, "using ADMM")
    options.addOption(NUM_BLOCKS_1_OPTION, true, "number of blocks for dim 1")
    options.addOption(NUM_BLOCKS_2_OPTION, true, "number of blocks for dim 2")
    options.addOption(NUM_BLOCKS_3_OPTION, true, "number of blocks for dim 3")
    options.addOption(LAMBDA_1_INIT_OPTION, true, "set lambda_1 value")
    options.addOption(LAMBDA_2_INIT_OPTION, true, "set lambda_2 value")
    options.addOption(LAMBDA_3_INIT_OPTION, true, "set lambda_3 value")
    options.addOption(UPDATE_GAMMA_1_OPTION, false, "empirical estimate on gamma_r")
    options.addOption(GAMMA_1_INIT_OPTION, true, "initial guess for gamma_r")
    options.addOption(UPDATE_GAMMA_2_OPTION, false, "empirical estimate on gamma_c")
    options.addOption(GAMMA_2_INIT_OPTION, true, "initial guess for gamma_c")
    options.addOption(UPDATE_GAMMA_3_OPTION, false, "empirical estimate on gamma_c")
    options.addOption(GAMMA_3_INIT_OPTION, true, "initial guess for gamma_c")
    options.addOption(NUM_LATENT_FACTORS_OPTION, true, "number of latent factors")
    options.addOption(JAR_OPTION, true, "the path to find jar file")
    options.addOption(TMP_DIR_OPTION, true, "the local dir for tmp files")
    options.addOption(L1_REGULARIZATION, false, "use l1 regularization")
    options.addOption(DIM_1_SIZE_OPTION, true, "size of dimention 1")
    options.addOption(DIM_2_SIZE_OPTION, true, "size of dimention 2")
    options.addOption(DIM_3_SIZE_OPTION, true, "size of dimention 3")
    options.addOption(MULTICORE_OPTION, false, 
        "using multicore computing on each worker machine")
    options.addOption(MAX_NORM_OPTION, false, "use max-norm regilarization")
    options.addOption(INTERVAL_OPTION, true, "interval to calculate testing RMSE")
    options.addOption(SYNTHETIC_DATA_OPTION, false, "using synthetic data")
    options.addOption(MARKOV_OPTION, false, "markov assumption option")
    options.addOption(TENSOR_OPTION, false,"tensor factorization option")
    options.addOption(BUFFER_SIZE_OPTION, true, 
        "buffer size for kyro serialization (in mb)")
    options.addOption(NUM_INIT_OPTION, true, 
        "num of init steps for tensor factorization")
        
    val parser = new GnuParser();
    val formatter = new HelpFormatter();
    val line = parser.parse(options, args);
    if (line.hasOption(HELP) || args.length == 0) {
      formatter.printHelp("Help", options);
      System.exit(0);
    }
    assert(line.hasOption(RUNNING_MODE_OPTION), "running mode not specified")
    assert(line.hasOption(TRAINING_OPTION), "training data path not specified")
    assert(line.hasOption(TESTING_OPTION), "testing data path not specified")
    assert(line.hasOption(JAR_OPTION), "jar file path not specified")
    
    val MODE = line.getOptionValue(RUNNING_MODE_OPTION)
    val TRAINING_PATH = line.getOptionValue(TRAINING_OPTION)
    val TESTING_PATH = line.getOptionValue(TESTING_OPTION)
    val JARS = Seq(line.getOptionValue(JAR_OPTION))
    val outputDir = 
      if (line.hasOption(OUTPUT_OPTION))
        if (!line.getOptionValue(OUTPUT_OPTION).endsWith(PATH_SEPERATOR))
          line.getOptionValue(OUTPUT_OPTION) + PATH_SEPERATOR
        else
          line.getOptionValue(OUTPUT_OPTION)
      else
        "output" + PATH_SEPERATOR
    val mean = 
      if (line.hasOption(MEAN_OPTION))
        line.getOptionValue(MEAN_OPTION).toFloat
      else 0f
    val scale = 
      if (line.hasOption(SCALE_OPTION))
        line.getOptionValue(SCALE_OPTION).toFloat
      else 1f
    val size1 = 
      if (line.hasOption(DIM_1_SIZE_OPTION))
        line.getOptionValue(DIM_1_SIZE_OPTION).toInt
      else 5000000
    val size2 = 
      if (line.hasOption(DIM_2_SIZE_OPTION))
        line.getOptionValue(DIM_2_SIZE_OPTION).toInt
      else 5000000
    val size3 = 
      if (line.hasOption(DIM_3_SIZE_OPTION))
        line.getOptionValue(DIM_3_SIZE_OPTION).toInt
      else 5000000
    val numReducers =
      if (line.hasOption(NUM_REDUCERS_OPTION))
        line.getOptionValue(NUM_REDUCERS_OPTION).toInt
      else 8
    val maxOuterIter =
      if (line.hasOption(OUTER_ITERATION_OPTION))
        line.getOptionValue(OUTER_ITERATION_OPTION).toInt
      else 10
    val maxInnerIter = 
      if (line.hasOption(INNER_ITERATION_OPTION))
        line.getOptionValue(INNER_ITERATION_OPTION).toInt
      else 5
    val vb = line.hasOption(VB_INFERENCE_OPTION)
    val admm = line.hasOption(ADMM_OPTION)
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
    val updateGamma1 = line.hasOption(UPDATE_GAMMA_1_OPTION)
    val updateGamma2 = line.hasOption(UPDATE_GAMMA_2_OPTION)
    val updateGamma3 = line.hasOption(UPDATE_GAMMA_3_OPTION)
    val gamma1_init = 
      if (line.hasOption(GAMMA_1_INIT_OPTION)) 
        line.getOptionValue(GAMMA_1_INIT_OPTION).toFloat
      else 10f
    val gamma2_init = 
      if (line.hasOption(GAMMA_2_INIT_OPTION)) 
        line.getOptionValue(GAMMA_2_INIT_OPTION).toFloat
      else 10f
    val gamma3_init = 
      if (line.hasOption(GAMMA_3_INIT_OPTION)) 
        line.getOptionValue(GAMMA_3_INIT_OPTION).toFloat
      else 1f
    val lambda1 = 
      if (line.hasOption(LAMBDA_1_INIT_OPTION))
        line.getOptionValue(LAMBDA_1_INIT_OPTION).toFloat
      else 0f
    val lambda2 = 
      if (line.hasOption(LAMBDA_2_INIT_OPTION))
        line.getOptionValue(LAMBDA_2_INIT_OPTION).toFloat
      else 0f
    val lambda3 = 
      if (line.hasOption(LAMBDA_3_INIT_OPTION))
        line.getOptionValue(LAMBDA_3_INIT_OPTION).toFloat
      else 0f
    val numFactors = 
      if (line.hasOption(NUM_LATENT_FACTORS_OPTION)) 
        line.getOptionValue(NUM_LATENT_FACTORS_OPTION).toInt
      else 20
    val tmpDir = if (line.hasOption(TMP_DIR_OPTION))
      line.getOptionValue(TMP_DIR_OPTION)
      else "tmp"
    val l1 = line.hasOption(L1_REGULARIZATION)
    val max_norm = line.hasOption(MAX_NORM_OPTION)
    val multicore = line.hasOption(MULTICORE_OPTION)
    val interval = 
      if (line.hasOption(INTERVAL_OPTION))
        line.getOptionValue(INTERVAL_OPTION).toInt
      else 2
    val synthetic = line.hasOption(SYNTHETIC_DATA_OPTION)
    val markov = line.hasOption(MARKOV_OPTION)
    val updateDim3 = line.hasOption(TENSOR_OPTION)
    val bufferSize = if (line.hasOption(BUFFER_SIZE_OPTION)) 
      line.getOptionValue(BUFFER_SIZE_OPTION).toInt
      else 100
    val num_init = if (line.hasOption(NUM_INIT_OPTION))
      line.getOptionValue(NUM_INIT_OPTION).toInt
      else 0
    
    val admms = new Array[Boolean](3)
    admms(0) = admm && (numBlocks2 > 1 || numBlocks3 > 1) 
    admms(1) = admm && (numBlocks1 > 1 || numBlocks3 > 1)
    admms(2) = admm && (numBlocks1 > 1 || numBlocks2 > 1) && updateDim3
    val updateGammas = new Array[Boolean](3)
    updateGammas(0) = updateGamma1 && vb && !admms(0)
    updateGammas(1) = updateGamma2 && vb && !admms(1)
    updateGammas(2) = updateGamma3 && vb && !admms(2) && updateDim3
    val numSlices = numBlocks1*numBlocks2*numBlocks3
    
    var JOB_NAME = if (updateDim3) "DIST_TF_ni_" + num_init else "DIST_MF"
    if (vb) JOB_NAME += "_VB" else JOB_NAME += "_MAP"
    if (admm) JOB_NAME += "_ADMM"
    if (l1) JOB_NAME += "_l1"
    if (max_norm) JOB_NAME += "_maxnorm"
    if (multicore) JOB_NAME += "_multicore"
    if (markov) JOB_NAME + "_markov"
    JOB_NAME +=  "_oi_" + maxOuterIter + "_ii_" + maxInnerIter + "_K_" + numFactors + 
    "_b1_" + numBlocks1 + "_b2_" + numBlocks2 + "_b3_" + numBlocks3 +
    "_ug1_" + updateGammas(0) + "_ug2_" + updateGammas(1) + "_ug3_" + updateGammas(2) +
    "_g1_" + gamma1_init + "_g2_" + gamma2_init + "_g3_" + gamma3_init +
    "_l1_" + lambda1 + "_l2_" + lambda2 + "_l3_" + lambda3
    val logPath = outputDir + JOB_NAME + ".txt"
    System.setProperty("spark.local.dir", tmpDir)
    System.setProperty("spark.default.parallelism", numReducers.toString)
    System.setProperty("spark.storage.memoryFraction", "0.5")
//    System.setProperty("spark.ui.port", "44717")
//    System.setProperty("spark.locality.wait", "10000")
//    System.setProperty("spark.worker.timeout", "3600")
//    System.setProperty("spark.storage.blockManagerSlaveTimeoutMs", "8000000")
//    System.setProperty("spark.akka.timeout", "60")
//    System.setProperty("spark.akka.askTimeout", "60")
//    System.setProperty("spark.serializer", 
//        "org.apache.spark.serializer.KryoSerializer")
//    System.setProperty("spark.kryo.registrator", "utilities.Registrator")
//    System.setProperty("spark.kryoserializer.buffer.mb", bufferSize.toString)
//    System.setProperty("spark.kryo.referenceTracking", "false")
//  System.setProperty("spark.mesos.coarse", "true")
//    System.setProperty("spark.cores.max", numCores.toString)
    
    val storageLevel = storage.StorageLevel.MEMORY_AND_DISK_SER
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))
//    val sc = new SparkContext(MODE, JOB_NAME, System.getenv("SPARK_HOME"), 
//        Seq(System.getenv("SPARK_EXAMPLES_JAR")))
    val sc = new SparkContext(MODE, JOB_NAME, System.getenv("SPARK_HOME"), JARS)
    
    // read in the sparse matrix:
    val trainingTuples =
      if (synthetic || TRAINING_PATH.toLowerCase.contains("ml")) 
        sc.sequenceFile[NullWritable, Triplet](TRAINING_PATH, numSlices).map(pair => 
          new Triplet(pair._2._1, pair._2._2, pair._2._3, (pair._2.value-mean)/scale))
      else
        sc.textFile(TRAINING_PATH, numSlices)
          .flatMap(line => parseLine(line, mean, scale))
    val map1 = sc.broadcast(getPartitionMap(size1, numBlocks1, bwLog))
    val map2 = sc.broadcast(getPartitionMap(size2, numBlocks2, bwLog))
    val map3 = sc.broadcast(getPartitionMap(size3, numBlocks3, bwLog))
    val partitioner = new HashPartitioner(numSlices)
    val trainingData = getPartitionedData(trainingTuples, numBlocks2, numBlocks3,
        map1, map2, map3, partitioner, multicore)
    trainingData.persist(storageLevel)
    val samplesPerBlock = trainingData.map{case(id, data) => data.ids1.length}.collect
    val dim1PerBlock = trainingData.map{case(id, data) => data.map1.length}.collect
    val dim2PerBlock = trainingData.map{case(id, data) => data.map2.length}.collect
    val dim3PerBlock = trainingData.map{case(id, data) => data.map3.length}.collect
    val numTrainingSamples = samplesPerBlock.sum
    var b = 0
    println("samples per block:")
    bwLog.write("samples per block:\n")
    while (b < numSlices) {
      val msg = b + ":" + samplesPerBlock(b) + "\t"
      print(msg); bwLog.write(msg)
      b += 1
    }
    println; bwLog.newLine
    println("size of dim1 in each block:")
    bwLog.write("size of dim1 in each block:\n")
    b = 0
    while (b < numSlices) {
      val msg = b + ":" + dim1PerBlock(b) + "\t"
      print(msg); bwLog.write(msg)
      b += 1
    }
    println; bwLog.newLine
    println("size of dim2 in each block:")
    bwLog.write("size of dim2 in each block:\n")
    b = 0
    while (b < numSlices) {
      val msg = b + ":" + dim2PerBlock(b) + "\t"
      print(msg); bwLog.write(msg)
      b += 1
    }
    println; bwLog.newLine
    println("size of dim3 in each block:")
    bwLog.write("size of dim3 in each block:\n")
    b = 0
    while (b < numSlices) {
      val msg = b + ":" + dim3PerBlock(b) + "\t"
      print(msg); bwLog.write(msg)
      b += 1
    }
    bwLog.newLine
    val testingTuples = 
      if (synthetic || TESTING_PATH.toLowerCase.contains("ml")) 
        sc.sequenceFile[NullWritable, Triplet](TESTING_PATH, numSlices).map(pair => 
          new Triplet(pair._2._1, pair._2._2, pair._2._3, (pair._2.value-mean)/scale))
      else 
        sc.textFile(TESTING_PATH, numSlices)
          .flatMap(line => parseLine(line, mean, scale))  
    val testingData = getPartitionedData(testingTuples, numBlocks2, numBlocks3, 
        map1, map2, map3, partitioner)
    testingData.persist(storageLevel)
    val numTestingSamples = testingData.map(pair => pair._2.ids1.length).sum
    
    println("Size of dim1: " + size1)
    println("Size of dim2: " + size2)
    println("Size of dim3: " + size3)
    println("Number of training samples: " + numTrainingSamples)
    println("Number of testing samples: " + numTestingSamples)    
    println("Mean: " + mean)
    println("Scale: " + scale)
    bwLog.write("Size of dim1: " + size1 + "\n")
    bwLog.write("Size of dim2: " + size2 + "\n")
    bwLog.write("Size of dim3: " + size3 + "\n")
    bwLog.write("Number of training samples: " + numTrainingSamples + "\n")
    bwLog.write("Number of testing samples: " + numTestingSamples + "\n")
    bwLog.write("Mean: " + mean + "\n")
    bwLog.write("Scale: " + scale + "\n")
    
    bwLog.write("Preprocesing data finished in  " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)\n")
    println("Preprocesing data finished in " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    
    var iterTime = System.currentTimeMillis()
    var iter = 0
    val thre = 0.001f
    var localOutputs = trainingData.mapValues(data => {
//      val updateDim3 = false
      val gammas = new Array[Float](3)
      gammas(0) = gamma1_init; gammas(1) = gamma2_init
      gammas(2) = if (updateDim3 && num_init == 0) gamma3_init 
      else Float.PositiveInfinity
      Model(data, numFactors, gammas, admms, vb)
        .ccd(data, maxInnerIter, 1, thre, updateDim3 && num_init == 0, multicore, vb)
    }).persist(storageLevel)    
    
    var localModels = localOutputs.mapValues{
      case (model, iter, rmse, sDe1, sDe2, sDe3) => {
        val gammas = new Array[Float](3)
        gammas(0) = gamma1_init; gammas(1) = gamma2_init 
        gammas(2) = if (updateDim3) gamma3_init else Float.PositiveInfinity
        model.setGamma(gammas)
        model
      }
    }
    var localInfo = localOutputs.mapValues{
      case(model, iter, rmse, sDe1, sDe2, sDe3) => (iter, rmse, sDe1, sDe2, sDe3)
    }.collect
    
    val prior1 = localModels.mapValues(model => 
      model.idxMaps(0).map(i=>(i, Array.ofDim[Float](model.numFactors)))
    ).persist(storageLevel)
    val prior2 = localModels.mapValues(model => 
      model.idxMaps(1).map(i=>(i, Array.ofDim[Float](model.numFactors)))
    ).persist(storageLevel)
    val prior3 = localModels.mapValues(model => 
      model.idxMaps(2).map(i=>(i, Array.fill(model.numFactors)(1f)))
    ).persist(storageLevel)
    val factors = Array.ofDim[Float](size3, numFactors)
    
    while (iter < maxOuterIter) {
      val globalFactorMat1 = if (numBlocks2>1 || numBlocks3>1)
        updateGlobalFactors(localModels.flatMap{
          case(pid, model) => timesGamma(model.getFactorStats(0, admms(0)), 
            model.gammas(0), model.idxMaps(0), pid)
        }, l1, max_norm, lambda1)
        .groupByKey(partitioner).mapValues(seq => (seq.toArray).sortBy(_._1))
      else prior1
      val globalFactorMat2 = if (numBlocks1>1 || numBlocks3>1)
        updateGlobalFactors(localModels.flatMap{
          case(pid, model) => timesGamma(model.getFactorStats(1, admms(1)), 
            model.gammas(1), model.idxMaps(1), pid)
        }, l1, max_norm, lambda2)
        .groupByKey(partitioner).mapValues(seq => (seq.toArray).sortBy(_._1))
      else prior2
      val globalFactorMat3 = 
      if ((numBlocks1>1 || numBlocks2>1) && updateDim3 && iter >= num_init ) {
        println("begin to update globalFactorMat3...")
        if (markov) {
          updateGlobalFactors(sc, numSlices, localModels.flatMap{
            case(pid, model) => timesGamma(model.getFactorStats(2, admms(2)), 
              model.gammas(2), model.idxMaps(2), pid)
          }, factors, lambda3)
        }
        else{
          updateGlobalFactors(localModels.flatMap{
            case(pid, model) => timesGamma(model.getFactorStats(2, admms(2)), 
              model.gammas(2), model.idxMaps(2), pid)
          }, l1, max_norm, lambda3)
        }
      }.groupByKey(partitioner).mapValues(seq => (seq.toArray).sortBy(_._1))
      else prior3
      
      val priors = globalFactorMat1.join(globalFactorMat2).join(globalFactorMat3)
      if (iter % interval == 0 || iter+1 == maxOuterIter) {
        priors.persist(storageLevel)
      }
      
      val globalModel = 
        if (numBlocks1>1 && numBlocks2>1 && numBlocks3>1) 
          priors.mapValues(pair => (pair._1._1, pair._1._2, pair._2))
        else
          localModels.join(priors).mapValues{
            case(model, prior) => (
              if (numBlocks2==1 && numBlocks3==1) 
                toGlobal(model.factorMats(0), model.idxMaps(0))
              else prior._1._1,
              if (numBlocks1==1 && numBlocks3==1) 
                toGlobal(model.factorMats(1), model.idxMaps(1))
              else prior._1._2,
              if ((numBlocks1==1 && numBlocks2==1)) 
                toGlobal(model.factorMats(2), model.idxMaps(2))
              else prior._2
            )
          }
      
      val trainingRMSE_global = 
        if (iter % interval == 0 || iter+1 == maxOuterIter) {
          scale*math.sqrt(trainingData.join(globalModel).map{
            case(id, (data, (piror1, prior2, prior3))) => data.getSE(
              Model.toLocal(data.map1, piror1), 
              Model.toLocal(data.map2, prior2),
              Model.toLocal(data.map3, prior3)
            )
          }.reduce(_+_)/numTrainingSamples)
        }
        else 0
      
      val testingRMSE_global = 
        if (iter % interval == 0 || iter+1 == maxOuterIter) {
          scale*math.sqrt(testingData.join(globalModel).map{
            case(id, (data, (piror1, prior2, prior3))) => data.getSE(
              Model.toLocal(data.map1, piror1), 
              Model.toLocal(data.map2, prior2),
              Model.toLocal(data.map3, prior3)
            )
          }.reduce(_+_)/numTestingSamples)
        }
        else 0
      
      val testingRMSE_local = scale*math.sqrt(testingData.join(localModels).map{
        case(idx, (data, model)) => data.getSE(
          Model.toLocal(data.map1, toGlobal(model.factorMats(0), model.idxMaps(0))),
          Model.toLocal(data.map2, toGlobal(model.factorMats(1), model.idxMaps(1))),
          Model.toLocal(data.map3, toGlobal(model.factorMats(2), model.idxMaps(2)))
        )
      }.reduce(_+_)/numTestingSamples)
      
      if (updateGammas(0) || updateGammas(1) || updateGammas(2)) {
        val gammas = localModels.mapValues(model => 
          (model.gammas(0), model.gammas(1), model.gammas(2))).collect
        gammas.foreach{
          case(pid, (gamma1, gamma2, gamma3)) => {
            val bid1 = pid/(numBlocks2*numBlocks3) 
            val bid2 = pid%(numBlocks2*numBlocks3)/numBlocks3
            val bid3 = bid1%numBlocks3
            val msg = "block(" + bid1 + "," + bid2 + "," + bid3 + ")" + 
              "\ngamma1: " + gamma1.mkString(" ") + 
              "\ngamma2: " + gamma2.mkString(" ") + 
              "\ngamma3: " + gamma3.mkString(" ")
            println(msg); bwLog.write(msg + "\n")
          }
        } 
      }
      localInfo.foreach{
        case(pid, (iter, rmse, sDe1, sDe2, sDe3)) => {
          val bid1 = pid/(numBlocks2*numBlocks3) 
          val bid2 = pid%(numBlocks2*numBlocks3)/numBlocks3
          val bid3 = bid1%numBlocks3
          val msg = "block(" + bid1 + "," + bid2 + "," + bid3 + ")" + 
            " num inner iters: " + iter + " block training rmse: " + rmse + 
            " sDe1: " + sDe1 + " sDe2: " + sDe2 + " sDe3: " + sDe3
          println(msg); bwLog.write(msg + "\n")
        }
      }
      val time = (System.currentTimeMillis() - iterTime)*0.001
      println("Training RMSE: " + trainingRMSE_global +
    		"\t Testing RMSE: global " + testingRMSE_global + 
        " local " + testingRMSE_local)
      bwLog.write("Training RMSE: global " + trainingRMSE_global +
        "\t Testing RMSE: global " + testingRMSE_global + 
        " local " + testingRMSE_local + "\n")
      println("Iter: " + iter + " finsied, time elapsed: " + time)
      bwLog.write("Iter: " + iter + " finished, time elapsed: " + time + "\n")
      iterTime = System.currentTimeMillis()
      iter += 1
      if (iter < maxOuterIter) {
        val thre = 0.00001f
//        val innerIter = maxInnerIter
        val innerIter = math.max(maxInnerIter - 2*iter, 2)
        val rddId = localOutputs.id
        localOutputs = trainingData.join(localModels).join(priors).mapValues{
          case((data, model), ((priors1, priors2), priors3)) => {
            val priorMat1 = Model.toLocal(data.map1, priors1)
            val priorMat2 = Model.toLocal(data.map2, priors2)
            val priorMat3 = Model.toLocal(data.map3, priors3)
            admms(2) = admm && (numBlocks1 > 1 || numBlocks2 > 1) && updateDim3 && 
              iter > num_init
            model.ccd(data, innerIter, 1, thre, updateDim3&&iter>=num_init, multicore, 
              vb, admms, updateGammas, priorMat1, priorMat2, priorMat3)
          }
        }
        localOutputs.persist(storageLevel)
        localModels = localOutputs.mapValues(_._1)
        localInfo = localOutputs.mapValues{
          case(model, iter, rmse, sDe1, sDe2, sDe3) => (iter, rmse, sDe1, sDe2, sDe3)
        }.collect
        priors.unpersist(false)
        println("Let's remove RDD: " + rddId)
        sc.getPersistentRDDs(rddId).unpersist(true)
      }
    }
    bwLog.write("Total time elapsed " 
      + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    bwLog.close()
    println("Total time elapsed " 
      + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    if (true) {
      localModels.map{
        case (bid, model) => model.factorMats(0).view.zipWithIndex.map{
          case(factor, k) => k + "\t" + factor.mkString(" ")
        }.mkString("\n")
      }.saveAsTextFile(outputDir + "localFactors1")
      if (vb) localModels.map{
        case (bid, model) => model.getPrecisionMat(0).view.zipWithIndex.map{
          case(prec, k) => k + "\t" + prec.mkString(" ")
        }.mkString("\n")
      }.saveAsTextFile(outputDir + "localPrecs1")
      localModels.map{
        case (bid, model) => model.factorMats(1).view.zipWithIndex.map{
          case(factor, k) => k + "\t" + factor.mkString(" ")
        }.mkString("\n")
      }.saveAsTextFile(outputDir + "localFactors2")
      if (vb) localModels.map{
        case (bid, model) => model.getPrecisionMat(1).view.zipWithIndex.map{
          case(prec, k) => k + "\t" + prec.mkString(" ")
        }.mkString("\n")
      }.saveAsTextFile(outputDir + "localPrecs2")
      localModels.map{
        case (bid, model) => model.factorMats(2).view.zipWithIndex.map{
          case(factor, k) => k + "\t" + factor.mkString(" ")
        }.mkString("\n")
      }.saveAsTextFile(outputDir + "localFactors3")
      if (vb) localModels.map{
        case (bid, model) => model.getPrecisionMat(2).view.zipWithIndex.map{
          case(prec, k) => k + "\t" + prec.mkString(" ")
        }.mkString("\n")
      }.saveAsTextFile(outputDir + "localPrecs3")
    }
    System.exit(0)
  }
}