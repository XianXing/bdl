package mf

import utilities._
import preprocess.MF._

import java.io._
import scala.math._
import scala.util.Sorting._
import scala.collection.mutable.HashSet
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, HashPartitioner, storage}
import org.apache.spark.SparkContext._
import org.apache.spark.serializer.KryoRegistrator

import org.apache.commons.cli._
import org.apache.hadoop.io.NullWritable

// random row/col subsampling
object PMF extends Settings {
  
  val syn = false
  val trainingPath = 
    if (syn) "output/train_Syn_M_1000_N_1000_K_20_spa_0.01_tr_0.9_ga_1.0_lam_10.0" 
    else "input/ml-1m/mf_train"
  val testingPath = 
    if (syn) "output/test_Syn_M_1000_N_1000_K_20_spa_0.01_tr_0.9_ga_1.0_lam_10.0" 
    else "input/ml-1m/mf_test"
  val outputDir = "output/"
  val numCores = 1
  val numRowBlocks = 2
  val numColBlocks = 1
  val gamma_r_init = 10f
  val gamma_c_init = 10f
  val gamma_x_init = 1f
  val lambda_r = 1f
  val lambda_c = 1f
  val maxOuterIter = 5
  val maxInnerIter = 10
  val numFactors = 20
  val l1 = false
  val max_norm = true
  val ccdpp = true
  val vb = true
  val admm = true
  //for each movie, numRows = 1621, numCols = 55423, mean: 4.037181f
  val numRows = if (syn) 50000 else 6041
  val numCols = if (syn) 50000 else 3953
  val mean = if (syn) 0f else 3.7668543f
  val scale = 1
  val numReducers = 5*numCores
  val numSlices = numRowBlocks*numColBlocks
  val admm_r = admm && numColBlocks > 1
  val admm_c = admm && numRowBlocks > 1
  val max_norm_r = max_norm && numColBlocks > 1
  val max_norm_c = max_norm && numRowBlocks > 1
  val iso = false
  val updateGammaR = vb
  val updateGammaC = vb
  val mode = "local[" + numCores + "]"
  val jars = Seq("sparkproject.jar")
  val memory = "1g"
  val tmpDir = "tmp"
  val multicore = numCores > numSlices
  val interval = 2
  
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
//    transformed.sortBy(pair => pair._1)
  }
  
  def timesGamma(
      factors: Array[Array[Float]], gamma: Array[Float], map: Array[Int], pid: Int) 
    : Array[(Int, (Array[Float], Array[Float], List[Int]))] = {
    val length = factors(0).length; val rank = factors.length
    val stats = 
      new Array[(Int, (Array[Float], Array[Float], List[Int]))](length)
    var r = 0
    while (r < length) {
      stats(r) = (map(r), 
          (Array.tabulate(rank)(k=>factors(k)(r)*gamma(k)), gamma.clone, List(pid)))
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
  
  def updateGlobalPriors(stats: RDD[(Int, (Array[Float], Array[Float], List[Int]))], 
      l1: Boolean, max_norm: Boolean, lambda: Float, part: HashPartitioner) = {
    
    def agregate(p1: (Array[Float], Array[Float], List[Int]), 
        p2: (Array[Float], Array[Float], List[Int])) = {
      // agregate the statistics for updating the global parameters
      (p1, p2) match {
        case((factor1, gamma1, pids1), (factor2, gamma2, pids2)) => {
          var i = 0; while(i < factor2.length) {
            factor1(i) += factor2(i); gamma1(i) += gamma2(i); i += 1 
          }
          (factor1, gamma1, pids1:::pids2)
        }
      }
    }
    
    stats.reduceByKey(part, (p1, p2) => agregate(p1, p2)).mapValues{
      // update the global parameters
      case(weightedSum, gammaSum, pids) => {
        if (pids.length == 1) (Array.ofDim[Float](weightedSum.length), pids)
        else {
          if (l1) l1Update(weightedSum, gammaSum, lambda)
          else if (max_norm) maxNormUpdate(weightedSum, gammaSum, lambda)
          else l2Update(weightedSum, gammaSum, lambda)
          (weightedSum, pids)
        }
      }
    }.flatMap{
      // distribute the updated global parameters to local partitions
      case(idx, (factor, pids)) => pids.map(id => (id, (idx, factor)))
    }
  }
  
  def main(args : Array[String]) {
    val currentTime = System.currentTimeMillis()
    
    val options = new Options()
    options.addOption(HELP, false, "print the help message")
    options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
    options.addOption(TRAINING_OPTION, true, "training data input path/directory")
    options.addOption(TESTING_OPTION, true, "testing data input path/directory")
    options.addOption(OUTPUT_OPTION, true, "output path/directory")
    options.addOption(MEAN_OPTION, true, "mean option for input values")
    options.addOption(SCALE_OPTION, true, "scale option for input values")
    options.addOption(NUM_CORES_OPTION, true, "number of cores to use")
    options.addOption(NUM_REDUCERS_OPTION, true, "number of reducers to use")
    options.addOption(NUM_SLICES_OPTION, true, "number of slices of the data")
    options.addOption(OUTER_ITERATION_OPTION, true, "max # of outer iterations")
    options.addOption(INNER_ITERATION_OPTION, true, "max # of inner iterations")
    options.addOption(VB_INFERENCE_OPTION, false, "Variational Bayesian inference")
    options.addOption(ADMM_OPTION, false, "using ADMM")
    options.addOption(NUM_COL_BLOCKS_OPTION, true, "number of column blocks")
    options.addOption(NUM_ROW_BLOCKS_OPTION, true, "number of row blocks")
    options.addOption(LAMBDA_C_INIT_OPTION, true, "set lambda_c value")
    options.addOption(LAMBDA_R_INIT_OPTION, true, "set lambda_r value")
    options.addOption(UPDATE_GAMMA_R_OPTION, false, "empirical estimate on gamma_r")
    options.addOption(GAMMA_R_INIT_OPTION, true, "initial guess for gamma_r")
    options.addOption(UPDATE_GAMMA_C_OPTION, false, "empirical estimate on gamma_c")
    options.addOption(GAMMA_C_INIT_OPTION, true, "initial guess for gamma_c")
    options.addOption(GAMMA_X_INIT_OPTION, true, "initial guess for gamma_x")
    options.addOption(NUM_LATENT_FACTORS_OPTION, true, "number of latent factors")
    options.addOption(ISO_OPTION, false, "isolated learning for partitions")
    options.addOption(JAR_OPTION, true, "the path to find jar file")
    options.addOption(TMP_DIR_OPTION, true, 
        "local dir for tmp files, including map output files and RDDs stored on disk")
    options.addOption(MEMORY_OPTION, true, 
        "amount of memory to use per executor process")
    options.addOption(L1_REGULARIZATION, false, "use l1 regularization")
    options.addOption(NUM_ROWS_OPTION, true, "number of rows")
    options.addOption(NUM_COLS_OPTION, true, "number of cols")
    options.addOption(MULTICORE_OPTION, false, 
        "multicore computing on each machine")
    options.addOption(MAX_NORM_OPTION, false, "use max-norm regilarization")
    options.addOption(INTERVAL_OPTION, true, "interval to calculate testing RMSE")
    options.addOption(SYNTHETIC_DATA_OPTION, false, "using syn data")
    
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
    
    val mode = line.getOptionValue(RUNNING_MODE_OPTION)
    val trainingPath = line.getOptionValue(TRAINING_OPTION)
    val testingPath = line.getOptionValue(TESTING_OPTION)
    val jars = Seq(line.getOptionValue(JAR_OPTION))
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
    val numRows = 
      if (line.hasOption(NUM_ROWS_OPTION))
        line.getOptionValue(NUM_ROWS_OPTION).toInt
      else 5000000
    val numCols = 
      if (line.hasOption(NUM_COLS_OPTION))
        line.getOptionValue(NUM_COLS_OPTION).toInt
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
    val numRowBlocks = 
      if (line.hasOption(NUM_ROW_BLOCKS_OPTION)) 
        line.getOptionValue(NUM_ROW_BLOCKS_OPTION).toInt
      else 1
    val admm_c = admm && numRowBlocks>1
    val numColBlocks = 
      if (line.hasOption(NUM_COL_BLOCKS_OPTION)) 
        line.getOptionValue(NUM_COL_BLOCKS_OPTION).toInt
      else 1
    val admm_r = admm && numColBlocks>1
    val NUM_PARTITIONS = numRowBlocks*numColBlocks
    val numCores =
      if (line.hasOption(NUM_CORES_OPTION))
        line.getOptionValue(NUM_CORES_OPTION).toInt
      else NUM_PARTITIONS
    val updateGammaR = line.hasOption(UPDATE_GAMMA_R_OPTION)
    val updateGammaC = line.hasOption(UPDATE_GAMMA_C_OPTION)
    val gamma_r_init = 
      if (line.hasOption(GAMMA_R_INIT_OPTION)) 
        line.getOptionValue(GAMMA_R_INIT_OPTION).toFloat
      else 10f
    val gamma_c_init = 
      if (line.hasOption(GAMMA_C_INIT_OPTION)) 
        line.getOptionValue(GAMMA_C_INIT_OPTION).toFloat
      else 10f
    val gamma_x_init = 
      if (line.hasOption(GAMMA_X_INIT_OPTION)) 
        line.getOptionValue(GAMMA_X_INIT_OPTION).toFloat
      else 1f
    val lambda_r = 
      if (line.hasOption(LAMBDA_R_INIT_OPTION))
        line.getOptionValue(LAMBDA_R_INIT_OPTION).toFloat
      else 0f
    val lambda_c =
      if (line.hasOption(LAMBDA_C_INIT_OPTION))
        line.getOptionValue(LAMBDA_C_INIT_OPTION).toFloat
      else 0f
    val numFactors = 
      if (line.hasOption(NUM_LATENT_FACTORS_OPTION)) 
        line.getOptionValue(NUM_LATENT_FACTORS_OPTION).toInt
      else 20
    val numSlices =
      if (line.hasOption(NUM_SLICES_OPTION))
        math.min(line.getOptionValue(NUM_SLICES_OPTION).toInt, 
            numRowBlocks*numColBlocks)
      else numRowBlocks*numColBlocks
    val iso = line.hasOption(ISO_OPTION)
    val tmpDir = if (line.hasOption(TMP_DIR_OPTION))
      line.getOptionValue(TMP_DIR_OPTION)
      else "tmp"
    val memory = if (line.hasOption(MEMORY_OPTION))
      line.getOptionValue(MEMORY_OPTION)
      else "512m"
    val l1 = line.hasOption(L1_REGULARIZATION)
    val max_norm = line.hasOption(MAX_NORM_OPTION)
    val multicore = line.hasOption(MULTICORE_OPTION)
    val interval = 
      if (line.hasOption(INTERVAL_OPTION))
        line.getOptionValue(INTERVAL_OPTION).toInt
      else 1
    val syn = line.hasOption(SYNTHETIC_DATA_OPTION)
    
    var jobName = "DIST_MF"
    if (vb) jobName += "_VB" else jobName += "_MAP"
    if (admm) jobName += "_ADMM" 
    if (multicore) jobName += "_multicore"
    if (l1) jobName += "_l1"
    if (max_norm) jobName += "_max_norm"
    if (iso) jobName += "_iso"
    if (syn) jobName += "_M_" + numRows + "_N_" + numCols
    jobName +=  "_oi_" + maxOuterIter + "_ii_" + maxInnerIter + "_K_" + numFactors +
      "_rb_" + numRowBlocks + "_cb_" + numColBlocks +
      "_ugr_" + updateGammaR + "_ugc_" + updateGammaC + 
      "_gr_" + gamma_r_init + "_gc_" + gamma_c_init + 
      "_lr_" + lambda_r + "_lc_" + lambda_c
    val logPath = outputDir + jobName + ".txt"
    
    System.setProperty("spark.executor.memory", memory)
    System.setProperty("spark.local.dir", tmpDir)
    System.setProperty("spark.default.parallelism", numReducers.toString)
    System.setProperty("spark.storage.memoryFraction", "0.5")
    System.setProperty("spark.locality.wait", "10000")
    System.setProperty("spark.worker.timeout", "3600")
    System.setProperty("spark.storage.blockManagerSlaveTimeoutMs", "8000000")
    System.setProperty("spark.akka.timeout", "60")
    System.setProperty("spark.akka.askTimeout", "60")
//    System.setProperty("spark.serializer", 
//        "org.apache.spark.serializer.KryoSerializer")
//    System.setProperty("spark.kryo.registrator", "utilities.Registrator")
//    System.setProperty("spark.kryoserializer.buffer.mb", "64")
//    System.setProperty("spark.kryo.referenceTracking", "false")
//  System.setProperty("spark.mesos.coarse", "true")
//    System.setProperty("spark.cores.max", numCores.toString)
    
    val storageLevel = storage.StorageLevel.MEMORY_ONLY
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))
    val sc = new SparkContext(mode, jobName, System.getenv("SPARK_HOME"), jars)
    
    val rowPartitionMap = sc.broadcast(getPartitionMap(numRows, numRowBlocks, bwLog))
    val colPartitionMap = sc.broadcast(getPartitionMap(numCols, numColBlocks, bwLog))
    val dataPartitioner = new HashPartitioner(numSlices)
    val paraPartitioner = new HashPartitioner(numReducers)
    
    // read in the sparse matrix:
    val trainingTuples =
      if (syn || trainingPath.toLowerCase.contains("ml")) 
        sc.sequenceFile[NullWritable, Record](trainingPath)
          .map(pair => 
            new Record(pair._2.rowIdx, pair._2.colIdx, (pair._2.value-mean)/scale))
      else if (trainingPath.toLowerCase.contains("eachmovie"))
        sc.textFile(trainingPath, numSlices)
          .map(line => parseLine(line, " ", mean, scale))
      else 
        sc.textFile(trainingPath, numSlices)
          .flatMap(line => parseLine(line, mean, scale))
        
    val trainingData = getPartitionedData(trainingTuples, numColBlocks, 
        rowPartitionMap, colPartitionMap, dataPartitioner).persist(storageLevel)
    
    val samplesPerBlock = trainingData.map(pair => pair._2.col_idx.length).collect
    val rowsPerBlock = trainingData.map(pair => pair._2.row_ptr.length).collect
    val colsPerBlock = trainingData.map(pair => pair._2.col_ptr.length).collect
    val numTrainingSamples = samplesPerBlock.sum
    var b = 0
    println("samples per block:")
    while (b < numSlices) {
      print(b + ":" + samplesPerBlock(b) + "\t")
      b += 1
    }
    println
    println("rows per block:")
    b = 0
    while (b < numRowBlocks) {
      print(b + ":" + rowsPerBlock(b) + "\t")
      b += 1
    }
    println
    println("cols per block:")
    b = 0
    while (b < numColBlocks) {
      print(b + ":" + colsPerBlock(b) + "\t")
      b += 1
    }
    println
    val testingTuples = 
      if (syn || testingPath.toLowerCase.contains("ml"))
        sc.sequenceFile[NullWritable, Record](testingPath, numSlices).map(pair =>
          new Record(pair._2.rowIdx, pair._2.colIdx, (pair._2.value-mean)/scale))
      else if (testingPath.toLowerCase.contains("eachmovie"))
        sc.textFile(testingPath, numSlices)
          .map(line => parseLine(line, " ", mean, scale))
      else 
        sc.textFile(testingPath, numSlices)
          .flatMap(line => parseLine(line, mean, scale))  
    val testingData = getPartitionedData(testingTuples, numColBlocks, 
        rowPartitionMap, colPartitionMap, dataPartitioner)
    testingData.persist(storageLevel)
    val numTestingSamples = testingData.map(pair => pair._2.col_idx.length).sum
    
    println("Number of rows: " + numRows)
    println("Number of columns: " + numCols)
    println("Number of training samples: " + numTrainingSamples)
    println("Number of testing samples: " + numTestingSamples)    
    println("Mean: " + mean)
    println("Scale: " + scale)
    bwLog.write("Number of rows: " + numRows + "\n")
    bwLog.write("Number of columns: " + numCols + "\n")
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
      val gamma_r = if (vb) 1f else 1f
      val gamma_c = if (vb) 1f else 1f
      val gamma_x = 1f
      Model(data, numFactors, gamma_r, gamma_c, gamma_x, admm_r, admm_c, vb)
        .ccdpp(data, maxInnerIter, 1, thre, multicore, vb, false, false)
    }).persist(storageLevel)    
    
    var localModels = localOutputs.mapValues{
      case (model, iter, rmse, rowSD, colSD) => {
        model.setGammaC(gamma_c_init)
        model.setGammaR(gamma_r_init)
        model
      }
    }
    var localInfo = localOutputs.mapValues{
      case(model, iter, rmse, rowSD, colSD) => (iter, rmse, rowSD, colSD)
    }.collect
    
    val rowPrior = localModels.mapValues{
          model => model.rowMap.map(i=>(i, Array.ofDim[Float](model.numFactors)))
        }.persist(storageLevel)
    val colPrior = localModels.mapValues{
          model => model.colMap.map(i=>(i, Array.ofDim[Float](model.numFactors)))
        }.persist(storageLevel)
    
    while (iter < maxOuterIter) {
      val globalRowFactors = 
        if (numColBlocks>1 && !iso)
          updateGlobalPriors(localModels.flatMap{
            case(pid, model) =>
            timesGamma(model.getRowStats(admm_r), model.gamma_r, model.rowMap, pid)
          }, l1, max_norm, lambda_r, paraPartitioner)
          .groupByKey(dataPartitioner).mapValues(seq => (seq.toArray).sortBy(_._1))
        else rowPrior
      val globalColFactors = 
        if (numRowBlocks>1 && !iso)
          updateGlobalPriors(localModels.flatMap{
            case(pid, model) =>
            timesGamma(model.getColStats(admm_c), model.gamma_c, model.colMap, pid)
          }, l1, max_norm, lambda_c, paraPartitioner)
          .groupByKey(dataPartitioner).mapValues(seq => (seq.toArray).sortBy(_._1))
        else colPrior
      
      val priors = globalRowFactors.join(globalColFactors)
      if (!iso && (iter % interval == 0 || iter+1 == maxOuterIter)) {
        priors.persist(storageLevel)
        if (iter+1 == maxOuterIter && false) {
          priors.map{case(bid, (rowFactors, colFactors)) =>
            rowFactors.map{case(r, rowFactor) => 
              r + "\t" + rowFactor.mkString(" ")
            }.mkString("\n")
          }.saveAsTextFile(outputDir + "globalRowFactors")
          priors.map{case(bid, (rowFactors, colFactors)) =>
            colFactors.map{case(c, colFactor) => 
              c + "\t" + colFactor.mkString(" ")
            }.mkString("\n")
          }.saveAsTextFile(outputDir + "globalColFactors")
        }
      }
      
      val globalModel = 
        if (numRowBlocks>1 && numColBlocks>1) priors
        else
          localModels.join(priors).mapValues{
            case(model, prior) => Pair(
              if (numColBlocks == 1) toGlobal(model.rowFactor, model.rowMap)
              else prior._1,
              if (numRowBlocks == 1) toGlobal(model.colFactor, model.colMap)
              else prior._2
            )
          }
      
      val testingRMSE_global = 
        if (!iso && (iter % interval == 0 || iter+1 == maxOuterIter)) {
          scale*math.sqrt(testingData.join(globalModel).map{
            case(id, (data, (rowPrior, colPrior))) => data.getSE(
            Model.toLocal(data.rowMap, rowPrior), Model.toLocal(data.colMap, colPrior))
          }.reduce(_+_)/numTestingSamples)
        }
        else 0
        
      val testingRMSE_local = scale*math.sqrt(testingData.join(localModels).map{
        case(idx, (data, model)) => data.getSE(
        Model.toLocal(data.rowMap, toGlobal(model.rowFactor, model.rowMap)),
        Model.toLocal(data.colMap, toGlobal(model.colFactor, model.colMap)))
      }.reduce(_+_)/numTestingSamples)
      
      if (updateGammaR || updateGammaC) {
        val gammas = localModels.mapValues(model => 
          (Vector(model.gamma_r), Vector(model.gamma_c), model.gamma_x)).collect
        gammas.foreach{
          case(pid, (gamma_r, gamma_c, gamma_x)) => {
            val r = pid/numColBlocks; val c = pid%numColBlocks
            val msg = "partition(" + r + "," + c + ")" +"\ngamma_r: " + gamma_r +
              "\ngamma_c: " + gamma_c + "\ngamma_x: " + gamma_x
            println(msg); bwLog.write(msg + "\n")
          }
        } 
      }
      localInfo.foreach{
        case(pid, (iter, rmse, rowSD, colSD)) => {
          val r = pid/numColBlocks; val c = pid%numColBlocks
          val msg = "partition(" + r + "," + c + ")" + " num inner iters: " + iter + 
            " block training rmse: " + rmse + " row sd: " + rowSD + " col sd: " + colSD
          println(msg); bwLog.write(msg + "\n")
        }
      }
      val time = (System.currentTimeMillis() - iterTime)*0.001
      println("Testing RMSE: global " + testingRMSE_global + 
        " local " + testingRMSE_local)
      bwLog.write("Testing RMSE: global " + testingRMSE_global + 
        " local " + testingRMSE_local + "\n")
      println("Iter: " + iter + " finsied, time elapsed: " + time)
      bwLog.write("Iter: " + iter + " finished, time elapsed: " + time + "\n")
      iterTime = System.currentTimeMillis()
      iter += 1
      if (iter < maxOuterIter) {
        val thre = 0.001f
        val innerIter = maxInnerIter
//        val innerIter = math.max(maxInnerIter - 2*iter, 2)
        val rddId = localOutputs.id
        localOutputs = trainingData.join(localModels).join(priors).mapValues{
          case((data, model), (rowPriorsPairs, colPriorsPairs)) => {
            val rowPriors = Model.toLocal(data.rowMap, rowPriorsPairs)
            val colPirors = Model.toLocal(data.colMap, colPriorsPairs)
            model.ccdpp(data, innerIter, 1, thre, multicore, vb, admm_r, admm_c,
              updateGammaR, updateGammaC, rowPriors, colPirors)
          }
        }
        localOutputs.persist(storageLevel)
        localModels = localOutputs.mapValues(_._1)
        localInfo = localOutputs.mapValues{
          case(model, iter, rmse, rowSD, colSD) => (iter, rmse, rowSD, colSD)
        }.collect
        priors.unpersist(false)
        println("Let's remove RDD: " + rddId)
        sc.getPersistentRDDs(rddId).unpersist(false)
      }
    }
    if (false) {
      localModels.map{case (bid, model) => 
        model.rowFactor.map(_.mkString(" ")).mkString("\n")}
        .saveAsTextFile(outputDir + "localRowFactors")
      localModels.map{case (bid, model) => 
        model.colFactor.map(_.mkString(" ")).mkString("\n")}
        .saveAsTextFile(outputDir + "localColFactors")
    }
    bwLog.write("Total time elapsed " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    bwLog.close()
    println("Total time elapsed " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    System.exit(0)
  }
}