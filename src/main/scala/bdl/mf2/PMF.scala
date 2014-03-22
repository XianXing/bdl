package mf2

import java.io._

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.serializer.KryoRegistrator

import org.apache.commons.cli._

import OptimizerType._
import RegularizerType._
import ModelType._
import utilities._
import utilities.Settings._
import preprocess.MF._

object PMF {
  
  val seq = false
  val trainingDir = 
    if (seq) {
      "output/train_mat_br_4_bc_4_Syn_M_5000_N_5000_K_10_spa_0.1_tr_0.9_lam_100.0" 
    }
    else "input/ml-1m/mf_train"
  val testingDir = 
    if (seq) "output/test_Syn_M_5000_N_5000_K_10_spa_0.1_tr_0.9_lam_100.0" 
    else "input/ml-1m/mf_test"
  val outputDir = "output/"
  val modelType = ADMM
  val optType = CD
  val regType = Trace
  val regPara = 1f
  val isVB = true
  val ec = true
  val numCores = 1
  val numRowBlocks = 1
  val numColBlocks = 4
  val gammaRInit = 1f
  val gammaCInit = 1f
  val numOuterIter = 10
  val numInnerIter = 10
  val numFactors = 20
  
  //for each movie, numRows = 1621, numCols = 55423, mean: 4.037181f
  val numRows = if (seq) 5000 else 6041
  val numCols = if (seq) 5000 else 3953
  val mean = if (seq) 0f else 3.7668543f
  val scale = 1
  val numSlices = numRowBlocks*numColBlocks
  val ecR = ec && numColBlocks > 1
  val ecC = ec && numRowBlocks > 1
  val mode = "local[" + numCores + "]"
  val jars = Seq("sparkproject.jar")
  val multicore = numCores > numSlices
  val interval = 1
  
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
    options.addOption(NUM_REDUCERS_OPTION, true, "number of reducers to use")
    options.addOption(NUM_SLICES_OPTION, true, "number of slices of the data")
    options.addOption(OUTER_ITERATION_OPTION, true, "max number of outer iterations")
    options.addOption(INNER_ITERATION_OPTION, true, "max number of inner iterations")
    options.addOption(VB_INFERENCE_OPTION, false, "Variational Bayesian inference")
    options.addOption(ADMM_OPTION, false, "Equality constraint option")
    options.addOption(NUM_COL_BLOCKS_OPTION, true, "number of column blocks")
    options.addOption(NUM_ROW_BLOCKS_OPTION, true, "number of row blocks")
    options.addOption(REG_PARA_OPTION, true, "set the regularization parameter")
    options.addOption(OPTIMIZER_OPTION, true, "optimizer type (e.g. ALS, CD)")
    options.addOption(REGULARIZER_OPTION, true, "regularizer type (e.g. Trace, Max)")
    options.addOption(MODEL_OPTION, true, "model type  (e.g. dac, dgm)")
    options.addOption(GAMMA_R_INIT_OPTION, true, "initial guess for gammaR")
    options.addOption(GAMMA_C_INIT_OPTION, true, "initial guess for gammaC")
    options.addOption(NUM_LATENT_FACTORS_OPTION, true, "number of latent factors")
    options.addOption(JAR_OPTION, true, "the path to find jar file")
    options.addOption(TMP_DIR_OPTION, true, 
        "local dir for tmp files, including mapoutput files and RDDs stored on disk")
    options.addOption(MEM_OPTION, true, 
        "amount of memory to use per executor process")
    options.addOption(NUM_ROWS_OPTION, true, "number of rows")
    options.addOption(NUM_COLS_OPTION, true, "number of cols")
    options.addOption(NUM_CORES_OPTION, true, "number of cores to use")
    options.addOption(MULTICORE_OPTION, false, 
        "multicore computing on each machine")
    options.addOption(INTERVAL_OPTION, true, "interval to calculate testing RMSE")
    options.addOption(SEQUENCE_FILE_OPTION, false, "input is sequence file")
    
    val parser = new GnuParser()
    val formatter = new HelpFormatter()
    val line = parser.parse(options, args)
    if (line.hasOption(HELP) || args.length == 0) {
      formatter.printHelp("Help", options)
      System.exit(0)
    }
    assert(line.hasOption(RUNNING_MODE_OPTION), "spark cluster mode not specified")
    assert(line.hasOption(TRAINING_OPTION), "training data path not specified")
    assert(line.hasOption(TESTING_OPTION), "testing data path not specified")
    assert(line.hasOption(JAR_OPTION), "running jar file path not specified")
    
    val mode = line.getOptionValue(RUNNING_MODE_OPTION)
    val trainingDir = line.getOptionValue(TRAINING_OPTION)
    val testingDir = line.getOptionValue(TESTING_OPTION)
    val jars = Seq(line.getOptionValue(JAR_OPTION))
    val outputDir = 
      if (line.hasOption(OUTPUT_OPTION))
        if (!line.getOptionValue(OUTPUT_OPTION).endsWith(PATH_SEPERATOR))
          line.getOptionValue(OUTPUT_OPTION) + PATH_SEPERATOR
        else
          line.getOptionValue(OUTPUT_OPTION)
      else
        "output" + PATH_SEPERATOR
    val modelType = 
      if (line.hasOption(MODEL_OPTION)) {
        line.getOptionValue(MODEL_OPTION).toLowerCase match {
          case "dgm" => `dGM`
          case _ => ADMM
        }
      }
      else ADMM
    val optType = 
      if (line.hasOption(OPTIMIZER_OPTION)) {
        val name = line.getOptionValue(OPTIMIZER_OPTION)
        name.toLowerCase match {
          case "cd" => CD
          case "cdpp" => CDPP
          case "als" => ALS
          case _ => {
            assert(false, "unexpected optimizer type: " + name)
            null
          }
        }
      }
      else CDPP
    val regType = 
      if (line.hasOption(REGULARIZER_OPTION)){
        val name = line.getOptionValue(REGULARIZER_OPTION)
        name.toLowerCase match {
          case "trace" => Trace
          case "max" => Max
          case _ => {
            assert(false, "unexpected regularizer type: " + name)
            null
          }
        }
      }
      else Trace
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
    val numCores = 
      if (line.hasOption(NUM_CORES_OPTION))
        line.getOptionValue(NUM_CORES_OPTION).toInt
      else 16
    val numOuterIter =
      if (line.hasOption(OUTER_ITERATION_OPTION))
        line.getOptionValue(OUTER_ITERATION_OPTION).toInt
      else 10
    val numInnerIter = 
      if (line.hasOption(INNER_ITERATION_OPTION))
        line.getOptionValue(INNER_ITERATION_OPTION).toInt
      else 5
    val isVB = line.hasOption(VB_INFERENCE_OPTION)
    val ec = line.hasOption(ADMM_OPTION)
    val numRowBlocks = 
      if (line.hasOption(NUM_ROW_BLOCKS_OPTION)) 
        line.getOptionValue(NUM_ROW_BLOCKS_OPTION).toInt
      else 1
    val ecC = ec && numRowBlocks>1
    val numColBlocks = 
      if (line.hasOption(NUM_COL_BLOCKS_OPTION)) 
        line.getOptionValue(NUM_COL_BLOCKS_OPTION).toInt
      else 1
    val ecR = ec && numColBlocks>1
    val gammaRInit = 
      if (line.hasOption(GAMMA_R_INIT_OPTION))
        line.getOptionValue(GAMMA_R_INIT_OPTION).toFloat
      else 10f
    val gammaCInit = 
      if (line.hasOption(GAMMA_C_INIT_OPTION))
        line.getOptionValue(GAMMA_C_INIT_OPTION).toFloat
      else 10f
    val regPara = 
      if (line.hasOption(REG_PARA_OPTION))
        line.getOptionValue(REG_PARA_OPTION).toFloat
      else 0f
    val numFactors = 
      if (line.hasOption(NUM_LATENT_FACTORS_OPTION)) 
        line.getOptionValue(NUM_LATENT_FACTORS_OPTION).toInt
      else 20
    val numSlices =
      if (line.hasOption(NUM_SLICES_OPTION))
        line.getOptionValue(NUM_SLICES_OPTION).toInt
      else numRowBlocks*numColBlocks
    val multicore = line.hasOption(MULTICORE_OPTION) || 
        (line.hasOption(NUM_CORES_OPTION) && numCores > 2*numSlices)
    val seq = line.hasOption(SEQUENCE_FILE_OPTION)
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
          line.getOptionValue(NUM_CORES_OPTION)*4)
      }
    }
    
    
    var jobName = "MF"
    modelType match {
      case `dGM` => jobName += "_dgm_b_" + numSlices
      case _ => {
        jobName += "_dac"
        if (ec) jobName += "_ec"
        jobName += "_rb_" + numRowBlocks + "_cb_" + numColBlocks + 
          "_gr_" + gammaRInit + "_gc_" + gammaCInit
      }
    }
    jobName +=  "_oi_" + numOuterIter + "_ii_" + numInnerIter + "_K_" + numFactors 
    if (isVB) jobName += "_vb" else jobName += "_map"
    if (multicore) jobName += "_mc"
    optType match {
      case CD => jobName += "_cd"
      case CDPP => jobName += "_cdpp"
      case ALS => jobName += "_als"
    }
    regType match {
      case Trace => jobName += "_trace"
      case Max => jobName += "_max"
    }
    jobName += "_reg_" + regPara
    if (multicore) jobName += "_multicore"
    
    val logPath = outputDir + jobName + ".txt"
    
//    System.setProperty("spark.storage.memoryFraction", "0.5")
    
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))
    val conf = new SparkConf()
//      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      .set("spark.kryo.registrator",  classOf[utilities.Registrator].getName)
//      .set("spark.kryo.referenceTracking", "false")
//      .set("spark.kryoserializer.buffer.mb", "64")
      .set("spark.locality.wait", "10000")
      .set("spark.akka.frameSize", "64")
      .setJars(jars)
//      .setSparkHome(System.getenv("SPARK_HOME"))
    val sc = new SparkContext(mode, jobName, conf)
    
    val emBayes = isVB
    var model = modelType match {
      case `dGM` => DistributedGradient(sc, trainingDir, testingDir,
      mean, scale, seq, numSlices, numFactors, regPara, isVB)
      case _ => DivideAndConquer(sc, trainingDir, testingDir, numRows, numCols,
        numRowBlocks, numColBlocks, numCores, seq, mean, scale, numFactors, 
        gammaRInit, gammaCInit, ecR, ecC, isVB, emBayes, bwLog)
    }
    
    bwLog.write("Preprocesing data finished in " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)\n")
    println("Preprocesing data finished in " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    
    for (iter <- 0 until numOuterIter) {
      val iterTime = System.currentTimeMillis()
      if (iter == 0) model = model.init(numInnerIter, optType, regPara, regType) 
      else model = model.train(numInnerIter, optType, regPara, regType)
      val testingRMSEGlobal = model.getValidatingRMSE(true)*scale
      val testingRMSELocal =
        modelType match {
          case `dGM` => testingRMSEGlobal
          case _ => model.getValidatingRMSE(false)*scale
        }
      val factorsRL2Norm = model.getFactorsRL2Norm
      val factorsCL2Norm = model.getFactorsCL2Norm
      val gammaR = model.getGammaR
      val gammaC = model.getGammaC
      val time = (System.currentTimeMillis() - iterTime)*0.001
      println("Iter: " + iter + " finsied, time elapsed: " + time)
      println("Testing RMSE: global " + testingRMSEGlobal + 
        " local " + testingRMSELocal)
      println("FactorsR L2 norm: " + factorsRL2Norm + 
        " , FactorsC L2 norm: " + factorsCL2Norm)
      bwLog.write("Iter: " + iter + " finished, time elapsed: " + time + "\n")
      bwLog.write("Testing RMSE: global " + testingRMSEGlobal + 
        " local " + testingRMSELocal + "\n")
      bwLog.write("FactorsR L2 norm: " + factorsRL2Norm + 
        " , FactorsC L2 norm: " + factorsCL2Norm + "\n")
      if (isVB && modelType != `dGM`) {
        println("GammaR: " + gammaR.mkString("(", ", ", ")"))
        println("GammaC: " + gammaC.mkString("(", ", ", ")"))
        bwLog.write("GammaR: " + gammaR.mkString("(", ", ", ")") + '\n')
        bwLog.write("GammaC: " + gammaC.mkString("(", ", ", ")") + '\n')
      }
    }
    bwLog.write("Total time elapsed " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    bwLog.close()
    println("Total time elapsed " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    System.exit(0)
  }
}