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
  val optType = CDPP
  val weightedReg = true
  val emBayes = false
  val regType = L2
  val regPara = 1f
  val numCores = 2
  val numRowBlocks = 2
  val numColBlocks = 2
  val gammaRInit = if (weightedReg) 0.1f else 10f
  val gammaCInit = if (weightedReg) 0.1f else 10f
  val numOuterIter = 10
  val maxInnerIter = 5
  val numFactors = 20
  val stopCrt = 0f
  //for each movie, numRows = 1621, numCols = 55423, mean: 4.037181f
  val numRows = if (seq) 5000 else 6041
  val numCols = if (seq) 5000 else 3953
  val mean = if (seq) 0f else 3.7668543f
  val scale = 1
  val numSlices = numRowBlocks*numColBlocks
  val mode = "local[" + numCores + "]"
  val jars = Seq("sparkproject.jar")
  val multicore = numCores >= 2*numSlices
  
  def main(args : Array[String]) {
    val startTime = System.currentTimeMillis()
    
//    val options = new Options()
//    options.addOption(HELP, false, "print the help message")
//    options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
//    options.addOption(TRAINING_OPTION, true, "training data input path/directory")
//    options.addOption(TESTING_OPTION, true, "testing data input path/directory")
//    options.addOption(OUTPUT_OPTION, true, "output path/directory")
//    options.addOption(MEAN_OPTION, true, "mean option for input values")
//    options.addOption(SCALE_OPTION, true, "scale option for input values")
//    options.addOption(NUM_REDUCERS_OPTION, true, "number of reducers to use")
//    options.addOption(NUM_SLICES_OPTION, true, "number of slices of the data")
//    options.addOption(OUTER_ITERATION_OPTION, true, "max number of outer iterations")
//    options.addOption(INNER_ITERATION_OPTION, true, "max number of inner iterations")
//    options.addOption(NUM_COL_BLOCKS_OPTION, true, "number of column blocks")
//    options.addOption(NUM_ROW_BLOCKS_OPTION, true, "number of row blocks")
//    options.addOption(REG_PARA_OPTION, true, "set the regularization parameter")
//    options.addOption(OPTIMIZER_OPTION, true, "optimizer type (e.g. ALS, CD, VBCD)")
//    options.addOption(REGULARIZER_OPTION, true, "regu type (e.g. L1, L2, Max)")
//    options.addOption(MODEL_OPTION, true, "model type  (e.g. admm, hecmem, dgm)")
//    options.addOption(GAMMA_R_INIT_OPTION, true, "initial guess for gammaR")
//    options.addOption(GAMMA_C_INIT_OPTION, true, "initial guess for gammaC")
//    options.addOption(NUM_LATENT_FACTORS_OPTION, true, "number of latent factors")
//    options.addOption(JAR_OPTION, true, "the path to find jar file")
//    options.addOption(TMP_DIR_OPTION, true, 
//        "local dir for tmp files, including mapoutput files and RDDs stored on disk")
//    options.addOption(MEM_OPTION, true, 
//        "amount of memory to use per executor process")
//    options.addOption(NUM_ROWS_OPTION, true, "number of rows")
//    options.addOption(NUM_COLS_OPTION, true, "number of cols")
//    options.addOption(NUM_CORES_OPTION, true, "number of cores to use")
//    options.addOption(MULTICORE_OPTION, false, 
//        "multicore computing on each machine")
//    options.addOption(INTERVAL_OPTION, true, "interval to calculate testing RMSE")
//    options.addOption(SEQUENCE_FILE_OPTION, false, "input is sequence file")
//    options.addOption(STOPPING_CRITERIA_OPTION, true, 
//        "stopping criteria for inner iterations")
//    options.addOption(MORE_PARA_OPTION, false, "gamma be row/document specific")
//    options.addOption(WEIGHTED_REG_OPTION, false, "weighted regularizer option")
//    options.addOption(EMPIRICAL_BAYES_OPTION, false, "empirical Bayes option")
//    
//    val parser = new GnuParser()
//    val formatter = new HelpFormatter()
//    val line = parser.parse(options, args)
//    if (line.hasOption(HELP) || args.length == 0) {
//      formatter.printHelp("Help", options)
//      System.exit(0)
//    }
//    assert(line.hasOption(RUNNING_MODE_OPTION), "spark cluster mode not specified")
//    assert(line.hasOption(TRAINING_OPTION), "training data path not specified")
//    assert(line.hasOption(TESTING_OPTION), "testing data path not specified")
//    assert(line.hasOption(JAR_OPTION), "running jar file path not specified")
//    assert(line.hasOption(MODEL_OPTION), "model type not specified")
//    
//    val mode = line.getOptionValue(RUNNING_MODE_OPTION)
//    val trainingDir = line.getOptionValue(TRAINING_OPTION)
//    val testingDir = line.getOptionValue(TESTING_OPTION)
//    val jars = Seq(line.getOptionValue(JAR_OPTION))
//    val outputDir = 
//      if (line.hasOption(OUTPUT_OPTION))
//        if (!line.getOptionValue(OUTPUT_OPTION).endsWith(PATH_SEPERATOR))
//          line.getOptionValue(OUTPUT_OPTION) + PATH_SEPERATOR
//        else
//          line.getOptionValue(OUTPUT_OPTION)
//      else
//        "output" + PATH_SEPERATOR
//    val modelType = {
//      val name = line.getOptionValue(MODEL_OPTION)
//      name.toLowerCase match {
//        case "avgm" => AVGM
//        case "admm" => ADMM
//        case "mem" => MEM
//        case "hmem" => hMEM
//        case "hecmem" => hecMEM
//        case "dvb" => dVB
//        case "dmap" => dMAP
//        case _ => {
//          assert(false, "unexpected model type: " + name)
//          null
//        }
//      }
//    }
//    val optType = 
//      if (line.hasOption(OPTIMIZER_OPTION)) {
//        val name = line.getOptionValue(OPTIMIZER_OPTION)
//        name.toLowerCase match {
//          case "cd" => CD
//          case "cdpp" => CDPP
//          case "als" => ALS
//          case _ => {
//            assert(false, "unexpected optimizer type: " + name)
//            null
//          }
//        }
//      }
//      else CDPP
//    val regType = 
//      if (line.hasOption(REGULARIZER_OPTION) 
//          && modelType != MEM && modelType != AVGM){
//        val name = line.getOptionValue(REGULARIZER_OPTION)
//        name.toLowerCase match {
//          case "l1" => L1
//          case "l2" => L2
//          case "max" => Max
//          case _ => {
//            assert(false, "unexpected regularizer type: " + name)
//            null
//          }
//        }
//      }
//      else L2
//    val regPara = 
//      if (line.hasOption(REG_PARA_OPTION) && modelType != MEM && modelType != AVGM)
//        line.getOptionValue(REG_PARA_OPTION).toFloat
//      else 0f
//    val weightedReg = line.hasOption(WEIGHTED_REG_OPTION)
//    val emBayes = line.hasOption(EMPIRICAL_BAYES_OPTION)
//    val mean = 
//      if (line.hasOption(MEAN_OPTION))
//        line.getOptionValue(MEAN_OPTION).toFloat
//      else 0f
//    val scale = 
//      if (line.hasOption(SCALE_OPTION))
//        line.getOptionValue(SCALE_OPTION).toFloat
//      else 1f
//    val numRows = 
//      if (line.hasOption(NUM_ROWS_OPTION))
//        line.getOptionValue(NUM_ROWS_OPTION).toInt
//      else 5000000
//    val numCols = 
//      if (line.hasOption(NUM_COLS_OPTION))
//        line.getOptionValue(NUM_COLS_OPTION).toInt
//      else 5000000
//    val numCores = 
//      if (line.hasOption(NUM_CORES_OPTION))
//        line.getOptionValue(NUM_CORES_OPTION).toInt
//      else 16
//    val numOuterIter =
//      if (line.hasOption(OUTER_ITERATION_OPTION))
//        line.getOptionValue(OUTER_ITERATION_OPTION).toInt
//      else 10
//    val maxInnerIter = 
//      if (line.hasOption(INNER_ITERATION_OPTION))
//        line.getOptionValue(INNER_ITERATION_OPTION).toInt
//      else 5
//    val numRowBlocks = 
//      if (line.hasOption(NUM_ROW_BLOCKS_OPTION)) 
//        line.getOptionValue(NUM_ROW_BLOCKS_OPTION).toInt
//      else 1
//    val numColBlocks = 
//      if (line.hasOption(NUM_COL_BLOCKS_OPTION)) 
//        line.getOptionValue(NUM_COL_BLOCKS_OPTION).toInt
//      else 1
//    val gammaRInit = 
//      if (line.hasOption(GAMMA_R_INIT_OPTION))
//        line.getOptionValue(GAMMA_R_INIT_OPTION).toFloat
//      else 10f
//    val gammaCInit = 
//      if (line.hasOption(GAMMA_C_INIT_OPTION))
//        line.getOptionValue(GAMMA_C_INIT_OPTION).toFloat
//      else 10f
//    val numFactors = 
//      if (line.hasOption(NUM_LATENT_FACTORS_OPTION)) 
//        line.getOptionValue(NUM_LATENT_FACTORS_OPTION).toInt
//      else 20
//    val numSlices =
//      if (line.hasOption(NUM_SLICES_OPTION))
//        line.getOptionValue(NUM_SLICES_OPTION).toInt
//      else numRowBlocks*numColBlocks
//    val multicore = line.hasOption(MULTICORE_OPTION) || 
//        (line.hasOption(NUM_CORES_OPTION) && numCores >= 2*numSlices)
//    val seq = line.hasOption(SEQUENCE_FILE_OPTION)
//    val stopCrt = 
//      if (line.hasOption(STOPPING_CRITERIA_OPTION)) {
//        line.getOptionValue(STOPPING_CRITERIA_OPTION).toFloat
//      }
//      else 0f
//    if (line.hasOption(TMP_DIR_OPTION)) {
//      System.setProperty("spark.local.dir", line.getOptionValue(TMP_DIR_OPTION))
//    }
//    if (line.hasOption(MEM_OPTION)) {
//      System.setProperty("spark.executor.memory", line.getOptionValue(MEM_OPTION))
//    }
//    if (line.hasOption(NUM_REDUCERS_OPTION) || line.hasOption(NUM_CORES_OPTION)) {
//      if (line.hasOption(NUM_REDUCERS_OPTION)) {
//        System.setProperty("spark.default.parallelism", 
//          line.getOptionValue(NUM_REDUCERS_OPTION))
//      }
//      else {
//        System.setProperty("spark.default.parallelism", (numCores*4).toString)
//      }
//    }
    
    val isEC = modelType match {
      case `hecMEM` | ADMM => true
      case _ => false
    }
    val isVB = modelType match {
      case MEM | `hMEM` | `hecMEM` | `dVB` => true
      case _ => false
    }
    val hasPrior = modelType match {
      case `hMEM` | `hecMEM` | ADMM => true
      case _ => false
    }
    
    var jobName = "MF_"
    modelType match {
      case ModelType.`dMAP` => jobName += "dMAP"
      case ModelType.`dVB` => jobName += "dVB"
      case ModelType.MEM => jobName += "MEM"
      case ModelType.`hMEM` => jobName += "hMEM"
      case ModelType.`hecMEM` => jobName += "hecMEM"
      case ModelType.AVGM => jobName += "AVGM"
      case ModelType.ADMM => jobName += "ADMM"
    }
    jobName +=  "_O_" + numOuterIter + "_I_" + maxInnerIter + "_K_" + numFactors
    modelType match {
      case `dMAP` | `dVB` => jobName += "_B_" + numSlices
      case _ => {
        jobName += "_RB_" + numRowBlocks + "_CB_" + numColBlocks + 
          "_GR_" + gammaRInit + "_GC_" + gammaCInit
      }
    }
    if (multicore) jobName += "_MC"
    optType match {
      case CD => jobName += "_CD"
      case CDPP => jobName += "_CD++"
      case ALS => jobName += "_ALS"
    }
    if (modelType != MEM && modelType != AVGM) {
      regType match {
        case L1 => jobName += "_L1"
        case L2 => jobName += "_L2"
        case Max => jobName += "_MAX"
      }
      jobName += "_REG_" + regPara
    }
    if (stopCrt > 0) jobName += "_CRT_" + stopCrt
    if (weightedReg) jobName += "_WR"
    if (emBayes) jobName += "_EB"
    val logPath = outputDir + jobName + "_LOG" + ".txt"
    val localRMSE = new Array[Double](numOuterIter)
    val globalRMSE = new Array[Double](numOuterIter)
    val times = new Array[Double](numOuterIter)
    val resultPath = outputDir + jobName + "_RESULT" + ".txt"
//    System.setProperty("spark.storage.memoryFraction", "0.5")
    
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))
    val bwResult = new BufferedWriter(new FileWriter(new File(resultPath)))
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
    
    var model = modelType match {
      case `dVB` | `dMAP` => DistributedGradient(sc, trainingDir, testingDir,
        mean, scale, seq, numSlices, numFactors, regPara, isVB, weightedReg)
      case _ => DivideAndConquer(sc, trainingDir, testingDir, numRows, numCols,
        numRowBlocks, numColBlocks, numCores, seq, mean, scale, numFactors, 
        gammaRInit, gammaCInit, isEC, hasPrior, isVB, emBayes, weightedReg, multicore, 
        bwLog)
    }
    
    bwLog.write("Preprocessing data finished in " 
        + (System.currentTimeMillis()-startTime)*0.001 + "(s)\n")
    println("Preprocessing data finished in " 
        + (System.currentTimeMillis()-startTime)*0.001 + "(s)")
    
    for (iter <- 0 until numOuterIter) {
      val iterTime = System.currentTimeMillis()
//      val numInnerIter = math.min(maxInnerIter, iter + 5)
      if (modelType != `dVB` && modelType != `dMAP`) model.setStopCrt(stopCrt)
      if (iter == 0) model = model.init(maxInnerIter, optType, regPara, regType) 
      else model = model.train(maxInnerIter, optType, regPara, regType)
      val testingRMSEGlobal = model.getValidatingRMSE(true)*scale
      val testingRMSELocal =
        modelType match {
          case `dMAP` | `dVB` => testingRMSEGlobal
          case _ => model.getValidatingRMSE(false)*scale
        }
      val factorsRL2Norm = model.getFactorsRL2Norm
      val factorsCL2Norm = model.getFactorsCL2Norm
      val time = (System.currentTimeMillis() - iterTime)*0.001
      localRMSE(iter) = testingRMSELocal
      globalRMSE(iter) = testingRMSEGlobal
      times(iter) = time
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
      if (modelType != `dMAP` && modelType != `dVB`) {
        val numIters = model.getNumIters
        println("num inner iters: " + numIters.mkString("(", ", ", ")"))
        bwLog.write("num inner iters: " + numIters.mkString("(", ", ", ")") + '\n')
        if (true || (iter == 0 && weightedReg)) {
          val gammaR = model.getGammaR
          val gammaC = model.getGammaC
          println("GammaR: " + gammaR.mkString("(", ", ", ")"))
          println("GammaC: " + gammaC.mkString("(", ", ", ")"))
          bwLog.write("GammaR: " + gammaR.mkString("(", ", ", ")") + '\n')
          bwLog.write("GammaC: " + gammaC.mkString("(", ", ", ")") + '\n')
        }
      }
    }
    bwLog.write("Total time elapsed " 
        + (System.currentTimeMillis()-startTime)*0.001 + "(s)")
    bwLog.close()
    bwResult.write(localRMSE.mkString("[", ", ", "];") + '\n')
    bwResult.write(globalRMSE.mkString("[", ", ", "];") + '\n')
    bwResult.write(times.mkString("[", ", ", "];") + '\n')
    bwResult.close()
    println("Total time elapsed " 
        + (System.currentTimeMillis()-startTime)*0.001 + "(s)")
    System.exit(0)
  }
}