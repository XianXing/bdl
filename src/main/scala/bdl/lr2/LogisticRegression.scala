package lr2

import java.io._

import org.apache.spark.{SparkContext, HashPartitioner, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.Logging

import org.apache.commons.cli._

import utilities.Settings._
import utilities.SparseVector
import utilities.SparseMatrix
import utilities.IntVector
import classification._
import preprocess.LR2._
import classification.OptimizerType._
import classification.RegularizerType._
import classification.ModelType._
import classification.VariationalType._

// the main driver function
object LogisticRegression extends Logging{
  //  val trainingDataDir = "input/KDDCUP2010/kdda"
//  val validatingDataDir = "input/KDDCUP2010/kdda.t"
  val trainingDataDir = "../datasets/UCI_Adult/a9a"
  val validatingDataDir = "../datasets/UCI_Adult/a9a.t"
  val modelType = ModelType.hMEM
  val optType = CD
  val regType = L2
  val varType = Taylor
  val subsampleRate = 0.0f
  val isBinary = true
  val isSeq = false
  val outputDir = "output/"
  val numCores = 1
  val numSlices = 10
  val mode = "local[" + numCores + "]"
  val maxOuterIter = 20
  val numInnerIter = 10
  val regPara = 0.1
  val gamma = 1.0
  val rho = 1.0
  val featureThre = 1
  val tmpDir = "tmp"
  val memory = "1g"
  val numReducers = 2*numCores
  val jars = Seq("sparkproject.jar")
  val stopCriteria = 1e-6
  
  def main(args : Array[String]) {
    val currentTime = System.currentTimeMillis()
    
    val options = new Options()
    options.addOption(HELP, false, "print the help message")
    options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
    options.addOption(TRAINING_OPTION, true, "training data path/directory")
    options.addOption(VALIDATING_OPTION, true, "validating data path/directory")
    options.addOption(OUTPUT_OPTION, true, "output path/directory")
    options.addOption(NUM_CORES_OPTION, true, "number of cores to use")
    options.addOption(NUM_REDUCERS_OPTION, true, "number of reducers to use")
    options.addOption(NUM_SLICES_OPTION, true, "number of slices of the data")
    options.addOption(OUTER_ITERATION_OPTION, true, "max number of outer iterations")
    options.addOption(INNER_ITERATION_OPTION, true, "max number of inner iterations")
    options.addOption(REG_PARA_OPTION, true, "set the regularization parameter")
    options.addOption(GAMMA_INIT_OPTION, true, "set gamma value")
    options.addOption(RHO_OPTION, true, "set rho value")
    options.addOption(JAR_OPTION, true, "the path to find jar file")
    options.addOption(TMP_DIR_OPTION, true, "the local dir for tmp files")
    options.addOption(MEM_OPTION, true, 
        "amount of memory to use per executor process")
    options.addOption(FEATURE_THRESHOLD_OPTION, true, 
        "threshold on the used features' frequences")
    options.addOption(BINARY_FEATURES_OPTION, false, "binary features option")
    options.addOption(TARGET_AUC_OPTION, true, "targeted AUC option")
    options.addOption(SUMSAMPLE_RATE_OPTION, true, "subsampling rate")
    options.addOption(MODEL_OPTION, true, 
        "model type  (e.g. avgm, savgm, admm, mem, hmem, hecmem, dgm)")
    options.addOption(OPTIMIZER_OPTION, true, "optimizer type (e.g. CG, LBFGS, CD)")
    options.addOption(REGULARIZER_OPTION, true, "regularizer type (e.g. L1, L2)")
    options.addOption(VARIATIONAL_OPTION, true, 
        "variational method type (e.g. taylor, jaakkola, bohning)")
    
    val parser = new GnuParser();
    val formatter = new HelpFormatter();
    val line = parser.parse(options, args);
    if (line.hasOption(HELP) || args.length == 0) {
      formatter.printHelp("Help", options);
      System.exit(0);
    }
    assert(line.hasOption(RUNNING_MODE_OPTION), "running mode not specified")
    assert(line.hasOption(TRAINING_OPTION), "training data path not specified")
    assert(line.hasOption(VALIDATING_OPTION), "testing data path not specified")
    assert(line.hasOption(JAR_OPTION), "jar file path not specified")
    assert(line.hasOption(MODEL_OPTION), "model type not specified")
    val mode = line.getOptionValue(RUNNING_MODE_OPTION)
    val trainingDataDir = line.getOptionValue(TRAINING_OPTION)
    val validatingDataDir = line.getOptionValue(VALIDATING_OPTION)
    val jars = Seq(line.getOptionValue(JAR_OPTION))
    val modelType = {
      val name = line.getOptionValue(MODEL_OPTION)
      name.toLowerCase match {
        case "avgm" => ModelType.AVGM
        case "savgm" => ModelType.sAVGM
        case "admm" => ModelType.ADMM
        case "mem" => ModelType.MEM
        case "hmem" => ModelType.hMEM
        case "hecmem" => ModelType.hecMEM
        case "dgm" => ModelType.dGM
        case _ => {
          assert(false, "unexpected model type: " + name)
          null
        }
      }
    }
    val subsampleRate = 
      if (line.hasOption(SUMSAMPLE_RATE_OPTION)) {
        modelType match {
          case `sAVGM` => line.getOptionValue(SUMSAMPLE_RATE_OPTION).toFloat
          case _ => 0
        }
      }
      else 0
    val optType = 
      if (line.hasOption(OPTIMIZER_OPTION)) {
        val name = line.getOptionValue(OPTIMIZER_OPTION)
        name.toLowerCase match {
          case "cd" => CD
          case "cg" => CG
          case "lbfgs" => LBFGS
          case _ => {
            assert(false, "unexpected optimizer type: " + name)
            null
          }
        }
      }
      else CD
    val regType = 
      if (line.hasOption(REGULARIZER_OPTION)){
        val name = line.getOptionValue(REGULARIZER_OPTION)
        name.toLowerCase match {
          case "l1" => L1
          case "l2" => L2
          case _ => {
            assert(false, "unexpected regularizer type: " + name)
            null
          }
        }
      }
      else L2
    val varType = if (line.hasOption(VARIATIONAL_OPTION)){
      val name = line.getOptionValue(VARIATIONAL_OPTION)
      name.toLowerCase match {
        case "taylor" => Taylor
        case "jaakkola" => Jaakkola
        case "bohning" => Bohning
        case _ => {
          assert(false, "unexpected variational method type: " + name)
          null
        }
      }
    }
    else Taylor
    val outputDir = 
      if (line.hasOption(OUTPUT_OPTION))
        if (!line.getOptionValue(OUTPUT_OPTION).endsWith(PATH_SEPERATOR))
          line.getOptionValue(OUTPUT_OPTION) + PATH_SEPERATOR
        else
          line.getOptionValue(OUTPUT_OPTION)
      else
        "output" + PATH_SEPERATOR
    val maxOuterIter =
      if (line.hasOption(OUTER_ITERATION_OPTION))
        line.getOptionValue(OUTER_ITERATION_OPTION).toInt
      else 10
    val numInnerIter = 
      if (line.hasOption(INNER_ITERATION_OPTION))
        line.getOptionValue(INNER_ITERATION_OPTION).toInt
      else 5
    val numCores =
      if (line.hasOption(NUM_CORES_OPTION))
        line.getOptionValue(NUM_CORES_OPTION).toInt
      else 1
    val numReducers =
      if (line.hasOption(NUM_REDUCERS_OPTION))
        line.getOptionValue(NUM_REDUCERS_OPTION).toInt
      else numCores
    val numSlices =
      if (line.hasOption(NUM_SLICES_OPTION))
        line.getOptionValue(NUM_SLICES_OPTION).toInt
      else numCores
    val gamma = 
      if (line.hasOption(GAMMA_INIT_OPTION)) 
        line.getOptionValue(GAMMA_INIT_OPTION).toDouble
      else 1
    val rho = 
      if (line.hasOption(RHO_OPTION)) line.getOptionValue(RHO_OPTION).toDouble
      else 1
    val regPara = 
      if (line.hasOption(REG_PARA_OPTION))
        line.getOptionValue(REG_PARA_OPTION).toDouble
      else 1
    val isBinary = line.hasOption(BINARY_FEATURES_OPTION)
    val isSeq = line.hasOption(SEQUENCE_FILE_OPTION)
    val featureThre = if (line.hasOption(FEATURE_THRESHOLD_OPTION))
      line.getOptionValue(FEATURE_THRESHOLD_OPTION).toInt
      else 0
    val targetAUC =
      if (line.hasOption(TARGET_AUC_OPTION))
        line.getOptionValue(TARGET_AUC_OPTION).toDouble
      else 0.9
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
    
    var jobName = "LR_"
    
    modelType match {
      case ModelType.`dGM` => jobName += "dGM"
      case ModelType.MEM => jobName += "MEM"
      case ModelType.`sMEM` => jobName += "sMEM"
      case ModelType.`hMEM` => jobName += "hMEM"
      case ModelType.`hecMEM` => jobName += "hecMEM"
      case ModelType.AVGM => jobName += "AVGM"
      case ModelType.`sAVGM` => jobName += "sAVGM"
      case ModelType.ADMM => jobName += "ADMM"
    }
    jobName += '_'
    modelType match {
      case ModelType.`hMEM` | ModelType.`hecMEM` | ModelType.ADMM => 
        jobName += "rho_" + rho + '_'
      case _ => null
    }
    modelType match {
      case ModelType.`hMEM` | ModelType.`hecMEM` => jobName += "ga_" + gamma + '_'
      case _ => null
    }
    if (subsampleRate > 0) jobName += "sr_" + subsampleRate + "_"
    optType match {
      case CD => jobName += "CD"
      case CG => jobName += "CG"
      case LBFGS => jobName += "LBFGS"
    }
    jobName += '_'
    regType match {
      case L1 => jobName += "L1"
      case L2 => jobName += "L2"
    }
    jobName += '_'
    if (jobName.contains("MEM")) {
      varType match {
        case Taylor => jobName += "Taylor"
        case Jaakkola => jobName += "Jaakkola"
        case Bohning => jobName += "Bohning"
      }
      jobName += '_'
    }
    jobName +=  "oi_" + maxOuterIter + "_ii_" + numInnerIter + "_reg_" + regPara  + 
      "_b_" + numSlices + "_th_" + featureThre
    val logPath = outputDir + jobName + ".txt"
    
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))    
    val conf = new SparkConf()
//      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      .set("spark.kryo.registrator",  classOf[utilities.Registrator].getName)
//      .set("spark.kryo.referenceTracking", "false")
//      .set("spark.kryoserializer.buffer.mb", "8")
      .set("spark.locality.wait", "10000")
      .set("spark.akka.frameSize", "64")
//      .set("spark.storage.memoryFraction", "0.5")
      .setJars(jars)
      .setSparkHome(System.getenv("SPARK_HOME"))
    val sc = new SparkContext(mode, jobName, conf)
    
    val featurePartitioner = new HashPartitioner(numReducers)
    val dataPartitioner = new HashPartitioner(numSlices)
    val rawTrainingData = sc.textFile(trainingDataDir)
    // to filter out infrequent features
    val featureSet = 
      if (featureThre > 0) {
        rawTrainingData.flatMap(countLine(_, featureThre/2)).reduceByKey(_+_)
        .filter(_._2 >= featureThre).map(_._1).collect.sorted
      }
      else null
    
    val featureMapBC = 
      if (featureThre > 0) sc.broadcast(featureSet.zipWithIndex.toMap)
      else null
    
    val trainingData = toSparseMatrix(trainingDataDir, sc, featureMapBC, 
        dataPartitioner, numSlices, featureThre, isBinary, isSeq).cache
    
    def hash(x: Int): Int = {
      val r = x ^ (x >>> 20) ^ (x >>> 12)
      r ^ (r >>> 7) ^ (r >>> 4)
    }
    val seed = hash(1234567890)
    
    val subsampledTrainingData = toSparseMatrix(trainingDataDir, sc, featureMapBC, 
        dataPartitioner, numSlices, featureThre, isBinary, isSeq, subsampleRate, seed)
        .cache
    
    val splitsStats = trainingData.mapValues{
      case(responses, features) => (responses.length, features.rowMap.length)
    }.collect
    
    val validatingData = toSparseVector(validatingDataDir, sc, featureMapBC, 
      numSlices, featureThre, isBinary, isSeq).map(_._2).cache
      
    val valiPos = validatingData.filter(_._1 == 1).map(_._2).cache
    val valiNeg = validatingData.filter(_._1 == -1).map(_._2).cache
    
    val numFeatures = 
      if (featureThre > 0) featureSet.size + 1 //+1 because of the intercept
      else trainingData.map(_._2._2.rowMap.last).reduce(math.max(_,_)) + 1
    val isSparse = splitsStats.forall(pair => pair._2._2 < 0.3*numFeatures)
    val numTrain = splitsStats.map(_._2._1).reduce(_+_)
    val nnzTrain = trainingData.map(_._2._2.col_idx.length.toLong).reduce(_+_)
    val numVali = validatingData.count.toInt
    val numValiPos = valiPos.count.toInt
    val numValiNeg = valiNeg.count.toInt
    val nnzValidate = validatingData.map(_._2.size).reduce(_+_)
    
    logInfo("#features: " + numFeatures + "; #training data " + numTrain + 
      "; training nnz: " + nnzTrain +"; #validate data " + numVali + " (+" + 
      numValiPos + "," + numValiNeg + "), validate nnz: " + nnzValidate)
    bwLog.write("#features: " + numFeatures + "; #training data " + numTrain + 
      "; training nnz:" + nnzTrain +"; #validate data " + numVali + " (" + 
      numValiPos + "," + numValiNeg + "), validate nnz: " + nnzValidate + '\n')
    splitsStats.foreach(pair => logInfo("partition " + pair._1 + 
      " has " + pair._2._1 + " samples and " + pair._2._2 + " features"))
    logInfo("sparse update: " + isSparse)
    bwLog.write("sparse update: " + isSparse + "\n")            
    //initialization
    val featureCount = modelType match {
      case ModelType.AVGM | ModelType.MEM | ModelType.`sAVGM` | ModelType.ADMM =>
        trainingData.map{
          case (id, (responses, features)) => 
            IntVector(features.rowMap, numFeatures)
        }.reduce(_+=_).toArray
      case _ => null
    }
    val weightsInit = new Array[Double](numFeatures)
    var localModels = trainingData.mapValues(data => {
      LocalModel(data._2.numRows, modelType, gamma)
    }).cache
    var model = {
      modelType match {
        case ModelType.`dGM` => new DistributedGradient(weightsInit)
        case ModelType.MEM => new MEM(weightsInit, localModels, featureCount)
        case ModelType.AVGM => new AVGM(weightsInit, localModels, featureCount)
        case ModelType.`sAVGM` => new AVGM(weightsInit, localModels, featureCount)
        case ModelType.`hMEM` => new HMEM(weightsInit, localModels, rho, false)
        case ModelType.`hecMEM` => new HMEM(weightsInit, localModels, rho, true)
        case ModelType.ADMM => new ADMM(weightsInit, localModels, featureCount, rho)
      }
    }
    
    var subsampledModel = modelType match {
      case ModelType.`sAVGM` => new AVGM(weightsInit, localModels, featureCount)
      case _ => null
    }
    logInfo("Time elapsed after preprocessing " + 
      (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    bwLog.write("Time elapsed after preprocessing " + 
      (System.currentTimeMillis()-currentTime)*0.001 + "(s)\n")
      
    var iter = 0
    var iterTime = System.currentTimeMillis()
    var auc = 0.0
    var old_auc = -1.0
    while (iter < maxOuterIter && math.abs(auc-old_auc) > stopCriteria) {
      model = model.train(trainingData, numInnerIter, optType, regPara, regType)
      val weights = 
        if (subsampleRate > 0) {
          val weights1 = model.weights
          val weights2 = subsampledModel.train(subsampledTrainingData, 
            numInnerIter, optType, regPara, regType).weights
          Optimizers.sAVGMUpdate(weights1, weights2, subsampleRate)
        }
        else model.weights
      old_auc = auc
      auc = Functions.calculateAUC(sc, valiPos, numValiPos, valiNeg, numValiNeg, 
        weights)
      val llh = Functions.calculateLLH(sc, validatingData, weights)
      val obj = Functions.calculateOBJ(sc, trainingData, weights, regPara, regType)
      val aveL1Norm = Functions.getL1Norm(weights)/weights.length
      val time = (System.currentTimeMillis() - iterTime)*0.001
      logInfo("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc +
          " llh: " + llh + " obj: " + obj)
      logInfo("Ave L1 norm: " + aveL1Norm)
      bwLog.write("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc +
        " llh: " + llh + " obj: " + obj + '\n')
      bwLog.write("Ave L1 norm: " + aveL1Norm + '\n')
      modelType match {
        case ModelType.`hMEM`|ModelType.`hecMEM` => {
          val gamma = model.getGamma.mkString(" ")
          logInfo("gamma:\t" +  gamma) 
          bwLog.write("gamma:\t" + gamma + '\n')
          logInfo("rho:\t" +  model.getRho) 
          bwLog.write("rho:\t" + model.getRho + '\n')
        }
        case _ => null
      }
      iterTime = System.currentTimeMillis()
      iter += 1
    }
    if (numFeatures < 200) {
      val str = model.weights.mkString(" ")
      println("global: " + str)
      bwLog.write("global: " + str + '\n')
    }
    bwLog.write("Total time elapsed " + 
        (System.currentTimeMillis()-currentTime)*0.001 + "(s)\n")
    bwLog.close()
    println("Total time elapsed " + 
        (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    System.exit(0)
  }
}