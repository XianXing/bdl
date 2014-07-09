package tm

import java.io._
import scala.io._

import org.apache.spark.{SparkContext, SparkConf, HashPartitioner}
import org.apache.spark.serializer.KryoRegistrator

import org.apache.commons.cli._

import ModelType._
import utilities.Settings._

object TM {
    
  val prefix = "/Users/xianxingzhang/Documents/workspace/datasets/Bags_Of_Words/"
  val trainingDir = prefix + "nips_processed/train"
  val validatingDir = prefix + "nips_processed/validate"
  val dictPath = prefix + "vocab.nips.txt"
  val outputDir = "output/LDA/NIPS/"
//  val trainingDir = prefix + "kos_processed/train"
//  val validatingDir = prefix + "kos_processed/validate"
//  val dictPath = prefix + "vocab.kos.txt"
//  val outputDir = "output/LDA/KOS/"
  val numCores = 2
  val numBlocks = 4
  val ratio = numBlocks/numBlocks
//  val modelType = ModelType.HDLDA
  val modelType = ModelType.VADMM
//  val modelType = ModelType.MRLDA
  val numTopics = 20
  val alphaInit = 1.0
  val emBayes = false
  val beta0 = 0.1
  val numOuterIters = 20
  val numInnerIters = 10
  val mode = "local[" + numCores + "]"
  val jars = Seq("sparkproject.jar")
  val multicore = numCores >= 2*numBlocks
  
  def main(args : Array[String]) {
    
    val startTime = System.currentTimeMillis()
    
    DivideAndConquer.calculate
    
//    val options = new Options()
//    options.addOption(HELP, false, "print the help message")
//    options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
//    options.addOption(TRAINING_OPTION, true, "path/directory to training data")
//    options.addOption(VALIDATING_OPTION, true, "path/directory to validating data")
//    options.addOption(DICTIONARY_PATH_OPTION, true, "path to dictionary")
//    options.addOption(OUTPUT_OPTION, true, "output path/directory")
//    options.addOption(NUM_REDUCERS_OPTION, true, "number of reducers to use")
//    options.addOption(NUM_SLICES_OPTION, true, "number of slices of the data")
//    options.addOption(OUTER_ITERATION_OPTION, true, "number of outer iterations")
//    options.addOption(INNER_ITERATION_OPTION, true, "number of inner iterations")
//    options.addOption(ALPHA_OPTION, true, "initialize parameter alpha")
//    options.addOption(BETA_OPTION, true, "initialize parameter beta0")
//    options.addOption(MODEL_OPTION, true, "model type  (e.g. vadmm, mrlda)")
//    options.addOption(NUM_TOPICS_OPTION, true, "number of topics")
//    options.addOption(JAR_OPTION, true, "the path to find jar file")
//    options.addOption(TMP_DIR_OPTION, true, 
//        "local dir for tmp files, including mapoutput files and RDDs stored on disk")
//    options.addOption(MEM_OPTION, true, 
//        "amount of memory to use per executor process")
//    options.addOption(MULTICORE_OPTION, false, 
//        "multicore computing on each machine (only for vadmm)")
//    options.addOption(EMPIRICAL_BAYES_OPTION, false, "updating alpha option")
//    options.addOption(INIT_RATIO_OPTION, true, 
//        "ratio of training data used for initialization")
//    
//    val parser = new GnuParser()
//    val formatter = new HelpFormatter()
//    val line = parser.parse(options, args)
//    if (line.hasOption(HELP) || args.length == 0) {
//      formatter.printHelp("Help", options)
//      System.exit(0)
//    }
//    assert(line.hasOption(RUNNING_MODE_OPTION), "spark cluster mode not specified")
//    assert(line.hasOption(TRAINING_OPTION), "path to training data not specified")
//    assert(line.hasOption(VALIDATING_OPTION), "path to validating data not specified")
//    assert(line.hasOption(DICTIONARY_PATH_OPTION), "path to dictionary not specified")
//    assert(line.hasOption(JAR_OPTION), "running jar file path not specified")
//    assert(line.hasOption(MODEL_OPTION), "model type not specified")
//    
//    val mode = line.getOptionValue(RUNNING_MODE_OPTION)
//    val trainingDir = line.getOptionValue(TRAINING_OPTION)
//    val validatingDir = line.getOptionValue(VALIDATING_OPTION)
//    val dictPath = line.getOptionValue(DICTIONARY_PATH_OPTION)
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
//        case "mrlda" => ModelType.MRLDA
//        case "vadmm" => ModelType.VADMM
//        case "hdlda" => ModelType.HDLDA
//        case _ => {
//          assert(false, "unexpected model type: " + name)
//          null
//        }
//      }
//    }
//    val emBayes = line.hasOption(EMPIRICAL_BAYES_OPTION)
//    val numCores = 
//      if (line.hasOption(NUM_CORES_OPTION))
//        line.getOptionValue(NUM_CORES_OPTION).toInt
//      else 16
//    val numOuterIters =
//      if (line.hasOption(OUTER_ITERATION_OPTION))
//        line.getOptionValue(OUTER_ITERATION_OPTION).toInt
//      else 10
//    val numInnerIters = 
//      if (line.hasOption(INNER_ITERATION_OPTION))
//        line.getOptionValue(INNER_ITERATION_OPTION).toInt
//      else 5
//    val numTopics = 
//      if (line.hasOption(NUM_TOPICS_OPTION)) 
//        line.getOptionValue(NUM_TOPICS_OPTION).toInt
//      else 20
//    val numBlocks =
//      if (line.hasOption(NUM_SLICES_OPTION))
//        line.getOptionValue(NUM_SLICES_OPTION).toInt
//      else numCores
//    val alphaInit = 
//      if (line.hasOption(ALPHA_OPTION))
//        line.getOptionValue(ALPHA_OPTION).toDouble
//      else 50.0/numTopics
//    val beta0 =
//      if (line.hasOption(BETA_OPTION))
//        line.getOptionValue(BETA_OPTION).toDouble
//      else 0.01
//    val multicore = line.hasOption(MULTICORE_OPTION) || 
//        (line.hasOption(NUM_CORES_OPTION) && numCores >= 2*numBlocks)
//    val ratio = 
//      if (line.hasOption(INIT_RATIO_OPTION)) {
//        line.getOptionValue(INIT_RATIO_OPTION).toDouble
//      }
//      else 1.0*numCores/(numBlocks*numBlocks)
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
    
    val jobName = modelType + "_B_" + numBlocks + "_T_" + numTopics + "_OI_" + 
      numOuterIters + "_II_" + numInnerIters + "_EB_" + emBayes + 
      "_beta0_" + beta0 + "_alpha_" + alphaInit + "_initRatio_" + ratio
    val outputPrefix = outputDir + jobName
    val validatePerps = new Array[Double](numOuterIters)
    val validateLLHs = new Array[Double](numOuterIters)
    val times = new Array[Double](numOuterIters)
    val conf = new SparkConf()
//      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      .set("spark.kryo.registrator",  classOf[utilities.Registrator].getName)
//      .set("spark.kryo.referenceTracking", "false")
//      .set("spark.kryoserializer.buffer.mb", "64")
      .set("spark.locality.wait", "10000")
      .set("spark.akka.frameSize", "64")
      .setJars(jars)
    
    val sc = new SparkContext(mode, jobName, conf)
    val partitioner = new HashPartitioner(numBlocks)
    val dict = Source.fromFile(dictPath).getLines.toArray
    val numWords = dict.length
    
    val seed = 1234567890
    var model = modelType match {
      case ModelType.VADMM => {
        val hasEC = true
        DivideAndConquer(sc, partitioner, trainingDir, validatingDir, numTopics, 
          numWords, numBlocks, ratio, seed, alphaInit, beta0, hasEC, multicore)
      }
       case ModelType.HDLDA => {
        val hasEC = false
        DivideAndConquer(sc, partitioner, trainingDir, validatingDir, numTopics, 
          numWords, numBlocks, ratio, seed, alphaInit, beta0, hasEC, multicore)
      }
      case ModelType.MRLDA => MRLDA(sc, partitioner, trainingDir, validatingDir, 
          numTopics, numWords, numBlocks, seed, alphaInit, beta0)
    }
    
    for (iter <- 0 until numOuterIters) {
      val iterTime = System.currentTimeMillis()
      model = model.train(numInnerIters, emBayes)
      val time = (System.currentTimeMillis - iterTime)*0.001
      times(iter) = time
      validatePerps(iter) = model.getValidatingPerplexity(100)
      validateLLHs(iter) = model.getValidatingLLH(100)
      println("Iter: " + iter + ", time elapsed: " + time + 
          ", validating Perplexity: " + validatePerps(iter) + 
          ", validating LLH: " + validateLLHs(iter))
    }
    
    val resultPath = outputPrefix + "_Results.txt"
    val perpLogger = new BufferedWriter(new FileWriter(new File(resultPath)))
    perpLogger.write(times.mkString("[", ", ", "];") + '\n')
    perpLogger.write(validatePerps.mkString("[", ", ", "];") + '\n')
    perpLogger.write(validateLLHs.mkString("[", ", ", "];") + '\n')
    perpLogger.close
    val topicsFileName = outputPrefix + "_Topics.txt"
    val topicsLogger = new BufferedWriter(new FileWriter(new File(topicsFileName)))
    LDA.printTopics(model.topics, dict, 10, topicsLogger)
    topicsLogger.close
    println("Total time elapsed " + (System.currentTimeMillis-startTime)*0.001 + "(s)")
    System.exit(0)
  }
}