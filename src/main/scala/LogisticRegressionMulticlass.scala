import java.io.File
import java.nio.file.{Files, Path, Paths}
import org.datavec.api.records.reader.RecordReader
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms.{exp, log, pow}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.datavec.api.split.FileSplit
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.schema.Schema

object LogisticRegressionMulticlass {

  /*
  Note -
  1. Dense Layer :
      a. Layer 1 - "n" ANs, Hidden layer
      b. Layer 2 - "k" Ans, Output layer, all connected to Layer 1
  2. Input X, dimensions - [m x n], where m is #Samples.
  3. Each AN will have its weight w :
      a. Layer 1 - "n" ANs have W vector, dimensions - [1 x n]
      b. Layer 2 - "k" ANs have W vector, dimensions - [1 x k], all connected to Layer 1
      c. Implies W matrix, dimensions - [n x k]
  4. Each AN will have its bias(b) :
      a. Layer 1 - "n" ANs have "b", but equal.
      b. Layer 2 - "k" ANs have "b" for each equal "b" in Layer 1.
      c. Implies B vector, dimensions - [1 x k]

  where :
      m -> (1 to #samples)
      n -> (1 to #features)
      k -> (1 to #classes)

  NOTE :
      i. MinMax normalization - used for equally distributed values.
      ii. Standardize normalization - used for normally distributed values.
      iii. Logarithmic normalization - used for values covering a wider range than other columns.
  */

  var hyperParams = Map(
    "batchSize" -> 200,
    "epochs" -> 100,
    "learningRate" -> 1e-9,
    "weight" -> Nd4j.empty(),
    "bias" -> Nd4j.empty(),
  )

  var dataParams : Map[String, Any] = Map(
    "splitRatio" -> 0.8,
    "numSamples" -> 0,
    "numClasses" -> 10,
    "filePath" -> ".\\data\\Wine Quality.csv"
  )

  def prepareDataSet() : RecordReaderDataSetIterator = {
    // Fetch Data
    val dataFile = new File(dataParams("filePath").toString)
    val recordReader : RecordReader = new CSVRecordReader(1)
    recordReader.initialize(new FileSplit(dataFile))

    val path : Path = Paths.get(dataParams("filePath").toString)
    dataParams += ("numSamples" -> Files.lines(path).count().toInt)

    // Prepare Schema
    val schema : Schema = new Schema.Builder()
      .addColumnsDouble("fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol")
      .addColumnCategorical("quality", "0","1","2","3","4","5","6","7","8","9")
      .build()

    // Vectorization
    val dataSetIterator : RecordReaderDataSetIterator = new RecordReaderDataSetIterator.Builder(recordReader, dataParams("numSamples").asInstanceOf[Int])
      .classification(schema.getIndexOfColumn("quality"), dataParams("numClasses").asInstanceOf[Int])
      .build()

    dataSetIterator
  }

  def softmax(X : INDArray) : INDArray = {
    exp(X).div(exp(X).sumNumber())
  }

  def forwardProp(weight: INDArray, bias: INDArray, X: INDArray): INDArray = {
    // Assuming Activation as LINEAR i.e. f(x) = x => YPred = X.crossProd(W) + b
    softmax(X.mmul(weight).add(bias)) // dimensions - [m x k]
  }

  def backwardProp(YPred: INDArray, YTrue: INDArray, X: INDArray): (INDArray, INDArray) = {
    // dW = [X.transpose).crossProd(YTrue.elemProd(1 - YPred)] / numSamples
    // dB = [YTrue.elemProd(1-YPred)].sumAcrossColumns / numSamples
    val numSamples = X.rows()
    val dWCrossProd : INDArray = X.transpose().mmul(YTrue.mul(YPred.sub(1).mul(-1))) // dimensions : [n x k]
    val dBDotProd : INDArray = YTrue.mul(YPred.sub(1).mul(-1)).sum(0) // dimensions : [1 x k]
    (dWCrossProd.div(numSamples), dBDotProd.div(numSamples))
  }

  def updateParams(weight: INDArray, bias: INDArray, dW: INDArray, dB: INDArray, learningRate: Double): Map[String, INDArray] = {
    val weightNew : INDArray = weight.sub(dW.mul(learningRate))
    val biasNew : INDArray = bias.sub(dB.mul(learningRate))
    Map(
      "weight" -> weightNew,
      "bias" -> biasNew
    )
  }

  // Cross-Entropy
  def costFunction(YPred: INDArray, YTrue: INDArray): Map[String, Double] = {
    // - (YTrue).elementProd(log(YPred).sumOfElements / (numSamples)
    val numSamples : Long = YPred.rows()
    val costDotProd : Double = YTrue.mul(log(YPred)).sumNumber().asInstanceOf[Double]
    val accuracyProb : Double = pow(YPred, YTrue).prod(1).sumNumber().asInstanceOf[Double]
    Map(
      "cost" -> -1 * costDotProd/numSamples,
      "probAcc" -> accuracyProb
    )
  }

  def logitisticMulticlassRegressionModel(mode: String, X: INDArray, Y: INDArray,weight: INDArray, bias: INDArray, batchSize: Int, epochs: Int): Map[String, Any] = {
    var updatedParams : Map[String, Any] = Map(
      "weight" -> weight,
      "bias" -> bias
    )
    var costAll : List[Double] = List.empty
    var probAccAll : List[Double] = List.empty
    var performanceParams : Map[String, Double] = Map(
      "cost" -> 0,
      "probAcc" -> 0
    )
    var xCurrent: INDArray = Nd4j.empty()
    var yCurrent: INDArray = Nd4j.empty()

    for(idx <- 0 to X.rows()/batchSize) {
      if((idx + 1)*batchSize > X.rows()) {
        xCurrent = X.get(
          NDArrayIndex.interval(idx*batchSize, 1, X.rows()),
          NDArrayIndex.interval(0, 1, X.columns())
        )
        yCurrent = Y.get(
          NDArrayIndex.interval(idx*batchSize, 1, Y.rows()),
          NDArrayIndex.interval(0, 1, Y.columns())
        )
      }
      else {
        xCurrent = X.get(
          NDArrayIndex.interval(idx*batchSize, 1, (idx + 1)*batchSize),
          NDArrayIndex.interval(0, 1, X.columns())
        )
        yCurrent = Y.get(
          NDArrayIndex.interval(idx*batchSize, 1, (idx + 1)*batchSize),
          NDArrayIndex.interval(0, 1, Y.columns())
        )
      }

      for(iEpoch <- 1 to epochs) {
        // Forward Propagation
        val yPred: INDArray = forwardProp(
          weight = updatedParams("weight").asInstanceOf[INDArray],
          bias = updatedParams("bias").asInstanceOf[INDArray],
          X = xCurrent
        )

        // Cost
        performanceParams = costFunction(YPred = yPred, YTrue = yCurrent)
        costAll = performanceParams("cost") :: costAll
        probAccAll = performanceParams("probAcc") :: probAccAll

        if(mode.toLowerCase.equals("train")) {
          // Backward Propagation or Optimization
          val (dW, dB) = backwardProp(
            YPred = yPred,
            YTrue = yCurrent,
            X = xCurrent
          )

          // Update parameters
          updatedParams = updateParams(
            weight = updatedParams("weight").asInstanceOf[INDArray],
            bias = updatedParams("bias").asInstanceOf[INDArray],
            dW = dW,
            dB = dB,
            learningRate = hyperParams("learningRate").asInstanceOf[Double]
          )
        }

        // Status at Epochs
        if(iEpoch % 10 == 0) {
          println(f"Batch - $idx%d, Epoch - $iEpoch%d, Cost - ${performanceParams("cost")}%2.2f, " +
            f"Accuracy - ${performanceParams("probAcc")}%2.2f")
        }

      }
      print("\n")
    }

    Map(
      "weight" -> updatedParams("weight"),
      "bias" -> updatedParams("bias"),
      "cost" -> costAll.sum/costAll.length,
      "probAcc" -> probAccAll.sum/probAccAll.length
    )
  }

  def main(args: Array[String]): Unit = {

    // Fetch Data
    val dataSetIterator : RecordReaderDataSetIterator = prepareDataSet()
    val allDataX : INDArray = dataSetIterator.next().getFeatures
    dataSetIterator.reset()
    val allDataY : INDArray = dataSetIterator.next().getLabels

    // Prepare Train and Validation data
    val xTrain: INDArray = allDataX.get(
      NDArrayIndex.interval(0,1, (dataParams("splitRatio").asInstanceOf[Double] * dataParams("numSamples").asInstanceOf[Int]).asInstanceOf[Int]),
      NDArrayIndex.interval(0,1, allDataX.columns())
    )
    val xValid: INDArray = allDataX.get(
      NDArrayIndex.interval((dataParams("splitRatio").asInstanceOf[Double] * dataParams("numSamples").asInstanceOf[Int]).asInstanceOf[Int],1,dataParams("numSamples").asInstanceOf[Int] - 1),
      NDArrayIndex.interval(0,1, allDataX.columns())
    )
    val yTrain: INDArray = allDataY.get(
      NDArrayIndex.interval(0,1, (dataParams("splitRatio").asInstanceOf[Double] * dataParams("numSamples").asInstanceOf[Int]).asInstanceOf[Int]),
      NDArrayIndex.interval(0,1, allDataY.columns())
    )
    val yValid: INDArray = allDataY.get(
      NDArrayIndex.interval((dataParams("splitRatio").asInstanceOf[Double] * dataParams("numSamples").asInstanceOf[Int]).asInstanceOf[Int],1,dataParams("numSamples").asInstanceOf[Int] - 1),
      NDArrayIndex.interval(0,1, allDataY.columns())
    )

    // Initialize weights
    hyperParams += ("weight" -> Nd4j.zeros(xTrain.columns().asInstanceOf[Long], yTrain.columns().asInstanceOf[Long]))
    hyperParams += ("bias" -> Nd4j.zeros(yTrain.columns()))

    // Training
    val trainStatus : Map[String, Any] = logitisticMulticlassRegressionModel(
      mode = "TRAIN",
      X = xTrain,
      Y = yTrain,
      weight = hyperParams("weight").asInstanceOf[INDArray],
      bias = hyperParams("bias").asInstanceOf[INDArray],
      batchSize = hyperParams("batchSize").asInstanceOf[Int],
      epochs = hyperParams("epochs").asInstanceOf[Int]
    )
    println(f"\nModel after Training : \nweight - ${trainStatus("weight").toString}%s \nbias - ${trainStatus("bias").toString}%s" +
      f"\nFinal Cost - ${trainStatus("cost").asInstanceOf[Double]}%2.2f\nFinal Accuracy - ${trainStatus("probAcc").asInstanceOf[Double]}%2.2f\n"
    )

    // Validation
    val validStatus : Map[String, Any] = logitisticMulticlassRegressionModel(
      mode = "VALIDATION",
      X = xValid,
      Y = yValid,
      weight = trainStatus("weight").asInstanceOf[INDArray],
      bias = trainStatus("bias").asInstanceOf[INDArray],
      batchSize = hyperParams("batchSize").asInstanceOf[Int],
      epochs = hyperParams("epochs").asInstanceOf[Int]
    )
    println(f"After Validation : \nFinal Cost - ${validStatus("cost").asInstanceOf[Double]}%2.2f \n" +
      f"Final Accuracy - ${validStatus("probAcc").asInstanceOf[Double]}%2.2f")
  }
}
