import java.io.File
import org.nd4j.linalg.factory.Nd4j
import org.datavec.api.split.FileSplit
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.util.ndarray.RecordConverter
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms.{log, exp, pow}
import scala.util.Random

object LogisticRegressionBinary {

  /*
  Note -
  1. Single Dense Layer of n ANNs.
  2. Input X, dimensions - [m x n], where m is #Samples.
  3. Each ANN will have its weight w => n ANNs have W vector, dimensions - [1 x n]
  4. Each ANN will have its bias(b), but equal => just 1 bias B is learnable where B = Sum(b)
  */

  var hyperParams = Map(
    "batchSize" -> 50,
    "epochs" -> 25,
    "learningRate" -> 1e-4,
    "weight" -> Nd4j.empty(),
    "bias" -> Random.nextFloat(),
  )

  val dataParams = Map(
    "splitRatio" -> 0.8,
    "maxRows" -> 1000,
    "filePath" -> ".\\data\\Indian Diabetes.csv"
  )

  def sigmoid(X : INDArray) : INDArray = {
    exp(X).div(exp(X).add(1)).add(1e-20) // Adding tolerance(1e-20) to remove 0 that produce -infinity at log(0)
  }

  def forwardProp(weight: INDArray, bias: Float, X: INDArray): INDArray = {
    // Assuming Activation as LINEAR i.e. f(x) = x => YPred = W.crossProd(X.transponse()) + b
    sigmoid(X.mmul(weight).add(bias)) // dimensions - [1 x m]
  }

  def backwardProp(YPred: INDArray, YTrue: INDArray, X: INDArray): (INDArray, Float) = {
    // dW = [YPred.crossProd(1 - YTrue) - YTrue.crossProd(1-YPred)].crossProd(X) / numSamples
    // dB = [YPred.dot(1 - YTrue) - YTrue.dot(1-YPred)] / numSamples
    val numSamples = X.rows()
    val dWCrossProd : INDArray = X.transpose(). mmul (YPred.mul(YTrue.sub(1).mul(-1)) .sub (YTrue.mul(YPred.sub(1).mul(-1)))) // dimensions : [1 x n]
    val dBDotProd : Float = YPred.mmul(YTrue.sub(1).mul(-1)) .sub (YTrue.mmul(YPred.sub(1).mul(-1))) .getFloat(0.asInstanceOf[Long])
    (dWCrossProd.div(numSamples), dBDotProd/numSamples)
  }

  def updateParams(weight: INDArray, bias: Float, dW: INDArray, dB: Float, learningRate: Double): Map[String, Any] = {
    val weightNew : INDArray = weight.sub(dW.mul(learningRate))
    val biasNew = bias - dB
    Map(
      "weight" -> weightNew,
      "bias" -> biasNew.asInstanceOf[Float]
    )
  }

  // Binary Cross-Entropy
  def costFunction(YPred: INDArray, YTrue: INDArray): Map[String, Float] = {
    // - [(YTrue).dot(log(YPred)) + (1-YTrue).dot(log(1-YPred))] / (numElements)
    val numElements : Long = YPred.length()
    val costDotProd : Float = YTrue.mmul(log(YPred)) .add (YTrue.sub(1).mul(-1).mmul(log(YPred.sub(1).mul(-1)))) .mul(-1) .getFloat(0.asInstanceOf[Long])
    val accuracyProb : Float = pow(YPred, YTrue) .mmul (pow(YPred.sub(1).mul(-1),YTrue.sub(1).mul(-1))) .getFloat(0.asInstanceOf[Long])
    Map(
      "cost" -> costDotProd/numElements,
      "probAcc" -> accuracyProb/numElements
    )
  }

  def logitisticBinaryRegressionModel(mode: String, X: INDArray, weight: INDArray, bias: Float, batchSize: Int, epochs: Int): Map[String, Any] = {
    var updatedParams : Map[String, Any] = Map(
      "weight" -> weight,
      "bias" -> bias
    )
    var costAll : List[Float] = List.empty
    var probAccAll : List[Float] = List.empty
    var performanceParams : Map[String, Float] = Map(
      "cost" -> 0,
      "probAcc" -> 0
    )
    var xCurrent: INDArray = Nd4j.empty()

    for(idx <- 0 to X.rows()/batchSize) {
      if((idx + 1)*batchSize > X.rows()) {
        xCurrent = X.get(
          NDArrayIndex.interval(idx*batchSize, 1, X.rows()),
          NDArrayIndex.interval(0, 1, X.columns())
        )
      }
      else {
        xCurrent = X.get(
          NDArrayIndex.interval(idx*batchSize, 1, (idx + 1)*batchSize),
          NDArrayIndex.interval(0, 1, X.columns())
        )
      }

      for(iEpoch <- 1 to epochs) {
        // Forward Propagation
        val yPred: INDArray = forwardProp(
          weight = updatedParams("weight").asInstanceOf[INDArray],
          bias = updatedParams("bias").asInstanceOf[Float],
          X = xCurrent.get(
            NDArrayIndex.interval(0,1, xCurrent.rows()),
            NDArrayIndex.interval(0,1, xCurrent.columns() - 1)
          )
        )

        // Cost
        performanceParams = costFunction(YPred = yPred, YTrue = xCurrent.getColumn(xCurrent.columns() - 1))
        costAll = performanceParams("cost") :: costAll
        probAccAll = performanceParams("probAcc") :: probAccAll

        if(mode.toLowerCase.equals("train")) {
          // Backward Propagation or Optimization
          val (dW, dB) = backwardProp(
            YPred = yPred,
            YTrue = xCurrent.getColumn(xCurrent.columns() - 1),
            X = xCurrent.get(
              NDArrayIndex.interval(0,1, xCurrent.rows()),
              NDArrayIndex.interval(0, 1, xCurrent.columns() - 1)
            )
          )

          // Update parameters
          updatedParams = updateParams(
            weight = updatedParams("weight").asInstanceOf[INDArray],
            bias = updatedParams("bias").asInstanceOf[Float],
            dW = dW,
            dB = dB,
            learningRate = hyperParams("learningRate").asInstanceOf[Double]
          )
        }

        // Status at Epochs
        if(iEpoch % 5 == 0) {
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
    val dataFile = new File(dataParams("filePath").toString)
    val recordReader : RecordReader = new CSVRecordReader(1)
    recordReader.initialize(new FileSplit(dataFile))
    val allData : INDArray = RecordConverter.toMatrix(recordReader.next(dataParams("maxRows").asInstanceOf[Int]))
    recordReader.close()

    // Initialize weights
    hyperParams += ("weight" -> Nd4j.zeros(allData.columns() - 1))

    // Prepare Train and Validation data
    val xTrain: INDArray = allData.get(
      NDArrayIndex.interval(0,1, (dataParams("splitRatio").asInstanceOf[Double] * allData.rows()).asInstanceOf[Int]),
      NDArrayIndex.interval(0,1, allData.columns())
    )
    val xValid: INDArray = allData.get(
      NDArrayIndex.interval((dataParams("splitRatio").asInstanceOf[Double] * allData.rows()).asInstanceOf[Int],1,allData.rows()),
      NDArrayIndex.interval(0,1,allData.columns())
    )

    // Training
    val trainStatus : Map[String, Any] = logitisticBinaryRegressionModel(
      mode = "TRAIN",
      X = xTrain,
      weight = hyperParams("weight").asInstanceOf[INDArray],
      bias = hyperParams("bias").asInstanceOf[Float],
      batchSize = hyperParams("batchSize").asInstanceOf[Int],
      epochs = hyperParams("epochs").asInstanceOf[Int]
    )
    println(f"\nModel after Training : \nweight - ${trainStatus("weight").toString}%s \nbias - ${trainStatus("bias").asInstanceOf[Float]}%2.2f" +
      f"\nFinal Cost - ${trainStatus("cost").asInstanceOf[Float]}%2.2f\nFinal Accuracy - ${trainStatus("probAcc").asInstanceOf[Float]}%2.2f\n"
    )

    // Validation
    val validStatus : Map[String, Any] = logitisticBinaryRegressionModel(
      mode = "VALIDATION",
      X = xValid,
      weight = trainStatus("weight").asInstanceOf[INDArray],
      bias = trainStatus("bias").asInstanceOf[Float],
      batchSize = hyperParams("batchSize").asInstanceOf[Int],
      epochs = hyperParams("epochs").asInstanceOf[Int]
    )
    println(f"After Validation : \nFinal Cost - ${validStatus("cost").asInstanceOf[Float]}%2.2f \n" +
      f"Final Accuracy - ${validStatus("probAcc").asInstanceOf[Float]}%2.2f")
  }
}
