import java.io.File
import org.nd4j.linalg.factory.Nd4j
import org.datavec.api.split.FileSplit
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.util.ndarray.RecordConverter
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex
import scala.util.Random

object MultipleLinearRegression {

  /*
  Note -
  1. Single Dense Layer of n ANNs.
  2. Input X, dimensions - [m x n], where m is #Samples.
  3. Each ANN will have its weight w => n ANNs have W vector, dimensions - [1 x n]
  4. Each ANN will have its bias(b), but equal => just 1 bias B is learnable where B = Sum(b)
  */

  var hyperParams = Map(
    "batchSize" -> 50,
    "epochs" -> 30,
    "learningRate" -> 1e-5,
    "weight" -> Nd4j.empty(),
    "bias" -> Random.nextFloat(),
  )

  val dataParams = Map(
    "splitRatio" -> 0.75,
    "maxRows" -> 1000,
    "filePath" -> ".\\data\\Boston Housing.csv"
  )

  def forwardProp(weight: INDArray, bias: Float, X: INDArray): INDArray = {
    // Assuming Activation as LINEAR i.e. f(x) = x => YPred = X.crossProd(W) + b
    X.mmul(weight).add(bias) // dimensions - [1 x m]
  }

  def backwardProp(YPred: INDArray, YTrue: INDArray, X: INDArray): (INDArray, Float) = {
    // dW = transpose(X).crossProd(YPred - YTrue) / numSamples
    // dB = (YPred - YTrue).sum / numSamples
    val numSamples = X.length()
    val dWCrossProd : INDArray = X.transpose().mmul(YPred.sub(YTrue)) // dimensions : [1 x n]
    val dBDotProd : Float = YPred.sub(YTrue).toFloatVector.sum // (YPred - YTrue).sum
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

  def costFunction(YPred: INDArray, YTrue: INDArray): Float = {
    val numElements = YPred.length()
    // [(YPred - YTrue).dot(YPred - YTrue)] / (2*numElements)
    val costDotProd : Float = YPred.sub(YTrue).mmul(YPred.sub(YTrue)).getFloat(0.asInstanceOf[Long])
    costDotProd/(2*numElements)
  }

  def multiRegressionModel(mode: String, X: INDArray, weight: INDArray, bias: Float, batchSize: Int, epochs: Int): Map[String, Any] = {
    var updatedParams : Map[String, Any] = Map(
      "weight" -> weight,
      "bias" -> bias
    )
    var cost : Float = 0
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
        cost = costFunction(YPred = yPred, YTrue = xCurrent.getColumn(xCurrent.columns() - 1))

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
          println(f"Batch - $idx%d, Epoch - $iEpoch%d, Cost - $cost%2.2f")
        }

      }
      print("\n")
    }

    Map(
      "weight" -> updatedParams("weight"),
      "bias" -> updatedParams("bias"),
      "cost" -> cost
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
    hyperParams += ("weight" -> Nd4j.randn(allData.columns() - 1))

    // Prepare Train and Validation data
    val xTrain: INDArray = allData.get(
      NDArrayIndex.interval(0,1,(dataParams("splitRatio").asInstanceOf[Double] * allData.rows()).asInstanceOf[Int]),
      NDArrayIndex.interval(0,1,allData.columns())
    )
    val xValid: INDArray = allData.get(
      NDArrayIndex.interval((dataParams("splitRatio").asInstanceOf[Double] * allData.rows()).asInstanceOf[Int],1,allData.rows()),
      NDArrayIndex.interval(0,1,allData.columns())
    )

    // Training
    val trainStatus : Map[String, Any] = multiRegressionModel(
      mode = "TRAIN",
      X = xTrain,
      weight = hyperParams("weight").asInstanceOf[INDArray],
      bias = hyperParams("bias").asInstanceOf[Float],
      batchSize = hyperParams("batchSize").asInstanceOf[Int],
      epochs = hyperParams("epochs").asInstanceOf[Int]
    )
    println(f"\nModel after Training : \nweight - ${trainStatus("weight").toString}%s \nbias - ${trainStatus("bias").asInstanceOf[Float]}%2.2f")
    println(f"Final Cost - ${trainStatus("cost").asInstanceOf[Float]}%2.2f\n")

    // Validation
    val validStatus : Map[String, Any] = multiRegressionModel(
      mode = "VALIDATION",
      X = xValid,
      weight = trainStatus("weight").asInstanceOf[INDArray],
      bias = trainStatus("bias").asInstanceOf[Float],
      batchSize = hyperParams("batchSize").asInstanceOf[Int],
      epochs = hyperParams("epochs").asInstanceOf[Int]
    )
    println(f"Final Cost after Validation - ${validStatus("cost").asInstanceOf[Float]}%2.2f")
  }
}