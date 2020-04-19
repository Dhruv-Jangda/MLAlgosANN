import java.io.File
import java.nio.file.{Files, Path, Paths}
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.analysis.DataAnalysis
import org.datavec.api.transform.schema.Schema
import org.datavec.api.transform.transform.normalize.Normalize
import org.datavec.api.transform.ui.HtmlAnalysis
import org.datavec.local.transforms.AnalyzeLocal
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex

object DataExploration {

  def main(args: Array[String]): Unit = {
    // Load Data
    val dataFile = new File(".\\data\\Wine Quality.csv")
    val recordReader : CSVRecordReader = new CSVRecordReader(1) // Skipping headers
    val inputSplit : FileSplit = new FileSplit(dataFile)
    recordReader.initialize(inputSplit)

    val path : Path = Paths.get(".\\data\\Wine Quality.csv")
    val numSamples : Int = Files.lines(path).count().toInt

    // Prepare Schema
    val schema : Schema = new Schema.Builder()
      .addColumnsDouble("fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol")
      .addColumnCategorical("quality", "0","1","2","3","4","5","6","7","8","9")
      .build()

    // Analysing Data
    val analysis : DataAnalysis = AnalyzeLocal.analyze(schema, recordReader)
    HtmlAnalysis.createHtmlAnalysisFile(analysis, new File(".\\data\\Wine Quality.html"))

    // Transforming Columns
    val transformProcess : TransformProcess = new TransformProcess.Builder(schema)
      .normalize("fixed acidity", Normalize.Standardize, analysis)
      .normalize("volatile acidity", Normalize.Standardize, analysis)
      .normalize("citric acid", Normalize.Standardize, analysis)
      .normalize("residual sugar", Normalize.Standardize, analysis)
      .normalize("chlorides", Normalize.Standardize, analysis)
      .normalize("free sulfur dioxide", Normalize.Log2Mean, analysis)
      .normalize("total sulfur dioxide", Normalize.Log2Mean, analysis)
      .normalize("density", Normalize.Standardize, analysis)
      .normalize("pH", Normalize.Standardize, analysis)
      .normalize("sulphates", Normalize.Standardize, analysis)
      .normalize("alcohol", Normalize.Standardize, analysis)
      .build()

    // Finalizing Schema
    val finalSchema : Schema = transformProcess.getFinalSchema
    val trainRecReader : TransformProcessRecordReader = new TransformProcessRecordReader(new CSVRecordReader(1), transformProcess)
    trainRecReader.initialize(inputSplit)

    // Vectorization
    val dataSetIterator : RecordReaderDataSetIterator = new RecordReaderDataSetIterator.Builder(trainRecReader, numSamples)
      .classification(finalSchema.getIndexOfColumn("quality"), 10)
      .build()

    // Fetch Data
    val allDataX : INDArray = dataSetIterator.next().getFeatures
    dataSetIterator.reset()
    val allDataY : INDArray = dataSetIterator.next().getLabels

    val xTrain: INDArray = allDataX.get(
      NDArrayIndex.interval(0,1, 5),
      NDArrayIndex.interval(0,1, allDataX.columns())
    )
    val yTrain: INDArray = allDataY.get(
      NDArrayIndex.interval(0,1, 5),
      NDArrayIndex.interval(0,1, allDataY.columns())
    )
    println(f"Input Features:\n ${xTrain.toString()}%s\n")
    println(f"Input Labels(i.e. OH encoded target variable):\n ${yTrain.toString()}%s")
  }
}
