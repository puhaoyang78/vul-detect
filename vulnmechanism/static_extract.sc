import io.shiftleft.semanticcpg.language.*
import io.joern.dataflowengineoss.language.*
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.*

@main def main(inputPath: String, outputPath: String) = {
  val cpg = io.shiftleft.codepropertygraph.cpgloading.CpgLoader.load(inputPath)
  val context = new io.shiftleft.semanticcpg.layers.LayerCreatorContext(cpg)
  new io.joern.dataflowengineoss.layers.dataflows.OssDataFlow(
    new io.joern.dataflowengineoss.layers.dataflows.OssDataFlowOptions()
  ).run(context)
  val writer = Files.newBufferedWriter(Paths.get(outputPath))
  try {
    cpg.method.isExternal(false).filter(m => m.lineNumber.isDefined && !m.name.startsWith("<")).foreach { m =>
      val nodes = m.ast.l
      val ids = nodes.map(_.id).toSet
      val ns = nodes.map { n =>
        ujson.Obj("id" -> n.id.toString, "kind" -> n.label, "code" -> n.code,
          "line" -> n.lineNumber.getOrElse(-1), "column" -> n.columnNumber.getOrElse(-1),
          "name" -> n.properties.getOrElse("NAME", "").toString)
      }
      val es = nodes.flatMap { n =>
        flatgraph.Accessors.getEdgesOut(n).filter(e => ids.contains(e.dst.id)).map { e =>
          ujson.Arr(n.id.toString, e.dst.id.toString, e.label)
        }
      }
      val obj = ujson.Obj("file" -> m.filename, "method" -> m.name, "nodes" -> ujson.Arr.from(ns), "edges" -> ujson.Arr.from(es))
      writer.write(ujson.write(obj)); writer.newLine(); writer.flush()
    }
  } finally { writer.close(); cpg.close() }
}
