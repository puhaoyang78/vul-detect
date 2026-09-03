import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.collection.mutable.ArrayBuffer
import io.shiftleft.semanticcpg.language._
import io.joern.dataflowengineoss.language._

@main def exec(
  cpgFile: String,
  outFile: String,
  sourceRoot: String
) = {
  def clean(value: String): String =
    value.replace("\\", "\\\\").replace("\t", " ").replace("\r", " ").replace("\n", " ")

  def relative(filename: String): String = {
    try {
      val root = Paths.get(sourceRoot).toAbsolutePath.normalize
      val path = Paths.get(filename).toAbsolutePath.normalize
      if (path.startsWith(root)) root.relativize(path).toString.replace('\\', '/')
      else filename.replace('\\', '/')
    } catch {
      case _: Throwable => filename.replace('\\', '/')
    }
  }

  val lines = ArrayBuffer[String]()
  try {
    importCpg(cpgFile)
    run.ossdataflow
    cpg.method.internal.l.foreach { method =>
      val start = method.lineNumber.getOrElse(-1)
      val end = method.lineNumberEnd.getOrElse(start)
      lines += (
        "METHOD\t" + clean(method.fullName) + "\t" + clean(method.name) + "\t" +
        clean(relative(method.filename)) + "\t" + start + "\t" + end
      )
      method.parameter.l.foreach { parameter =>
        lines += (
          "PARAM\t" + clean(method.fullName) + "\t" + (parameter.index - 1) + "\t" +
          clean(parameter.name) + "\t" + clean(parameter.typeFullName)
        )
      }
      method.call.l.foreach { call =>
        lines += (
          "CALL\t" + clean(method.fullName) + "\t" + call.lineNumber.getOrElse(-1) + "\t" +
          clean(call.name) + "\t" + clean(call.methodFullName) + "\t" + clean(call.dispatchType)
        )
        call.argument.l.foreach { arg =>
          val argIndex = arg.argumentIndex - 1
          lines += (
            "ARGFACT\t" + clean(method.fullName) + "\t" + call.lineNumber.getOrElse(-1) + "\t" +
            clean(call.name) + "\t" + argIndex + "\t" + clean(call.code) + "\t" + clean(arg.code)
          )
          method.parameter.l.foreach { parameter =>
            if (arg.reachableBy(parameter).l.nonEmpty) {
              lines += (
                "FLOWFACT\t" + clean(method.fullName) + "\t" + (parameter.index - 1) + "\t" +
                call.lineNumber.getOrElse(-1) + "\t" + clean(call.name) + "\t" + argIndex
              )
            }
          }
        }
      }
      method.ast.isReturn.l.foreach { ret =>
        lines += ("RETFACT\t" + clean(method.fullName) + "\t" + clean(ret.code))
        method.parameter.l.foreach { parameter =>
          if (ret.reachableBy(parameter).l.nonEmpty) {
            lines += (
              "RETFLOWFACT\t" + clean(method.fullName) + "\t" + (parameter.index - 1)
            )
          }
        }
      }
    }
  } catch {
    case error: Throwable =>
      lines.clear()
      lines += ("ERROR\t" + clean(error.getClass.getSimpleName + ":" + Option(error.getMessage).getOrElse("")))
  }

  Files.write(
    Paths.get(outFile),
    (lines.mkString("\n") + "\n").getBytes(StandardCharsets.UTF_8)
  )
}
