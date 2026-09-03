import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.collection.mutable.ArrayBuffer
import io.shiftleft.semanticcpg.language._
import io.joern.dataflowengineoss.language._

@main def exec(
  cpgFile: String,
  outFile: String
) = {
  def clean(value: String): String =
    value.replace("\\", "\\\\").replace("\t", " ").replace("\r", " ").replace("\n", " ")

  val lines = ArrayBuffer[String]()
  try {
    importCpg(cpgFile)
    run.ossdataflow

    cpg.method.internal.l.foreach { method =>
      val key = clean(method.fullName)
      val start = method.lineNumber.getOrElse(-1)
      val end = method.lineNumberEnd.getOrElse(start)
      lines += (
        "METHOD\t" + key + "\t" + clean(method.name) + "\t" +
        clean(method.filename) + "\t" + start + "\t" + end
      )

      val params = method.parameter.filter(_.index > 0).l
      params.foreach { p =>
        lines += (
          "PARAM\t" + key + "\t" + (p.index - 1) + "\t" +
          clean(p.name) + "\t" + clean(p.typeFullName)
        )
      }

      method.call.l.foreach { call =>
        val callId = call.id.toString
        call.argument.filter(_.argumentIndex > 0).l.foreach { arg =>
          val argIndex = arg.argumentIndex - 1
          lines += (
            "ARG\t" + key + "\t" + callId + "\t" +
            call.lineNumber.getOrElse(-1) + "\t" + clean(call.name) + "\t" +
            argIndex + "\t" + clean(call.code) + "\t" + clean(arg.code)
          )
          params.foreach { p =>
            if (arg.reachableBy(p).l.nonEmpty) {
              lines += (
                "FLOW\t" + key + "\t" + (p.index - 1) + "\t" +
                callId + "\t" + argIndex
              )
            }
          }
        }
      }

      method.ast.isReturn.l.foreach { ret =>
        lines += ("RET\t" + key + "\t" + clean(ret.code))
        params.foreach { p =>
          if (ret.reachableBy(p).l.nonEmpty) {
            lines += ("RETFLOW\t" + key + "\t" + (p.index - 1))
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
