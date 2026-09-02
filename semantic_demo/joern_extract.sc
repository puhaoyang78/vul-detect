import scala.io.Source
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.collection.mutable.ArrayBuffer
import io.shiftleft.semanticcpg.language._
import io.joern.dataflowengineoss.language._

@main def exec(sourceFile: String, outFile: String, functionName: String) = {
  def clean(value: String): String =
    value.replace("\\", "\\\\").replace("\t", " ").replace("\r", " ").replace("\n", " ")

  val lines = ArrayBuffer[String]()
  try {
    val code = Source.fromFile(sourceFile).mkString
    importCode.c.fromString(code)
    run.ossdataflow

    val methods = cpg.method.nameExact(functionName).l
    if (methods.isEmpty) {
      lines += ("ERROR\tmethod_not_found:" + clean(functionName))
    } else if (methods.size != 1) {
      lines += ("ERROR\tambiguous_method:" + clean(functionName))
    } else {
      val method = methods.head
      val params = method.parameter.l

      params.foreach { p =>
        lines += ("PARAM\t" + (p.index - 1) + "\t" + clean(p.name) + "\t" + clean(p.typeFullName))
      }

      method.call.l.foreach { call =>
        call.argument.l.foreach { arg =>
          val argIndex = arg.argumentIndex - 1
          lines += (
            "ARG\t" + call.lineNumber.getOrElse(-1) + "\t" + clean(call.name) + "\t" +
            argIndex + "\t" + clean(call.code) + "\t" + clean(arg.code)
          )
          params.foreach { p =>
            try {
              if (arg.reachableBy(p).l.nonEmpty) {
                lines += (
                  "FLOW\t" + (p.index - 1) + "\t" + call.lineNumber.getOrElse(-1) + "\t" +
                  clean(call.name) + "\t" + argIndex
                )
              }
            } catch {
              case _: Throwable => ()
            }
          }
        }
      }

      method.ast.isReturn.l.foreach { ret =>
        lines += ("RET\t" + clean(ret.code))
        params.foreach { p =>
          try {
            if (ret.reachableBy(p).l.nonEmpty) {
              lines += ("RETFLOW\t" + (p.index - 1))
            }
          } catch {
            case _: Throwable => ()
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
