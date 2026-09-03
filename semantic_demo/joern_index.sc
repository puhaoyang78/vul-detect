import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.collection.mutable.ArrayBuffer
import io.shiftleft.semanticcpg.language._

@main def exec(
  cpgFile: String,
  outFile: String,
  sourceRoot: String,
  scopeFile: String
) = {
  def clean(value: String): String =
    value.replace("\\", "\\\\").replace("\t", " ").replace("\r", " ").replace("\n", " ")

  val sourcePath = Paths.get(sourceRoot).toAbsolutePath.normalize
  val scopes = Files.readAllLines(Paths.get(scopeFile)).toArray(new Array[String](0)).toSet

  def inAnalysisScope(relativePath: String): Boolean =
    scopes.exists { scope =>
      relativePath == scope || relativePath.startsWith(scope.stripSuffix("/") + "/")
    }

  def sourceRelative(filename: String): Option[String] = {
    try {
      val rawPath = Paths.get(filename)
      val path =
        if (rawPath.isAbsolute) rawPath.normalize
        else sourcePath.resolve(rawPath).normalize
      if (path.startsWith(sourcePath) && Files.isRegularFile(path)) {
        val relative = sourcePath.relativize(path).toString.replace('\\', '/')
        if (inAnalysisScope(relative)) Some(relative) else None
      }
      else
        None
    } catch {
      case _: Throwable => None
    }
  }

  val lines = ArrayBuffer[String]()
  try {
    importCpg(cpgFile)
    cpg.method.internal.l.foreach { method =>
      sourceRelative(method.filename).foreach { relativePath =>
        val start = method.lineNumber.getOrElse(-1)
        val end = method.lineNumberEnd.getOrElse(start)
        val returnType = method.methodReturn.typeFullName
        lines += (
          "METHOD\t" + clean(method.fullName) + "\t" + clean(method.name) + "\t" +
          clean(relativePath) + "\t" + start + "\t" + end + "\t" +
          clean(returnType)
        )
        method.parameter.filter(_.index > 0).l.foreach { parameter =>
          lines += (
            "PARAM\t" + clean(method.fullName) + "\t" + (parameter.index - 1) + "\t" +
            clean(parameter.name) + "\t" + clean(parameter.typeFullName)
          )
        }
        method.call.l.foreach { call =>
          lines += (
            "CALL\t" + clean(method.fullName) + "\t" + call.id + "\t" +
            call.lineNumber.getOrElse(-1) + "\t" + clean(call.name) + "\t" +
            clean(call.methodFullName) + "\t" + clean(call.dispatchType)
          )
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
