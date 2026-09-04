import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._
import io.shiftleft.semanticcpg.language._

@main def exec(
  cpgFile: String,
  outFile: String,
  sourceRoot: String,
  scopeFile: String,
  entryPath: String
) = {
  def clean(value: String): String =
    value.replace("\\", "\\\\").replace("\t", " ").replace("\r", " ").replace("\n", " ")

  val sourcePath = Paths.get(sourceRoot).toAbsolutePath.normalize
  val scopes = Files.readAllLines(Paths.get(scopeFile)).asScala.toSet
  val normalizedEntry = entryPath.replace('\\', '/')
  val preprocessedEntry = {
    val dot = normalizedEntry.lastIndexOf('.')
    if (dot >= 0) normalizedEntry.substring(0, dot) + ".i"
    else normalizedEntry + ".i"
  }

  def inAnalysisScope(relativePath: String): Boolean =
    scopes.exists { scope =>
      relativePath == scope || relativePath.startsWith(scope.stripSuffix("/") + "/")
    }

  def matchesEntry(filename: String): Boolean = {
    val normalized = filename.replace('\\', '/')
    normalized == normalizedEntry || normalized.endsWith("/" + normalizedEntry) ||
    normalized == preprocessedEntry || normalized.endsWith("/" + preprocessedEntry)
  }

  def sourceRelative(filename: String): Option[String] = {
    try {
      if (matchesEntry(filename)) {
        val original = sourcePath.resolve(normalizedEntry).normalize
        if (Files.isRegularFile(original) && inAnalysisScope(normalizedEntry)) Some(normalizedEntry)
        else None
      } else {
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
      }
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
        method.parameter.filter(p => p.index > 0 && !p.isVariadic).l.foreach { parameter =>
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
