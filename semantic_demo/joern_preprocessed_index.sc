import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._
import io.shiftleft.semanticcpg.language._

@main def exec(
  cpgFile: String,
  outFile: String,
  sourceRoot: String,
  scopeFile: String,
  entryPath: String,
  preprocessedFile: String
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

  def matchesOriginal(filename: String): Boolean = {
    val normalized = filename.replace('\\', '/')
    normalized == normalizedEntry || normalized.endsWith("/" + normalizedEntry)
  }

  def matchesEntry(filename: String): Boolean = {
    val normalized = filename.replace('\\', '/')
    matchesOriginal(normalized) || normalized == preprocessedEntry || normalized.endsWith("/" + preprocessedEntry)
  }

  val lineMap = mutable.Map[Int, Int]()
  val marker = """^\s*#\s+(\d+)\s+\"([^\"]+)\".*$""".r
  var currentFile = ""
  var currentLine = 1
  Files.readAllLines(Paths.get(preprocessedFile)).asScala.zipWithIndex.foreach { case (line, index) =>
    line match {
      case marker(number, filename) =>
        currentFile = filename.replace('\\', '/')
        currentLine = number.toInt
      case _ =>
        val physicalLine = index + 1
        if (matchesOriginal(currentFile)) {
          lineMap(physicalLine) = currentLine
        }
        currentLine += 1
    }
  }

  def mapLine(filename: String, line: Int): Int =
    if (line > 0 && matchesEntry(filename)) lineMap.getOrElse(line, line) else line

  def inAnalysisScope(relativePath: String): Boolean =
    scopes.exists { scope =>
      relativePath == scope || relativePath.startsWith(scope.stripSuffix("/") + "/")
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
        val rawStart = method.lineNumber.getOrElse(-1)
        val rawEnd = method.lineNumberEnd.getOrElse(rawStart)
        val start = mapLine(method.filename, rawStart)
        val end = mapLine(method.filename, rawEnd)
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
          val callLine = mapLine(method.filename, call.lineNumber.getOrElse(-1))
          lines += (
            "CALL\t" + clean(method.fullName) + "\t" + call.id + "\t" +
            callLine + "\t" + clean(call.name) + "\t" +
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
