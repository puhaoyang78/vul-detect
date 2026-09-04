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
  val entrySource = Files.readAllLines(sourcePath.resolve(normalizedEntry)).asScala.toVector

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

  def identifierAt(line: String, name: String): Boolean = {
    val pattern = (".*(^|[^A-Za-z0-9_])" + java.util.regex.Pattern.quote(name) + "\\s*\\(.*").r
    pattern.pattern.matcher(line).matches()
  }

  def originalRange(name: String): Option[(Int, Int)] = {
    val candidates = ArrayBuffer[(Int, Int)]()
    var index = 0
    while (index < entrySource.length) {
      if (identifierAt(entrySource(index), name)) {
        val start = index
        var scan = index
        var sawOpen = false
        var depth = 0
        var inBlockComment = false
        var inString = false
        var inChar = false
        var escaped = false
        var rejected = false
        var done = false
        while (scan < entrySource.length && !done && !rejected) {
          val line = entrySource(scan)
          var column = 0
          var inLineComment = false
          while (column < line.length && !done && !rejected && !inLineComment) {
            val ch = line.charAt(column)
            val next = if (column + 1 < line.length) line.charAt(column + 1) else '\u0000'
            if (inBlockComment) {
              if (ch == '*' && next == '/') {
                inBlockComment = false
                column += 1
              }
            } else if (inString) {
              if (escaped) escaped = false
              else if (ch == '\\') escaped = true
              else if (ch == '"') inString = false
            } else if (inChar) {
              if (escaped) escaped = false
              else if (ch == '\\') escaped = true
              else if (ch == '\'') inChar = false
            } else if (ch == '/' && next == '*') {
              inBlockComment = true
              column += 1
            } else if (ch == '/' && next == '/') {
              inLineComment = true
            } else if (ch == '"') {
              inString = true
            } else if (ch == '\'') {
              inChar = true
            } else if (!sawOpen && ch == ';') {
              rejected = true
            } else if (ch == '{') {
              sawOpen = true
              depth += 1
            } else if (ch == '}' && sawOpen) {
              depth -= 1
              if (depth == 0) done = true
            }
            column += 1
          }
          if (!done && !rejected) scan += 1
        }
        if (done && sawOpen) candidates += ((start + 1, scan + 1))
      }
      index += 1
    }
    if (candidates.size == 1) Some(candidates.head) else None
  }

  val lines = ArrayBuffer[String]()
  try {
    importCpg(cpgFile)
    cpg.method.internal.l.foreach { method =>
      val entryRange =
        if (matchesEntry(method.filename)) originalRange(method.name)
        else None
      val relativePath =
        if (matchesEntry(method.filename)) entryRange.map(_ => normalizedEntry)
        else sourceRelative(method.filename)

      relativePath.foreach { path =>
        val rawStart = method.lineNumber.getOrElse(-1)
        val rawEnd = method.lineNumberEnd.getOrElse(rawStart)
        val (start, end) = entryRange.getOrElse((rawStart, rawEnd))
        val returnType = method.methodReturn.typeFullName
        lines += (
          "METHOD\t" + clean(method.fullName) + "\t" + clean(method.name) + "\t" +
          clean(path) + "\t" + start + "\t" + end + "\t" +
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
