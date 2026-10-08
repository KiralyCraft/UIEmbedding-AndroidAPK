import com.intellij.openapi.util.Disposer
import com.intellij.psi.PsiErrorElement
import com.intellij.psi.util.PsiTreeUtil
import org.jetbrains.kotlin.cli.jvm.compiler.EnvironmentConfigFiles
import org.jetbrains.kotlin.cli.jvm.compiler.KotlinCoreEnvironment
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.psi.KtPsiFactory
import java.io.File

fun main(args: Array<String>)
{
    val disposable = Disposer.newDisposable()
    try
    {
        val environment = KotlinCoreEnvironment.createForProduction(disposable, CompilerConfiguration(), EnvironmentConfigFiles.JVM_CONFIG_FILES)
        val factory = KtPsiFactory(environment.project)
        var count = 0
        var errors = 0
        File(args[0]).walkTopDown().filter { it.isFile && (it.extension == "kt" || it.extension == "kts") }.forEach { path ->
            val text = path.readText()
            val parsed = factory.createFile(path.name, text)
            count += 1
            PsiTreeUtil.collectElementsOfType(parsed, PsiErrorElement::class.java).forEach { error ->
                val line = text.take(error.textOffset).count { it == '\n' } + 1
                println("${path.path}:$line: ${error.errorDescription}")
                errors += 1
            }
        }
        check(errors == 0) { "$errors Kotlin parser errors" }
        println("Parsed $count Kotlin/Gradle files without syntax errors. This is NOT Android compilation or type checking.")
    }
    finally { Disposer.dispose(disposable) }
}
