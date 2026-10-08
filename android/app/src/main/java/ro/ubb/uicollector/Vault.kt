package ro.ubb.uicollector

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import android.util.Base64
import org.json.JSONObject

/** Tokens and pending payloads share a per-install Keystore key with domain-separated authenticated contexts. */
class Vault(private val context: Context)
{
    private val preferences = context.getSharedPreferences("secure_auth", Context.MODE_PRIVATE)
    private val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
    private val key: SecretKey

    init
    {
        val alias = "ui-embedding-collector-v1"
        if (store.containsAlias(alias) == false)
        {
            val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
            generator.init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT).setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).setRandomizedEncryptionRequired(true).build())
            generator.generateKey()
        }
        key = store.getKey(alias, null) as SecretKey
    }

    fun seal(plaintext: ByteArray, associatedData: String): ByteArray
    {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, key)
        cipher.updateAAD(associatedData.toByteArray(Charsets.UTF_8))
        return cipher.iv + cipher.doFinal(plaintext)
    }

    fun open(ciphertext: ByteArray, associatedData: String): ByteArray
    {
        require(ciphertext.size >= 28)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, key, GCMParameterSpec(128, ciphertext.copyOfRange(0, 12)))
        cipher.updateAAD(associatedData.toByteArray(Charsets.UTF_8))
        return cipher.doFinal(ciphertext, 12, ciphertext.size - 12)
    }

    @Synchronized
    fun identity(): JSONObject?
    {
        val stored = preferences.getString("identity", null) ?: return null
        return JSONObject(String(open(Base64.decode(stored, Base64.NO_WRAP), "identity"), Charsets.UTF_8))
    }

    @Synchronized
    fun saveIdentity(identity: JSONObject)
    {
        val encrypted = seal(identity.toString().toByteArray(Charsets.UTF_8), "identity")
        check(preferences.edit().putString("identity", Base64.encodeToString(encrypted, Base64.NO_WRAP)).commit())
    }
}
