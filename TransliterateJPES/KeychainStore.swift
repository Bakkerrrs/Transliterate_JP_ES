import Foundation
import Security

/// Almacén seguro para la API Key en el Keychain del sistema.
/// El valor queda cifrado por iOS y persiste entre lanzamientos de la app.
enum KeychainStore {

    private static let service = "com.example.TransliterateJPES"
    private static let account = "openai_api_key"

    /// Guarda (o reemplaza) el valor. Si está vacío, lo borra.
    static func save(_ value: String) {
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(base as CFDictionary)

        guard !value.isEmpty else { return }

        var attrs = base
        attrs[kSecValueData as String] = Data(value.utf8)
        attrs[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(attrs as CFDictionary, nil)
    }

    /// Lee el valor guardado, o `nil` si no hay nada.
    static func load() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let str = String(data: data, encoding: .utf8) else {
            return nil
        }
        return str
    }

    static func clear() {
        save("")
    }
}
