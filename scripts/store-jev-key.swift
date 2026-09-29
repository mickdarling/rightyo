// SPDX-License-Identifier: AGPL-3.0-or-later
// User input goes from NSSecureTextField directly to macOS Keychain Services.
// Never print the key, put it in argv, or write it to a configuration file.
import AppKit
import Foundation
import Security

let service = "rightyo.jev"
let account = "api-key"
let loginPath = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Keychains/login.keychain-db").path
var loginKeychain: SecKeychain?
let openStatus = SecKeychainOpen(loginPath, &loginKeychain)
guard openStatus == errSecSuccess, let keychain = loginKeychain else {
    FileHandle.standardError.write(Data("The login Keychain is unavailable.\n".utf8))
    exit(1)
}
let selector: [String: Any] = [
    kSecClass as String: kSecClassGenericPassword,
    kSecAttrService as String: service,
    kSecAttrAccount as String: account,
    kSecAttrSynchronizable as String: false,
    kSecMatchSearchList as String: [keychain],
]

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments == ["--status"] {
    var query = selector
    query[kSecReturnAttributes as String] = true
    query[kSecMatchLimit as String] = kSecMatchLimitOne
    var attributes: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &attributes)
    if status == errSecSuccess {
        print("Jev Keychain credential is configured; value was not read.")
        exit(0)
    }
    if status == errSecItemNotFound {
        print("Jev Keychain credential is not configured.")
        exit(2)
    }
    fail("Jev Keychain status is unavailable (OSStatus \(status)).")
}
guard arguments.isEmpty else {
    fail("Usage: store-jev-key [--status]")
}

let application = NSApplication.shared
application.setActivationPolicy(.accessory)
let menu = NSMenu()
let editItem = NSMenuItem()
let editMenu = NSMenu(title: "Edit")
editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
editItem.submenu = editMenu
menu.addItem(editItem)
application.mainMenu = menu
application.activate(ignoringOtherApps: true)

let alert = NSAlert()
alert.messageText = "Save your Jev API key for RightyO"
alert.informativeText = "Paste the API key into the secure field. It will be stored in your login Keychain, outside this repository. The key is not returned to chat or printed. If macOS later asks whether security may read it, choose Allow for that use, rather than Always Allow. Jev evaluations send text to TypeSafe only when hosted mode is explicitly enabled."
alert.alertStyle = .informational
alert.addButton(withTitle: "Save to Keychain")
alert.addButton(withTitle: "Cancel")

let field = NSSecureTextField(frame: NSRect(x: 0, y: 0, width: 400, height: 26))
field.placeholderString = "Jev API key"
field.setAccessibilityLabel("Jev API key")
alert.accessoryView = field
alert.window.initialFirstResponder = field
let response = alert.runModal()
guard response == .alertFirstButtonReturn else {
    field.stringValue = ""
    print("Credential entry cancelled; no key was saved.")
    exit(2)
}

let key = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
field.stringValue = ""
guard key.utf8.count >= 8, key.utf8.count <= 8192,
      key.unicodeScalars.allSatisfy({ $0.value >= 33 && $0.value <= 126 }) else {
    fail("No key saved: the entry must be a nonempty printable API token.")
}

var item = selector
item.removeValue(forKey: kSecMatchSearchList as String)
item[kSecUseKeychain as String] = keychain
item[kSecValueData as String] = Data(key.utf8)
item[kSecAttrLabel as String] = "RightyO Jev API key"
var access: SecAccess?
guard SecAccessCreate("RightyO Jev API key" as CFString, [] as CFArray, &access) == errSecSuccess,
      let access else {
    fail("No key saved: Keychain access policy could not be created.")
}
item[kSecAttrAccess as String] = access
var status = SecItemAdd(item as CFDictionary, nil)
var replaced = false
if status == errSecDuplicateItem {
    // Recreate instead of retaining an older item's access list/accessibility.
    let deletion = SecItemDelete(selector as CFDictionary)
    guard deletion == errSecSuccess else {
        fail("No key saved: previous Keychain item could not be replaced.")
    }
    replaced = true
    status = SecItemAdd(item as CFDictionary, nil)
}
guard status == errSecSuccess else {
    if replaced {
        fail("The previous credential was removed; the new key was not saved (OSStatus \(status)).")
    }
    fail("No key saved: Keychain write failed (OSStatus \(status)).")
}
print("Jev API key saved in the login Keychain. Its value was not returned.")
