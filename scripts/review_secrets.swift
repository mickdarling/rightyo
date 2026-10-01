// SPDX-License-Identifier: AGPL-3.0-or-later
// Credentials go from masked native fields to gh's stdin. Never write them to files,
// command-line arguments, environment variables, stdout, stderr, or a model-facing API.
import AppKit
import Darwin
import Foundation

let repository = "mickdarling/rightyo"
let args = Array(CommandLine.arguments.dropFirst())
guard args.count == 4 || args.count == 5,
      args[0] == "--gh", args[2] == "--claude-auth",
      ["subscription", "api"].contains(args[3]),
      args.count == 4 || args[4] == "--check" else {
    print("Invalid review-secret setup arguments.")
    exit(1)
}
let ghPath = args[1]
let approvedPrefixes = ["/opt/homebrew/", "/usr/local/", "/usr/bin/"]
guard approvedPrefixes.contains(where: { ghPath.hasPrefix($0) }),
      FileManager.default.isExecutableFile(atPath: ghPath) else {
    print("The GitHub CLI is unavailable in a standard installation location.")
    exit(1)
}
let subscription = args[3] == "subscription"
let claudeSecret = subscription ? "CLAUDE_CODE_OAUTH_TOKEN" : "ANTHROPIC_API_KEY"
if args.count == 5 {
    print("Review credential helper compiled; metadata check passed. No dialog or secret access.")
    exit(0)
}

func validToken(_ token: String, openAI: Bool) -> Bool {
    return !token.isEmpty && token.utf8.count <= 16384 &&
        token.unicodeScalars.allSatisfy({ $0.value >= 33 && $0.value <= 126 }) &&
        (!openAI || token.hasPrefix("sk-"))
}

// Only metadata/success leave this function; subprocess diagnostic text is discarded.
func store(_ name: String, token: String) -> Bool {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: ghPath)
    process.arguments = ["secret", "set", name, "--repo", repository]
    process.environment = [
        "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "GH_HOST": "github.com", "GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1",
        "GH_NO_EXTENSION_UPDATE_NOTIFIER": "1", "NO_COLOR": "1"
    ]
    let pipe = Pipe()
    process.standardInput = pipe
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    do {
        try process.run()
        // Enforce a bound even if networking or authentication hangs. A timeout's
        // result is uncertain: GitHub could already have stored the encrypted value.
        let watchdog = DispatchWorkItem {
            if process.isRunning {
                process.terminate()
                DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                    if process.isRunning { kill(process.processIdentifier, SIGKILL) }
                }
            }
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + 60, execute: watchdog)
        try pipe.fileHandleForWriting.write(contentsOf: Data(token.utf8))
        try pipe.fileHandleForWriting.close()
        process.waitUntilExit()
        watchdog.cancel()
        return process.terminationReason == .exit && process.terminationStatus == 0
    } catch {
        try? pipe.fileHandleForWriting.close()
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
            process.waitUntilExit()
        }
        return false
    }
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
alert.messageText = "Configure Claude and Codex PR checks"
let billing = subscription
    ? "Claude uses your Claude Code subscription token (create with claude setup-token). Codex uses an OpenAI API key and API billing."
    : "Both providers use API credentials and API billing."
alert.informativeText = "Destination: GitHub.com Actions repository secrets for \(repository) ONLY.\n\n\(billing)\n\nPaste credentials into the masked fields below. Store replaces these named repository secrets. Values are sent directly to the authenticated GitHub CLI through its input pipe; they are never returned to chat or saved locally. GitHub CLI must already be logged in with permission to manage this repository's secrets. Cancel makes no changes."
alert.alertStyle = .informational
alert.addButton(withTitle: "Store in GitHub")
alert.addButton(withTitle: "Cancel")
let container = NSView(frame: NSRect(x: 0, y: 0, width: 480, height: 124))
let openAILabel = NSTextField(labelWithString: "OPENAI_API_KEY")
openAILabel.frame = NSRect(x: 0, y: 102, width: 480, height: 20)
let openAIField = NSSecureTextField(frame: NSRect(x: 0, y: 72, width: 480, height: 26))
openAIField.placeholderString = "OpenAI API key (sk-…)"
openAIField.setAccessibilityLabel("OpenAI API key")
let claudeLabel = NSTextField(labelWithString: claudeSecret)
claudeLabel.frame = NSRect(x: 0, y: 42, width: 480, height: 20)
let claudeField = NSSecureTextField(frame: NSRect(x: 0, y: 12, width: 480, height: 26))
claudeField.placeholderString = subscription ? "Claude Code subscription OAuth token" : "Anthropic API key"
claudeField.setAccessibilityLabel(claudeSecret)
for view in [openAILabel, openAIField, claudeLabel, claudeField] { container.addSubview(view) }
alert.accessoryView = container
alert.window.initialFirstResponder = openAIField
let response = alert.runModal()
guard response == .alertFirstButtonReturn else {
    openAIField.stringValue = ""
    claudeField.stringValue = ""
    print("Credential entry cancelled; no GitHub secrets were changed.")
    exit(2)
}
let openAIToken = openAIField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
let claudeToken = claudeField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
openAIField.stringValue = ""
claudeField.stringValue = ""
guard validToken(openAIToken, openAI: true), validToken(claudeToken, openAI: false) else {
    print("No secrets sent: both entries must be printable tokens (at most 16 KiB); OpenAI must start sk-.")
    exit(1)
}
let openAISuccess = store("OPENAI_API_KEY", token: openAIToken)
let claudeSuccess = store(claudeSecret, token: claudeToken)
let result = NSAlert()
result.messageText = openAISuccess && claudeSuccess ? "Review credentials configured" : "Credential setup incomplete"
result.informativeText = "\(repository)\nOPENAI_API_KEY: \(openAISuccess ? "stored" : "not confirmed")\n\(claudeSecret): \(claudeSuccess ? "stored" : "not confirmed")\n\nValues are not shown. An unconfirmed write may still have reached GitHub; verify permissions and retry if needed. This does not verify provider access or run a PR review."
result.addButton(withTitle: "OK")
application.activate(ignoringOtherApps: true)
_ = result.runModal()
print(openAISuccess && claudeSuccess
    ? "Both named repository secrets were stored; no values were returned."
    : "Credential storage is incomplete; no values or CLI diagnostics were returned.")
exit(openAISuccess && claudeSuccess ? 0 : 1)
