// SPDX-License-Identifier: AGPL-3.0-or-later
// Explicitly started default microphone. No files, network, credentials or transcript logs.
import AVFoundation
import Darwin
import Foundation

func status(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments == ["--check"] {
    let permission = AVCaptureDevice.authorizationStatus(for: .audio)
    switch permission {
    case .authorized: print("authorized")
    case .denied: print("denied")
    case .restricted: print("restricted")
    case .notDetermined: print("not-determined")
    @unknown default: print("unavailable")
    }
    exit(0)
}
guard arguments == ["--capture"] else {
    status("Usage: RightyOMicrophone --check | --capture")
    exit(2)
}

let engine = AVAudioEngine()
var tapInstalled = false
var stopping = false
let activityLock = NSLock()
var lastBuffer = DispatchTime.now().uptimeNanoseconds
var inputWatchdog: DispatchSourceTimer?
func stop(_ code: Int32, _ message: String) {
    guard !stopping else { return }
    stopping = true
    inputWatchdog?.cancel()
    if tapInstalled { engine.inputNode.removeTap(onBus: 0) }
    engine.stop()
    status(message)
    exit(code)
}

// Dispatch sources avoid invoking AVFoundation from an async POSIX signal handler.
signal(SIGTERM, SIG_IGN)
signal(SIGINT, SIG_IGN)
signal(SIGPIPE, SIG_IGN)
let termination = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
termination.setEventHandler { stop(0, "STOPPED") }
termination.resume()
let interruption = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
interruption.setEventHandler { stop(0, "STOPPED") }
interruption.resume()

// A full pipe fails closed; capture must not keep accumulating stale audio.
let descriptorFlags = fcntl(STDOUT_FILENO, F_GETFL)
guard descriptorFlags >= 0,
      fcntl(STDOUT_FILENO, F_SETFL, descriptorFlags | O_NONBLOCK) >= 0 else {
    stop(17, "CAPTURE_FAILED")
    exit(17)
}

func beginCapture() {
    guard !stopping else { return }
    let input = engine.inputNode
    let format = input.outputFormat(forBus: 0)
    guard format.sampleRate.isFinite, format.sampleRate > 0, format.channelCount > 0,
          let target = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                    sampleRate: 16_000, channels: 1, interleaved: false),
          let converter = AVAudioConverter(from: format, to: target) else {
        stop(13, "NO_INPUT")
        return
    }
    converter.downmix = true
    // A bounded callback; converted Float32 is clamped to little-endian PCM16.
    input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
        activityLock.lock()
        lastBuffer = DispatchTime.now().uptimeNanoseconds
        activityLock.unlock()
        let needed = ceil(Double(buffer.frameLength) * 16_000 / format.sampleRate) + 1024
        guard needed.isFinite, needed > 0, needed <= 16_384 else {
            DispatchQueue.main.async { stop(14, "CONVERSION_FAILED") }
            return
        }
        let capacity = AVAudioFrameCount(needed)
        guard
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
            DispatchQueue.main.async { stop(14, "CONVERSION_FAILED") }
            return
        }
        var supplied = false
        var conversionError: NSError?
        let result = converter.convert(to: output, error: &conversionError) { _, state in
            if supplied {
                state.pointee = .noDataNow
                return nil
            }
            supplied = true
            state.pointee = .haveData
            return buffer
        }
        guard result != .error, conversionError == nil,
              let samples = output.floatChannelData?[0] else {
            DispatchQueue.main.async { stop(14, "CONVERSION_FAILED") }
            return
        }
        var pcm = [Int16](repeating: 0, count: Int(output.frameLength))
        for index in pcm.indices {
            let sample = samples[index]
            guard sample.isFinite else {
                DispatchQueue.main.async { stop(14, "CONVERSION_FAILED") }
                return
            }
            pcm[index] = Int16(max(-32768, min(32767, (sample * 32768).rounded()))).littleEndian
        }
        let succeeded = pcm.withUnsafeBytes { bytes -> Bool in
            var offset = 0
            while offset < bytes.count {
                let count = Darwin.write(STDOUT_FILENO,
                                         bytes.baseAddress!.advanced(by: offset),
                                         bytes.count - offset)
                if count <= 0 { return false }
                offset += count
            }
            return true
        }
        if !succeeded {
            DispatchQueue.main.async { stop(15, "CAPTURE_BACKPRESSURE") }
        }
    }
    tapInstalled = true
    engine.prepare()
    do { try engine.start() }
    catch { stop(17, "CAPTURE_FAILED"); return }
    // Some routes can start without ever delivering a tap callback. Fail closed.
    activityLock.lock()
    lastBuffer = DispatchTime.now().uptimeNanoseconds
    activityLock.unlock()
    let watchdog = DispatchSource.makeTimerSource(queue: .main)
    watchdog.schedule(deadline: .now() + 2, repeating: 2)
    watchdog.setEventHandler {
        activityLock.lock()
        let last = lastBuffer
        activityLock.unlock()
        if DispatchTime.now().uptimeNanoseconds - last > 10_000_000_000 {
            stop(13, "NO_INPUT")
        }
    }
    inputWatchdog = watchdog
    watchdog.resume()
    status("READY")
}

let notification = NotificationCenter.default.addObserver(
    forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
) { _ in stop(16, "INPUT_CHANGED") }

switch AVCaptureDevice.authorizationStatus(for: .audio) {
case .authorized: beginCapture()
case .notDetermined:
    status("PERMISSION_PENDING")
    AVCaptureDevice.requestAccess(for: .audio) { allowed in
        DispatchQueue.main.async {
            if allowed { beginCapture() }
            else { stop(12, "PERMISSION_DENIED") }
        }
    }
case .denied, .restricted: stop(12, "PERMISSION_DENIED")
@unknown default: stop(17, "CAPTURE_FAILED")
}
RunLoop.main.run()
