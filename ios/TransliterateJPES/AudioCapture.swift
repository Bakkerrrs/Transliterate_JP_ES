import Foundation
import AVFoundation

/// Captura audio del **micrófono**, lo trocea en chunks de habla (fijo o VAD),
/// los convierte a WAV mono 16 kHz de 16 bits y los entrega por un callback.
///
/// IMPORTANTE: en iOS no es posible capturar el audio del sistema o de otras
/// apps (sandbox). Por eso la entrada es el micrófono — el dispositivo "escucha"
/// el japonés que suena alrededor (TV, parlante, una persona).
///
/// La lógica de VAD reproduce la de la app de escritorio (app.py).
final class AudioCapture {

    enum Mode {
        case fixed(seconds: Double)
        case vad
    }

    static let sampleRate: Double = 16000

    // Parámetros de VAD (reflejan AudioCapture en app.py)
    private let vadMinSpeech = 0.8          // segundos mínimos de habla antes de enviar
    private let vadMaxSpeech = 7.0          // segundos máximos antes de forzar envío
    private let vadSilenceTimeout = 0.6     // silencio tras habla que dispara el envío
    private let vadSpeechThreshold: Float = 0.005   // umbral RMS de detección de voz
    private let preBufferSeconds = 0.5      // pre-buffer para no cortar el inicio

    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let outputFormat: AVAudioFormat
    private var isRunning = false

    // Acumulación de chunk (todo se toca solo dentro de `queue`)
    private var mode: Mode = .vad
    private var buffer: [Float] = []
    private var speechStarted = false
    private var silenceSeconds = 0.0
    private var onChunk: ((Data, Float) -> Void)?
    private let queue = DispatchQueue(label: "audio.capture")

    init() {
        outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: AudioCapture.sampleRate,
            channels: 1,
            interleaved: false
        )!
    }

    // MARK: - Permisos

    func requestPermission(_ completion: @escaping (Bool) -> Void) {
        AVAudioApplication.requestRecordPermission { granted in
            DispatchQueue.main.async { completion(granted) }
        }
    }

    // MARK: - Control

    func start(mode: Mode, onChunk: @escaping (Data, Float) -> Void) throws {
        guard !isRunning else { return }
        self.mode = mode
        self.onChunk = onChunk
        queue.sync {
            buffer.removeAll()
            speechStarted = false
            silenceSeconds = 0
        }

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: [])
        try session.setActive(true)

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        converter = AVAudioConverter(from: inputFormat, to: outputFormat)

        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buf, _ in
            self?.queue.async { self?.process(buffer: buf) }
        }

        engine.prepare()
        try engine.start()
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false)
        queue.async { self.buffer.removeAll() }
    }

    // MARK: - Conversión a 16 kHz mono

    private func process(buffer inBuf: AVAudioPCMBuffer) {
        guard let converter else { return }
        let ratio = AudioCapture.sampleRate / inBuf.format.sampleRate
        let capacity = AVAudioFrameCount(Double(inBuf.frameLength) * ratio + 1)
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }

        var fed = false
        var error: NSError?
        converter.convert(to: outBuf, error: &error) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return inBuf
        }
        if error != nil { return }

        guard let ch = outBuf.floatChannelData else { return }
        let frames = Int(outBuf.frameLength)
        guard frames > 0 else { return }
        let samples = Array(UnsafeBufferPointer(start: ch[0], count: frames))
        appendSamples(samples)
    }

    // MARK: - Troceado

    private func appendSamples(_ samples: [Float]) {
        switch mode {
        case .fixed(let seconds):
            buffer.append(contentsOf: samples)
            let needed = Int(AudioCapture.sampleRate * seconds)
            while buffer.count >= needed {
                let chunk = Array(buffer.prefix(needed))
                buffer.removeFirst(needed)
                emit(chunk)
            }
        case .vad:
            vadAppend(samples)
        }
    }

    private func vadAppend(_ samples: [Float]) {
        let blockSeconds = Double(samples.count) / AudioCapture.sampleRate
        let level = rms(samples[...])
        buffer.append(contentsOf: samples)
        let preFrames = Int(AudioCapture.sampleRate * preBufferSeconds)

        if !speechStarted {
            if level >= vadSpeechThreshold {
                // Empieza el habla: conserva solo el pre-buffer + este bloque.
                speechStarted = true
                silenceSeconds = 0
                if buffer.count > preFrames {
                    buffer.removeFirst(buffer.count - preFrames)
                }
            } else {
                // Sin habla todavía: recorta el pre-buffer para no crecer sin límite.
                if buffer.count > preFrames {
                    buffer.removeFirst(buffer.count - preFrames)
                }
            }
        } else {
            let totalSeconds = Double(buffer.count) / AudioCapture.sampleRate
            if level < vadSpeechThreshold {
                silenceSeconds += blockSeconds
                if silenceSeconds >= vadSilenceTimeout && totalSeconds >= vadMinSpeech {
                    flushSpeech()
                    return
                }
            } else {
                silenceSeconds = 0
            }
            if totalSeconds >= vadMaxSpeech {
                flushSpeech()
            }
        }
    }

    private func flushSpeech() {
        let chunk = buffer
        buffer.removeAll()
        speechStarted = false
        silenceSeconds = 0
        emit(chunk)
    }

    private func emit(_ samples: [Float]) {
        let level = rms(samples[...])
        guard level >= 0.001 else { return }   // descarta silencio
        let wav = AudioCapture.makeWav(samples: samples, sampleRate: Int(AudioCapture.sampleRate))
        onChunk?(wav, level)
    }

    // MARK: - Helpers

    private func rms(_ samples: ArraySlice<Float>) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for s in samples { sum += s * s }
        return (sum / Float(samples.count)).squareRoot()
    }

    /// Construye un WAV PCM 16-bit mono en memoria (equivale a `_float_to_wav`).
    static func makeWav(samples: [Float], sampleRate: Int) -> Data {
        var data = Data()
        let numChannels = 1
        let bitsPerSample = 16
        let byteRate = sampleRate * numChannels * bitsPerSample / 8
        let blockAlign = numChannels * bitsPerSample / 8
        let dataSize = samples.count * 2

        func appendStr(_ s: String) { data.append(s.data(using: .ascii)!) }
        func appendUInt32(_ v: UInt32) { var x = v.littleEndian; data.append(Data(bytes: &x, count: 4)) }
        func appendUInt16(_ v: UInt16) { var x = v.littleEndian; data.append(Data(bytes: &x, count: 2)) }

        appendStr("RIFF")
        appendUInt32(UInt32(36 + dataSize))
        appendStr("WAVE")
        appendStr("fmt ")
        appendUInt32(16)
        appendUInt16(1)                       // PCM
        appendUInt16(UInt16(numChannels))
        appendUInt32(UInt32(sampleRate))
        appendUInt32(UInt32(byteRate))
        appendUInt16(UInt16(blockAlign))
        appendUInt16(UInt16(bitsPerSample))
        appendStr("data")
        appendUInt32(UInt32(dataSize))

        for s in samples {
            let clamped = max(-1.0, min(1.0, s))
            var le = Int16(clamped * 32767).littleEndian
            data.append(Data(bytes: &le, count: 2))
        }
        return data
    }
}
