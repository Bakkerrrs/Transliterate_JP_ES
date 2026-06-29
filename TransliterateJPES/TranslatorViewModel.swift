import Foundation
import SwiftUI

/// Orquesta captura de audio → STT → traducción y publica el estado para la UI.
@MainActor
final class TranslatorViewModel: ObservableObject {

    /// Se guarda automáticamente en el Keychain al cambiar.
    @Published var apiKey: String = "" {
        didSet { KeychainStore.save(apiKey) }
    }
    @Published var translationModel = translationModels[0]
    @Published var recordingMode: RecordingMode = .vad
    @Published var isRunning = false
    @Published var status = "Detenido"
    @Published var subtitles: [Subtitle] = []

    private let audio = AudioCapture()
    private var service: OpenAIService?

    init() {
        // Asignar en init NO dispara didSet, así no reescribimos el Keychain al cargar.
        // Prioridad: variable de entorno OPENAI_API_KEY (útil al desarrollar desde
        // Xcode) y, si no existe, la key guardada en el Keychain.
        if let envKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"],
           !envKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            apiKey = envKey
        } else if let saved = KeychainStore.load() {
            apiKey = saved
        }
    }

    func toggle() {
        isRunning ? stop() : start()
    }

    /// Borra la API Key guardada.
    func clearKey() {
        apiKey = ""
    }

    func start() {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            status = "⚠ Ingresa tu API Key de OpenAI"
            return
        }

        audio.requestPermission { [weak self] granted in
            guard let self else { return }
            guard granted else {
                self.status = "⚠ Permiso de micrófono denegado"
                return
            }

            self.service = OpenAIService(apiKey: key, translationModel: self.translationModel)

            let mode: AudioCapture.Mode
            switch self.recordingMode {
            case .fixed4: mode = .fixed(seconds: 4)
            case .fixed2: mode = .fixed(seconds: 2)
            case .vad:    mode = .vad
            }

            do {
                try self.audio.start(mode: mode) { [weak self] wav, _ in
                    self?.handleChunk(wav)
                }
                self.isRunning = true
                self.status = "🎧 Escuchando..."
            } catch {
                self.status = "⚠ Error de audio: \(error.localizedDescription)"
            }
        }
    }

    func stop() {
        audio.stop()
        isRunning = false
        status = "Detenido"
    }

    /// Llamado desde el hilo de audio por cada chunk WAV listo.
    private func handleChunk(_ wav: Data) {
        guard let service else { return }
        let start = Date()

        Task {
            do {
                let jp = try await service.transcribe(wav: wav)
                guard !jp.isEmpty else { return }

                let sub = Subtitle(japanese: jp)
                await MainActor.run {
                    self.subtitles.append(sub)
                    self.status = "🌐 Traduciendo..."
                }

                try await service.translateStream(japanese: jp) { token in
                    Task { @MainActor in
                        if let i = self.subtitles.firstIndex(where: { $0.id == sub.id }) {
                            self.subtitles[i].spanish += token
                        }
                    }
                }

                let elapsed = Date().timeIntervalSince(start)
                await MainActor.run {
                    if let i = self.subtitles.firstIndex(where: { $0.id == sub.id }) {
                        self.subtitles[i].elapsed = elapsed
                    }
                    self.status = self.isRunning ? "🎧 Escuchando..." : "Detenido"
                }
            } catch {
                await MainActor.run {
                    self.status = "⚠ \(error.localizedDescription)"
                }
            }
        }
    }
}
