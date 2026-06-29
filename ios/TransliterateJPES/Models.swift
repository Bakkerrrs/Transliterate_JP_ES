import Foundation

/// Un bloque de subtítulo: texto japonés transcrito + su traducción al español.
struct Subtitle: Identifiable {
    let id = UUID()
    let japanese: String
    var spanish: String = ""
    var elapsed: Double = 0
}

/// Modos de grabación, equivalentes a los de la app de escritorio.
enum RecordingMode: String, CaseIterable, Identifiable {
    case fixed4 = "Fijo (4s)"
    case fixed2 = "Fijo (2s)"
    case vad = "VAD (detección de voz)"

    var id: String { rawValue }
}

/// Modelos de traducción GPT disponibles (mismos que models.txt del proyecto Python).
let translationModels = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.1",
]
