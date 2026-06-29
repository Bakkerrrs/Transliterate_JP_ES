import Foundation

/// Cliente de la API de OpenAI: transcripción Whisper (japonés) + traducción GPT (JP→ES).
///
/// Es un `actor` para que el contexto de conversación se acceda de forma segura
/// aunque varios chunks se procesen en paralelo.
actor OpenAIService {

    private let apiKey: String
    private var translationModel: String

    private var context: [String] = []
    private let contextLimit = 5

    init(apiKey: String, translationModel: String) {
        self.apiKey = apiKey
        self.translationModel = translationModel
    }

    func updateTranslationModel(_ model: String) {
        translationModel = model
    }

    // MARK: - STT (Whisper API)

    func transcribe(wav: Data) async throws -> String {
        let url = URL(string: "https://api.openai.com/v1/audio/transcriptions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        func field(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(value)\r\n".data(using: .utf8)!)
        }
        field("model", "whisper-1")
        field("language", "ja")

        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(wav)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        let (data, response) = try await URLSession.shared.upload(for: request, from: body)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw OpenAIError.http(String(data: data, encoding: .utf8) ?? "Error de transcripción")
        }
        struct Resp: Decodable { let text: String }
        let decoded = try JSONDecoder().decode(Resp.self, from: data)
        return decoded.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Traducción (GPT streaming)

    /// Traduce `japanese` al español, invocando `onToken` por cada token recibido.
    func translateStream(japanese: String, onToken: @Sendable (String) -> Void) async throws {
        guard !japanese.isEmpty else { return }

        let url = URL(string: "https://api.openai.com/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let contextStr = context.isEmpty ? "(sin contexto previo)" : context.joined(separator: "\n")

        // Mismo system prompt que app.py
        let systemPrompt = """
        Eres un traductor profesional de japonés a español. \
        Traduce el texto japonés al español de forma natural y fluida. \
        Mantén el tono y registro del original. \
        Si el texto contiene onomatopeyas o expresiones culturales japonesas, \
        adapta al equivalente más cercano en español. \
        Responde SOLO con la traducción, sin explicaciones.
        """
        let userPrompt = """
        Contexto previo de la conversación:
        \(contextStr)

        Traduce al español:
        \(japanese)
        """

        let payload: [String: Any] = [
            "model": translationModel,
            "messages": [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": userPrompt],
            ],
            "temperature": 0.3,
            "max_completion_tokens": 500,
            "stream": true,
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw OpenAIError.http("Error de traducción (HTTP \(code))")
        }

        var full = ""
        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let payloadStr = String(line.dropFirst(6))
            if payloadStr == "[DONE]" { break }
            guard let d = payloadStr.data(using: .utf8) else { continue }
            if let token = Self.extractDelta(d) {
                full += token
                onToken(token)
            }
        }

        context.append("JP: \(japanese)\nES: \(full)")
        if context.count > contextLimit {
            context.removeFirst(context.count - contextLimit)
        }
    }

    private static func extractDelta(_ data: Data) -> String? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = obj["choices"] as? [[String: Any]],
              let delta = choices.first?["delta"] as? [String: Any],
              let content = delta["content"] as? String else { return nil }
        return content
    }
}

enum OpenAIError: Error, LocalizedError {
    case http(String)
    var errorDescription: String? {
        switch self {
        case .http(let message): return message
        }
    }
}
