import SwiftUI

struct ContentView: View {
    @StateObject private var vm = TranslatorViewModel()
    @State private var showKey = false

    var body: some View {
        VStack(spacing: 12) {

            // -- API Key --
            HStack {
                Group {
                    if showKey {
                        TextField("sk-... (OpenAI)", text: $vm.apiKey)
                    } else {
                        SecureField("sk-... (OpenAI)", text: $vm.apiKey)
                    }
                }
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)

                Button(showKey ? "Ocultar" : "Mostrar") { showKey.toggle() }
                    .font(.footnote)
            }

            // -- Selectores --
            HStack(spacing: 12) {
                Picker("Traducción", selection: $vm.translationModel) {
                    ForEach(translationModels, id: \.self) { Text($0) }
                }
                Picker("Grabación", selection: $vm.recordingMode) {
                    ForEach(RecordingMode.allCases) { Text($0.rawValue).tag($0) }
                }
            }
            .pickerStyle(.menu)
            .disabled(vm.isRunning)

            // -- Subtítulos --
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        ForEach(vm.subtitles) { sub in
                            VStack(alignment: .leading, spacing: 4) {
                                Text("🇯🇵 \(sub.japanese)")
                                    .font(.headline)
                                Text("🇪🇸 \(sub.spanish)")
                                    .font(.title3)
                                if sub.elapsed > 0 {
                                    Text(String(format: "%.1f s", sub.elapsed))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Divider()
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(sub.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: vm.subtitles.count) {
                    if let last = vm.subtitles.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))

            // -- Estado + control --
            Text(vm.status)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            Button(action: vm.toggle) {
                Text(vm.isRunning ? "⏹  Detener" : "▶  Iniciar")
                    .fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent)
            .tint(vm.isRunning ? .red : .green)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
