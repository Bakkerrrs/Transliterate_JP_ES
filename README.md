# Transliterate JP→ES — versión iOS (starter)

App iOS en **SwiftUI** que escucha por el **micrófono** el japonés que suena
alrededor (TV, parlante, una persona), lo transcribe con **OpenAI Whisper API**
y lo traduce al español con **GPT** (streaming).

> Originalmente fue una app de escritorio en Python (Windows). Esta es la
> reescritura nativa para iOS; el código Python permanece en el historial de Git.

## ⚠️ Captura de audio en iOS

**En iOS no es posible capturar el audio del sistema o de otras apps**: el
sandbox lo impide. Por eso la entrada es el **micrófono** — el dispositivo
escucha el japonés que suena alrededor. Es la única vía nativa y soportada
para este tipo de app.

## Requisitos

- **macOS** con **Xcode 16 o superior** (el proyecto usa grupos sincronizados
  con el sistema de archivos, `objectVersion = 77`).
- Un **iPhone/iPad con iOS 17+** para probar (el simulador no tiene micrófono
  real útil; usa un dispositivo físico).
- Una **API Key de OpenAI**.
- Para instalar en tu dispositivo: una Apple ID (perfil gratuito de 7 días) o el
  **Apple Developer Program** ($99/año) para publicar en la App Store.

## Cómo abrir y ejecutar

1. Abre `TransliterateJPES.xcodeproj` en Xcode.
2. Selecciona el target **TransliterateJPES** y, en *Signing & Capabilities*,
   elige tu *Team* (tu Apple ID). Cambia el *Bundle Identifier*
   `com.example.TransliterateJPES` por uno tuyo si hace falta.
3. Conecta tu iPhone por cable y selecciónalo como destino.
4. Pulsa ▶ (Run). Acepta el permiso de micrófono cuando lo pida.
5. Pega tu API Key, elige modelo y modo de grabación, y pulsa **Iniciar**.

## Estructura

| Archivo | Rol |
|---|---|
| `TransliterateJPESApp.swift` | Punto de entrada |
| `ContentView.swift` | UI SwiftUI |
| `TranslatorViewModel.swift` | Orquestación rec→STT→traducción |
| `AudioCapture.swift` | Micrófono + VAD + WAV |
| `OpenAIService.swift` | Whisper API + GPT streaming |
| `Models.swift` | Tipos de datos |

## Próximos pasos sugeridos

- **STT en el dispositivo (offline):** sustituir Whisper API por `SFSpeechRecognizer`
  (framework Speech de Apple, gratis) o integrar **whisper.cpp** con Metal.
- **Guardar la API Key** de forma segura en el **Keychain** (hoy solo vive en memoria).
- **Indicadores de pipeline** por etapa (rec/STT/trad).
- **Modelos de traducción** configurables desde un archivo o ajustes.

## Notas

- La API Key no se persiste; se ingresa en cada sesión. No la subas al repo.
- El audio se envía a la API de OpenAI; revisa los términos según tu caso de uso.
