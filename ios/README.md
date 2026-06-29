# Transliterate JP→ES — versión iOS (starter)

App iOS en **SwiftUI** que escucha por el **micrófono** el japonés que suena
alrededor (TV, parlante, una persona), lo transcribe con **OpenAI Whisper API**
y lo traduce al español con **GPT** (streaming). Es el equivalente iOS de la app
de escritorio `app.py`.

## ⚠️ Diferencia clave con la versión de escritorio

La app de Windows captura el **audio del sistema** (loopback). **En iOS eso no
es posible**: el sandbox impide que una app acceda al audio de otras apps o de
la salida del sistema. Por eso aquí la entrada es el **micrófono**. Es la única
vía nativa y soportada para este tipo de app.

## Requisitos

- **macOS** con **Xcode 16 o superior** (el proyecto usa grupos sincronizados
  con el sistema de archivos, `objectVersion = 77`).
- Un **iPhone/iPad con iOS 17+** para probar (el simulador no tiene micrófono
  real útil; usa un dispositivo físico).
- Una **API Key de OpenAI**.
- Para instalar en tu dispositivo: una Apple ID (perfil gratuito de 7 días) o el
  **Apple Developer Program** ($99/año) para publicar en la App Store.

## Cómo abrir y ejecutar

1. Abre `ios/TransliterateJPES.xcodeproj` en Xcode.
2. Selecciona el target **TransliterateJPES** y, en *Signing & Capabilities*,
   elige tu *Team* (tu Apple ID). Cambia el *Bundle Identifier*
   `com.example.TransliterateJPES` por uno tuyo si hace falta.
3. Conecta tu iPhone por cable y selecciónalo como destino.
4. Pulsa ▶ (Run). Acepta el permiso de micrófono cuando lo pida.
5. Pega tu API Key, elige modelo y modo de grabación, y pulsa **Iniciar**.

## Estructura

| Archivo | Rol | Equivalente en `app.py` |
|---|---|---|
| `TransliterateJPESApp.swift` | Punto de entrada | `main()` |
| `ContentView.swift` | UI SwiftUI | `SubtitleApp` (customtkinter) |
| `TranslatorViewModel.swift` | Orquestación rec→STT→traducción | `_pipe_worker` |
| `AudioCapture.swift` | Micrófono + VAD + WAV | `AudioCapture` |
| `OpenAIService.swift` | Whisper API + GPT streaming | `TranscriptionService` |
| `Models.swift` | Tipos de datos | constantes / estado |

## Próximos pasos sugeridos

- **STT en el dispositivo (offline):** sustituir Whisper API por `SFSpeechRecognizer`
  (framework Speech de Apple, gratis) o integrar **whisper.cpp** con Metal.
- **Guardar la API Key** de forma segura en el **Keychain** (hoy solo vive en memoria).
- **Indicadores de pipeline** por etapa (rec/STT/trad), como los semáforos de la app de escritorio.
- **Modelos de traducción** configurables desde un archivo, como `models.txt`.

## Notas

- La API Key no se persiste; se ingresa en cada sesión. No la subas al repo.
- El audio se envía a la API de OpenAI; revisa los términos según tu caso de uso.
