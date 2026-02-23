# Transliterate JP→ES

Aplicación de escritorio para Windows que captura audio del sistema en tiempo real, transcribe japonés usando OpenAI Whisper y traduce a español con GPT.

## Flujo

```
Audio del sistema → Captura loopback → Whisper (STT japonés) → GPT (traducción JP→ES) → Subtítulos en pantalla
```

## Requisitos

- Python 3.10+
- Windows 10/11
- API Key de OpenAI

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python app.py
```

1. Ingresa tu API Key de OpenAI
2. Selecciona el dispositivo de audio (speakers/auriculares del sistema)
3. Elige los modelos de STT y traducción
4. Presiona **Iniciar** para comenzar la captura

## Dependencias

| Paquete | Uso |
|---------|-----|
| `customtkinter` | Interfaz gráfica moderna |
| `openai` | Whisper STT + GPT traducción |
| `soundcard` | Captura de audio del sistema (loopback) |
| `numpy` | Procesamiento de señal de audio |
