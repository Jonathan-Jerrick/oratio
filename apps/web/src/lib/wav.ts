export const TARGET_RATE = 16_000;
export const MAX_SECONDS = 170;

/** Decode any browser-readable audio, mix to mono, resample to 16 kHz and encode PCM16 WAV. */
export async function toWav16k(input: Blob): Promise<{ wav: Blob; seconds: number }> {
  const bytes = await input.arrayBuffer();
  const context = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await context.decodeAudioData(bytes);
  } catch {
    throw new Error("This file could not be read as audio. Try WAV, MP3, M4A, OGG or WebM.");
  } finally {
    void context.close();
  }
  if (decoded.duration > MAX_SECONDS) {
    throw new Error(
      `The recording is ${Math.round(decoded.duration)} s long. Trim it to ${MAX_SECONDS} s or less.`,
    );
  }
  if (decoded.duration < 1) throw new Error("The recording is shorter than one second.");
  const frames = Math.ceil(decoded.duration * TARGET_RATE);
  const offline = new OfflineAudioContext(1, frames, TARGET_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  const samples = rendered.getChannelData(0);
  return { wav: encodeWav(samples, TARGET_RATE), seconds: rendered.duration };
}

function encodeWav(samples: Float32Array, rate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i] as number));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
