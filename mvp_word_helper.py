# requirements (install via pip):
# faster-whisper==1.*  sounddevice webrtcvad numpy scipy torch
# sentence-transformers wordfreq pronouncing nltk
# (for Windows CPU-only whisper you may want: pip install torch --index-url https://download.pytorch.org/whl/cpu)

import queue, threading, time, sys, re, json, os, math
import numpy as np
import sounddevice as sd
import webrtcvad

from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from wordfreq import zipf_frequency
import pronouncing

def normalize_token(w: str) -> str:
    w = w.lower()
    # ultra-light stemming to catch simple repeats
    for suf in ("'s", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            w = w[: -len(suf)]
            break
    return w

# ---------------------------
# Config
# ---------------------------
ASR_MODEL = "small"        # try "small" or "medium" depending on your hardware
DEVICE = "cpu"              # 'cuda' if you have a GPU
VAD_FRAME_MS = 20           # 10/20/30 ms frames allowed by webrtcvad
SAMPLE_RATE = 16000
CHANNELS = 1
CONTEXT_SECONDS = 15        # window for suggestions
TOP_K = 8
PERSONAL_VOCAB_PATH = "./personal_vocab.txt"  # one word/phrase per line

# Optional: letter / sound hint controls (connect to UI fields)
HINT_FIRST_LETTER = None    # e.g., 'b'
HINT_SYLLABLES = None       # e.g., 2

# ---------------------------
# Audio + VAD
# ---------------------------
audio_q = queue.Queue()
vad = webrtcvad.Vad(2)  # 0-3 (3 is most aggressive)

def mic_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    # convert to 16-bit mono samples as bytes
    pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    audio_q.put(pcm16)

def audio_stream():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        blocksize=int(SAMPLE_RATE * VAD_FRAME_MS / 1000),
        callback=mic_callback,
    ):
        while True:
            time.sleep(0.1)

# ---------------------------
# Rolling transcript store
# ---------------------------
class RollingTranscript:
    
    def __init__(self, max_seconds=120):
        self.segments = []  # list of (t, text)
        self.max_seconds = max_seconds

    def add(self, text):
        t = time.time()
        self.segments.append((t, text))
        self.trim()

    def trim(self):
        cutoff = time.time() - self.max_seconds
        self.segments = [(t, s) for (t, s) in self.segments if t >= cutoff]

    def get_context(self, seconds):
        cutoff = time.time() - seconds
        text = " ".join(s for (t, s) in self.segments if t >= cutoff)
        return text.strip()
    
    def normalize_token(w: str) -> str:
        w = w.lower()
        # super-light stemming to catch simple variants
        for suf in ("'s", "ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                w = w[: -len(suf)]
                break
        return w

    def recent_words_with_ages(self, window_seconds=20):
        """Return {normalized_word: age_seconds} for words in the last window."""
        WORD_RE = re.compile(r"[A-Za-z]+")
        now = time.time()
        cutoff = now - window_seconds
        seen = {}
        for (t, s) in self.segments:
            if t < cutoff: continue
            for w in WORD_RE.findall(s):
                nw = normalize_token(w)
                age = now - t
                # keep the *youngest* (smallest age) occurrence
                seen[nw] = min(seen.get(nw, 1e9), age)
        return seen
    


transcript = RollingTranscript(max_seconds=300)

# ---------------------------
# ASR worker (chunked)
# ---------------------------
def asr_worker():
    model = WhisperModel(ASR_MODEL, device=DEVICE, compute_type="int8")
    buf = b""
    bytes_per_frame = int(SAMPLE_RATE * (VAD_FRAME_MS/1000)) * 2  # 16-bit mono
    silence_bytes = b"\x00" * bytes_per_frame
    speech_chunk = b""
    last_speech_time = time.time()
    speaking = False

    while True:
        try:
            frame = audio_q.get(timeout=1.0)
        except queue.Empty:
            frame = silence_bytes

        is_speech = vad.is_speech(frame, SAMPLE_RATE)

        if is_speech:
            speaking = True
            last_speech_time = time.time()
            speech_chunk += frame
        else:
            # if we were speaking and now silence, flush after short tail
            if speaking and (time.time() - last_speech_time) > 0.35:
                # decode this speech_chunk
                audio_np = np.frombuffer(speech_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                segments, _ = model.transcribe(audio_np, language="en", beam_size=1, vad_filter=False)
                for seg in segments:
                    piece = seg.text.strip()
                    if piece:
                        transcript.add(piece)
                        print("[ASR]", piece)
                speech_chunk = b""
                speaking = False

# ---------------------------
# Suggestion Engine
# ---------------------------
class SuggestionEngine:
    def __init__(self, personal_vocab_path):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.personal_vocab = self._load_personal_vocab(personal_vocab_path)
        self.core_vocab = self._load_core_vocab()
        self.candidate_cache = {}  # word -> embedding
        self.repetition_weight = 0.45
        self.repetition_tau    = 18.0

        # build embeddings for personal vocab (do once)
        _ = self._embed_list(self.personal_vocab)

    def _load_personal_vocab(self, path):
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            words = [w.strip() for w in f if w.strip()]
        return list(dict.fromkeys(words))  # unique, preserve order

    def _load_core_vocab(self, top_n=5000):
        # you can replace this with a curated list; for demo, a small seed:
        seed = [
            "appointment","breakfast","coffee","doctor","nurse","medication",
            "glasses","television","bathroom","kitchen","remote","charger",
            "daughter","son","grandson","granddaughter","neighbor","friend",
            "church","pharmacy","grocery","walker","wheelchair","keys","phone",
            "wallet","jacket","mailbox","bus","taxi","ride","library","park","three","house","dock","pier","lake"
        ]
        return seed

    def _embed_list(self, words):
        to_compute = [w for w in words if w not in self.candidate_cache]
        if to_compute:
            embs = self.embedder.encode(to_compute, normalize_embeddings=True)
            for w, e in zip(to_compute, embs):
                self.candidate_cache[w] = e
        return np.array([self.candidate_cache[w] for w in words])

    def _semantic_scores(self, context, candidates):
        if not context.strip():
            return np.zeros(len(candidates))
        ctx_emb = self.embedder.encode([context], normalize_embeddings=True)[0]
        cand_embs = self._embed_list(candidates)
        return (cand_embs @ ctx_emb)  # cosine since normalized

    def _freq_prior(self, word):
        # Zipf 1-7; clamp and scale
        return max(0.0, zipf_frequency(word, "en") - 3.0) / 4.0

    def _phonetic_score(self, word, hint_first_letter=None, hint_syllables=None):
        score = 0.0
        if hint_first_letter:
            score += 0.5 if word.lower().startswith(hint_first_letter.lower()) else 0.0
        if hint_syllables is not None:
            # crude syllable match via CMU dict:
            phones = pronouncing.phones_for_word(word.lower())
            if phones:
                syls = [pronouncing.syllable_count(p) for p in phones]
                if syls and min(abs(s - hint_syllables) for s in syls) == 0:
                    score += 0.3
        return score
    
    def _topic_candidates(self, context: str):
            # combine and dedupe while preserving order
            return list(dict.fromkeys(self.personal_vocab + self.core_vocab))

    def suggest(self, context_text, top_k=8, hint_first_letter=None, hint_syllables=None, recent_map=None):
        # build candidate pool
        candidates = self._topic_candidates(context_text)  # or your current candidate builder

        # rank components
        sem = self._semantic_scores(context_text, candidates)
        pri = np.array([self._freq_prior(w) for w in candidates])
        pho = np.array([self._phonetic_score(w, hint_first_letter, hint_syllables) for w in candidates])
        rep = self._repetition_penalties(candidates, recent_map or {})

        # weighted blend (tune these)
        scores = 0.85*sem + 0.10*pri + 0.05*pho + rep
        
        

        # sort & return
        idx = np.argsort(-scores)
        ranked = [(candidates[i], float(scores[i])) for i in idx[:top_k]]
        return ranked
    
    def _repetition_penalties(self, candidates, recent_map):
        """Return negative penalties for candidates recently spoken.
        recent_map: {normalized_word: age_seconds} from transcript."""
        if not recent_map:
            return np.zeros(len(candidates))
        pen = np.zeros(len(candidates), dtype=np.float32)
        for i, w in enumerate(candidates):
            nw = normalize_token(w)
            if nw in recent_map:
                age = recent_map[nw]
                # exponential decay: very recent → strong penalty; fades with time
                pen[i] = - self.repetition_weight * math.exp(-age / self.repetition_tau)
        return pen

engine = SuggestionEngine(PERSONAL_VOCAB_PATH)

# ---------------------------
# UI stub: button → suggestions
# ---------------------------
def on_button_press():
    context = transcript.get_context(CONTEXT_SECONDS)
    recent = transcript.recent_words_with_ages(window_seconds=12) # tune 12–30s
    suggestions = engine.suggest(
        context,
        top_k=TOP_K,
        hint_first_letter=HINT_FIRST_LETTER,
        hint_syllables=HINT_SYLLABLES,
        recent_map=recent
    )
    print("\n--- SUGGESTIONS ---")
    for w, s in suggestions:
        print(f"{w:20s}  {s:.3f}")
    print("-------------------\n")

# ---------------------------
# Launch threads
# ---------------------------
if __name__ == "__main__":
    t_audio = threading.Thread(target=audio_stream, daemon=True)
    t_asr = threading.Thread(target=asr_worker, daemon=True)
    t_audio.start()
    t_asr.start()

    print("Live. Press Enter to show suggestions. Ctrl+C to exit.")
    try:
        while True:
            input()  # simulate the big on-screen button
            on_button_press()
    except KeyboardInterrupt:
        pass
