<template>
  <main class="container">
    <h1 class="app-title">LiveTTS</h1>

    <!-- Text Synthesis -->
    <section class="card">
      <label class="label">Text to Synthesize</label>
      <textarea
        v-model="text"
        rows="7"
        class="textarea"
        placeholder="Enter your text here in the selected language..."
      ></textarea>

      <div class="row wrap gap-16 align-end">
        <div class="col">
          <label class="label">Select Language</label>
          <select v-model="selectedLanguage" class="select center italic" :disabled="!selectedVoice">
            <option value="">{{ selectedVoice ? 'Language' : 'Set voice' }}</option>
            <option v-for="lang in availableLanguagesForVoice" :key="lang" :value="lang">
              {{ getLanguageDisplayName(lang) }}
            </option>
          </select>
        </div>

        <div class="col">
          <label class="label">Select Voice</label>
          <select v-model="selectedVoice" class="select italic">
            <option value="">Not Selected</option>
            <option v-for="voice in voices" :key="voice.name" :value="voice">
              {{ voice.name }}<span v-if="voice.is_zero_shot"> (zero-shot)</span>
              <template v-else> (Original: {{ getLanguageDisplayName(voice.language) }})</template>
            </option>
          </select>
        </div>

        <div class="col">
          <label class="label">Playback Speed: {{ speed.toFixed(1) }}x</label>
          <div class="range-row">
            <span class="muted small">0.5x</span>
            <input type="range" v-model.number="speed" min="0.5" max="2.0" step="0.1" class="range" />
            <span class="muted small">2.0x</span>
          </div>
        </div>

        <button :disabled="loading || !isReadyForSynthesis" @click="synthesize" class="btn synthesize-btn">
          {{ loading ? "Generating..." : "Synthesize" }}
        </button>
      </div>

      <div v-if="errorMessage" class="alert alert-error">
        {{ errorMessage }}
      </div>
    </section>

    <!-- Zero-shot uploader -->
    <section class="card">
      <div class="row justify-end">
        <div class="upload-icon-wrapper"
             @click="triggerFileDialog"
             @mouseenter="showTooltip = true"
             @mouseleave="showTooltip = false">
          <span class="upload-icon" title="Upload reference.wav">📎</span>
          <input ref="fileInput" type="file" accept="audio/wav,.wav" class="hidden" @change="handleFileSelected" />
          <div v-if="showTooltip" class="tooltip">
            Upload a clean mono WAV (16k–24kHz, 5–15 seconds, up to 20MB).
          </div>
        </div>
      </div>

      
    </section>

    <div v-if="zeroShotActive" class="inline-info">
        <div><strong>Zero-shot voice ready:</strong> {{ zeroShotName }}</div>
        <button class="btn refresh-btn" @click="clearZeroShot" :disabled="loading">Clear</button>
      </div>

    <!-- Result -->
    <section v-if="audioUrl" class="card">
      <h3>Generated Audio</h3>
      <div class="row gap-8">
        <div><strong>Voice Used:</strong> {{ lastUsedVoice }}</div>
      </div>
      <audio :src="apiBase + audioUrl" controls class="audio"></audio>
      <div class="spacer-12">
        <a :href="apiBase + audioUrl" :download="filename" class="btn download-btn">Download</a>
      </div>
    </section>

    <!-- Voices -->
    <section class="card">
      <div class="row space-between align-center">
        <h3>Available Voices ({{ voices.length }})</h3>
        <button @click="refreshVoices" :disabled="refreshing" class="btn refresh-btn">
          {{ refreshing ? "Refreshing..." : "Refresh" }}
        </button>
      </div>

      <div v-if="voices.length === 0" class="empty">
        <p>No voices found. Please check the /volumes/voices/ directory.</p>
      </div>

      <div v-else class="grid">
        <div v-for="voice in voices" :key="voice.name"
             class="voice-card"
             :class="{ selected: selectedVoice?.name === voice.name }"
             @click="selectedVoice = voice">
          <h4 class="voice-title">
            {{ voice.name }}
            <span v-if="voice.is_zero_shot" class="badge">zero-shot</span>
          </h4>
          <p class="muted small">
            <template v-if="!voice.is_zero_shot">
              Original Language: {{ getLanguageDisplayName(voice.language) }}
            </template>
            <template v-else>
              Base Model: {{ voice.base_model || 'xtts_v2_base' }}
            </template>
          </p>
          <p class="muted small">
            Supported Languages:
            {{
              ((voice.supported_languages && voice.supported_languages.length ? voice.supported_languages
                : (voice.m_supported_languages || availableLanguages)))
                .map(lang => getLanguageDisplayName(lang))
                .join(', ')
            }}
          </p>
          <p v-if="voice.fixed_language" class="warn small">
            ⚠️ Fixed Language: {{ getLanguageDisplayName(voice.fixed_language) }}
          </p>
          <p class="muted xsmall">Status: {{ voice.available ? 'Available' : 'Unavailable' }}</p>
          <div v-if="selectedVoice?.name === voice.name" class="selected-badge">Selected</div>
        </div>
      </div>
    </section>

    <!-- System Info -->
    <section class="card muted-bg system-info">
      <h3>System Information</h3>
      <div class="sys-grid">
        <div><strong>Model:</strong> XTTS v2</div>
        <div><strong>Languages:</strong> {{ availableLanguages.length }} Supported</div>
        <div><strong>Voices Loaded:</strong> {{ voices.length }}</div>
        <div><strong>Status:</strong> {{ systemStatus }}</div>
      </div>
    </section>

    <!-- GPU Diagnostics -->
    <section class="section-inline muted">
      <details>
        <summary>GPU Diagnostics</summary>
        <pre class="code-block">{{ gpuInfo }}</pre>
      </details>
    </section>
  </main>
</template>

<script setup>
import { onMounted } from 'vue'
import './styles.css'
import {
  // Constants
  apiBase,
  
  // Reactive state
  text,
  voices,
  selectedLanguage,
  selectedVoice,
  audioUrl,
  filename,
  lastUsedVoice,
  loading,
  refreshing,
  errorMessage,
  systemStatus,
  gpuInfo,
  availableLanguages,
  speed,
  
  // Zero-shot helpers
  fileInput,
  showTooltip,
  zeroShotActive,
  zeroShotName,
  
  // Computed
  availableLanguagesForVoice,
  isReadyForSynthesis,
  
  // Functions
  refreshVoices,
  getLanguageDisplayName,
  triggerFileDialog,
  handleFileSelected,
  clearZeroShot,
  synthesize,
  initializeApp
} from './composables.js'

// Initialize the app when component mounts
onMounted(initializeApp)
</script>
